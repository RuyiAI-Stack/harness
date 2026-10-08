import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codex_agent import resource_probe as probe
from codex_agent import remote_executor as remote


class ResourceProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        (self.proc / "self").mkdir(parents=True)
        (self.proc / "self/cgroup").write_text("4:cpu,cpuacct:/system.slice/sshd.service\n")
        (self.proc / "cgroups").write_text("#subsys_name hierarchy num_cgroups enabled\ncpu 4 44 1\ncpuacct 4 44 1\n")
        (self.proc / "self/mountinfo").write_text("55 51 0:33 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n")

    def snapshot(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file()}

    def test_shared_v1_without_memory_is_not_resource_isolation(self):
        before = self.snapshot()
        result = probe.probe(self.proc)
        self.assertEqual(result["probe_status"], "observed")
        self.assertEqual(result["cgroup_mode"], "v1")
        self.assertTrue(result["cpu_controller_visible"])
        self.assertFalse(result["memory_controller_visible"])
        self.assertFalse(result["aggregate_cpu_quota"])
        self.assertFalse(result["aggregate_memory_quota"])
        self.assertEqual(result["memberships"][0]["path"], "/system.slice/sshd.service")
        self.assertEqual(before, self.snapshot())

    def test_disabled_memory_is_not_available(self):
        (self.proc / "cgroups").write_text("cpu 4 44 1\nmemory 0 1 0\n")
        self.assertFalse(probe.probe(self.proc)["memory_controller_visible"])

    def test_even_visible_v1_memory_does_not_prove_enforcement(self):
        (self.proc / "cgroups").write_text("cpu 4 44 1\nmemory 5 1 1\n")
        result = probe.probe(self.proc)
        self.assertTrue(result["memory_controller_visible"])
        self.assertFalse(result["aggregate_memory_quota"])

    def test_writable_v2_with_mount_root_offset_is_only_observation(self):
        group = self.root / "mount with space/task"
        group.mkdir(parents=True)
        (group / "cgroup.controllers").write_text("cpu memory pids\n")
        mount = str(group.parent).replace(" ", r"\040")
        (self.proc / "self/mountinfo").write_text(f"50 40 0:32 /delegated {mount} rw - cgroup2 cgroup rw\n")
        (self.proc / "self/cgroup").write_text("0::/delegated/task\n")
        before = self.snapshot()
        result = probe.probe(self.proc)
        self.assertEqual(result["probe_status"], "observed")
        self.assertTrue(result["memory_controller_visible"])
        self.assertEqual(result["current_group"]["path"], str(group))
        self.assertTrue(result["current_group"]["directory_writable_observed"])
        self.assertFalse(result["current_group"]["delegation_verified"])
        self.assertFalse(result["aggregate_memory_quota"])
        self.assertEqual(before, self.snapshot())

    def test_unresolved_v2_is_unknown_not_missing_memory(self):
        (self.proc / "self/mountinfo").write_text("50 40 0:32 /other /no-such-mount rw - cgroup2 cgroup rw\n")
        (self.proc / "self/cgroup").write_text("0::/delegated/task\n")
        result = probe.probe(self.proc)
        self.assertEqual(result["probe_status"], "unknown")
        self.assertIsNone(result["memory_controller_visible"])

    def test_missing_proc_is_unknown(self):
        result = probe.probe(self.root / "missing")
        self.assertEqual(result["probe_status"], "unknown")
        self.assertIsNone(result["memory_controller_visible"])

    def test_bad_empty_or_oversized_inputs_do_not_certify_absence(self):
        for value in ("", "broken", "1:cpu:/../other\n", "x" * (probe.MAX_BYTES + 1)):
            with self.subTest(value=value[:20]):
                (self.proc / "self/cgroup").write_text(value)
                result = probe.probe(self.proc)
                self.assertEqual(result["probe_status"], "unknown")
                self.assertIsNone(result["memory_controller_visible"])

    def test_missing_duplicate_and_bad_transport_evidence_are_unknown(self):
        row = probe.PREFIX + json.dumps(probe.probe(self.proc))
        for text in ("", row + "\n" + row, probe.PREFIX + "[]", probe.PREFIX + "bad"):
            with self.subTest(text=text[:30]):
                self.assertEqual(probe.decode(text)["probe_status"], "unknown")

    def test_transport_never_elevates_probe_to_enforcement(self):
        result = probe.probe(self.proc)
        result.update(aggregate_cpu_quota=True, aggregate_memory_quota=True, quota_enforcement="enabled")
        decoded = probe.decode(probe.PREFIX + json.dumps(result))
        self.assertFalse(decoded["aggregate_cpu_quota"])
        self.assertFalse(decoded["aggregate_memory_quota"])
        self.assertEqual(decoded["quota_enforcement"], "not-implemented")

    def test_strict_policy_blocks_even_if_controller_is_present(self):
        for value in ("1", "true", "TRUE"):
            with self.subTest(value=value):
                self.assertIn("no verified", probe.quota_policy_error({probe.REQUIRE_ENV: value}))

    def test_compatibility_default_is_explicit(self):
        for value in ("0", "false", "", "FALSE"):
            with self.subTest(value=value):
                self.assertIsNone(probe.quota_policy_error({probe.REQUIRE_ENV: value}))

    def test_invalid_policy_never_silently_disables_requirement(self):
        self.assertIn("must be", probe.quota_policy_error({probe.REQUIRE_ENV: "tru"}))

    def test_preflight_exposes_readonly_probe_evidence(self):
        output = ("ARCHITECTURE=riscv64\nPYTHON_PATH=/repo/.venv/bin/python\n"
                  "TRITON_VERSION=3.4.0\nTRITON_SHARED_OPT=/bin/shared\nBUDDY_OPT=/bin/buddy\n" +
                  probe.PREFIX + json.dumps(probe.probe(self.proc)))
        with patch.dict(os.environ, {probe.REQUIRE_ENV: "0"}), patch.object(remote, "_run_ssh_script",
                return_value=subprocess.CompletedProcess([], 0, output)):
            result = remote.check_remote_environment(remote.RemoteValidationConfig("host", "/repo"))
        self.assertEqual(result.status, "passed")
        self.assertFalse(result.resource_capabilities["aggregate_memory_quota"])
        self.assertEqual(result.resource_capabilities["probe_status"], "observed")
        self.assertIn('python -I -', remote.PREFLIGHT_SCRIPT)
        self.assertIn(Path(probe.__file__).read_text(), remote.PREFLIGHT_SCRIPT)

    def test_strict_preflight_never_connects(self):
        with patch.dict(os.environ, {probe.REQUIRE_ENV: "1"}), patch.object(remote, "_run_ssh_script") as ssh:
            result = remote.check_remote_environment(remote.RemoteValidationConfig("host", "/repo"))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.exit_code, 82)
        ssh.assert_not_called()

    def test_new_dispatch_is_blocked_before_source_upload(self):
        operator = {"name": "demo", "implementation_file": "demo.py", "test_files": []}
        with patch.dict(os.environ, {probe.REQUIRE_ENV: "1"}), \
             patch.object(remote, "check_remote_environment") as check, \
             patch.object(remote, "_run_ssh_script") as ssh, patch.object(remote, "run_bounded") as upload:
            result, preflight = remote.run_remote_operator(operator, repo_root=self.root,
                results_dir=self.root / "result", timeout_seconds=10,
                config=remote.RemoteValidationConfig("host", "/repo"))
        self.assertEqual(result.failure_stage, "environment")
        self.assertFalse(result.diagnosis["repairable"])
        self.assertFalse((self.root / "result").exists())
        self.assertEqual(preflight.status, "blocked")
        check.assert_not_called()
        ssh.assert_not_called()
        upload.assert_not_called()

    def test_strict_mode_still_allows_observing_original_job(self):
        folder = self.root / "python/examples/flaggems"
        folder.mkdir(parents=True)
        (folder / "demo.py").write_text("pass\n")
        (folder / "test_demo.py").write_text("def test_demo(): pass\n")
        operator = {"name": "demo", "implementation_file": "python/examples/flaggems/demo.py",
                    "test_files": ["python/examples/flaggems/test_demo.py"],
                    "test_nodes": ["python/examples/flaggems/test_demo.py::test_demo"]}
        job = {"host": "host", "repository": "/repo", "job_id": "a" * 32,
               "stage": "/tmp/triton-riscv-agent/" + "a" * 32, "request_digest": "b" * 64,
               "preflight": remote.RemotePreflightResult(configured=True, status="passed").model_dump()}
        with patch.dict(os.environ, {probe.REQUIRE_ENV: "1"}), \
             patch("codex_agent.execution_guard.execution_kind", return_value="validation"), \
             patch("codex_agent.execution_guard.execution_details", return_value={"remote_job": job}), \
             patch.object(remote, "_collect_job", side_effect=RuntimeError("observation reached")) as collect, \
             patch.object(remote, "check_remote_environment") as check, \
             patch.object(remote, "run_bounded") as upload:
            with self.assertRaisesRegex(RuntimeError, "observation reached"):
                remote.run_remote_operator(operator, repo_root=self.root, results_dir=self.root / "result",
                    timeout_seconds=10, config=remote.RemoteValidationConfig("host", "/repo"))
        collect.assert_called_once()
        check.assert_not_called()
        upload.assert_not_called()
