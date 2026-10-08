from __future__ import annotations

import subprocess
import os
import sys
import base64
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.remote_executor import (
    RemoteValidationConfig,
    RemotePreflightResult,
    run_remote_operator,
    PREFLIGHT_SCRIPT,
    VALIDATION_SCRIPT,
    _cleanup_stage,
    _run_ssh_script,
    _validated_relative_file,
    check_remote_environment,
)


class RemoteExecutorTests(unittest.TestCase):
    def test_transport_retry_keeps_job_identity_and_never_dispatches(self):
        from codex_agent.remote_executor import _job_rpc
        import json
        job = {"job_id": "a" * 32, "stage": "/tmp/triton-riscv-agent/" + "a" * 32,
               "request_digest": "b" * 64, "host": "host", "repository": "/repo"}
        response = {"job_id": job["job_id"], "request_digest": job["request_digest"], "state": "running"}
        with patch("codex_agent.remote_retry.RETRY_DELAYS", (0, 0)), patch(
            "codex_agent.remote_executor._run_ssh_script", side_effect=[
                subprocess.CompletedProcess([], 255, "connection dropped"),
                subprocess.CompletedProcess([], 0, json.dumps(response)),
            ]) as ssh:
            result = _job_rpc(RemoteValidationConfig("host", "/repo"), job, "inspect")
        self.assertEqual(result, response)
        self.assertEqual(ssh.call_args_list[0], ssh.call_args_list[1])
        self.assertEqual(job["transport_retry_count"], 1)

    def test_incomplete_recovery_handle_cannot_turn_into_a_new_job(self):
        with patch("codex_agent.execution_guard.execution_kind", return_value="validation"), \
             patch("codex_agent.execution_guard.execution_details", return_value={"remote_job": {}}), \
             patch("codex_agent.remote_executor.check_remote_environment") as preflight, \
             patch("codex_agent.remote_executor.run_bounded") as upload:
            with self.assertRaisesRegex(PermissionError, "incomplete"):
                run_remote_operator({}, repo_root=Path("/fixture"), results_dir=Path("/fixture"),
                                    timeout_seconds=10, config=RemoteValidationConfig("host", "/repo"))
            preflight.assert_not_called()
            upload.assert_not_called()

    def test_batch_records_jobs_separately_without_single_operator_resume_handle(self):
        from codex_agent.remote_executor import _record_job
        first = {"job_id": "first", "host": "host", "stage": "/tmp/first"}
        second = {"job_id": "second", "host": "host", "stage": "/tmp/second"}
        with patch("codex_agent.execution_guard.execution_kind", return_value="job"), \
             patch("codex_agent.execution_guard.execution_details", return_value={"remote_jobs": {"first": first}}), \
             patch("codex_agent.execution_guard.record_execution_detail") as record:
            _record_job(second)
        fields = record.call_args.kwargs
        self.assertEqual(fields["remote_jobs"], {"first": first, "second": second})
        self.assertNotIn("remote_job", fields)

    def test_wrong_remote_identity_and_corrupt_log_are_not_accepted(self):
        from codex_agent.remote_executor import _job_rpc, _collect_job
        import json
        job = {"job_id": "a" * 32, "stage": "/tmp/triton-riscv-agent/" + "a" * 32,
               "request_digest": "b" * 64}
        config = RemoteValidationConfig("host", "/repo")
        with patch("codex_agent.remote_executor._run_ssh_script", return_value=subprocess.CompletedProcess(
            [], 0, json.dumps({"job_id": "wrong", "request_digest": job["request_digest"], "state": "completed"})
        )):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _job_rpc(config, job, "inspect")
        with patch("codex_agent.remote_executor._job_rpc", return_value={
            "state": "completed", "log_base64": base64.b64encode(b"fake pass").decode(),
            "log_bytes": 9, "log_sha256": "bad", "exit_code": 0,
        }):
            with self.assertRaisesRegex(RuntimeError, "evidence mismatch"):
                _collect_job(config, job, 10)

    def test_queue_is_observed_not_restarted_or_treated_as_operator_failure(self):
        from codex_agent.remote_executor import _collect_job
        raw = b"CAPACITY_ERROR=busy\n"
        terminal = {"state": "completed", "log_base64": base64.b64encode(raw).decode(),
                    "log_bytes": len(raw), "log_sha256": hashlib.sha256(raw).hexdigest(), "exit_code": 81}
        with patch("codex_agent.remote_executor._job_rpc", side_effect=[
            {"state": "queued"}, {"state": "running"}, terminal, terminal
        ]) as rpc, patch("codex_agent.remote_executor._record_job") as record, \
             patch("codex_agent.remote_executor.time.sleep"):
            result, log = _collect_job(RemoteValidationConfig("host", "/repo"), {}, 10)
        self.assertEqual(result["exit_code"], 81)
        self.assertEqual(log, raw)
        self.assertEqual([call.args[2] for call in rpc.call_args_list], ["inspect", "inspect", "inspect", "collect"])
        self.assertEqual([call.args[0]["observed_state"] for call in record.call_args_list], ["queued", "running"])

    def test_remote_configuration_must_be_complete_and_safe(self) -> None:
        self.assertIsNone(RemoteValidationConfig.from_env({}))
        with self.assertRaisesRegex(ValueError, "configured together"):
            RemoteValidationConfig.from_env({"RISCV_HOST": "sg2044"})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            RemoteValidationConfig.from_env(
                {"RISCV_HOST": "sg2044;whoami", "RISCV_REPO": "/home/work/repo"}
            )
        config = RemoteValidationConfig.from_env(
            {"RISCV_HOST": "sg2044", "RISCV_REPO": "/home/work/triton-riscv"}
        )
        assert config is not None
        self.assertEqual(config.host, "sg2044")

    def test_sync_is_restricted_to_flaggems_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "python/examples/flaggems/demo.py"
            allowed.parent.mkdir(parents=True)
            allowed.write_text("# demo\n", encoding="utf-8")

            self.assertEqual(
                _validated_relative_file(root, "python/examples/flaggems/demo.py"),
                allowed.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "only FlagGems"):
                _validated_relative_file(root, "scripts/demo.py")
            with self.assertRaisesRegex(ValueError, "escaped"):
                _validated_relative_file(root, "../demo.py")

    def test_preflight_parses_required_remote_tools(self) -> None:
        config = RemoteValidationConfig("sg2044", "/home/work/triton-riscv")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "ARCHITECTURE=riscv64\n"
                "PYTHON_PATH=/home/work/.venv/bin/python\n"
                "TRITON_VERSION=3.4.0\n"
                "TRITON_SHARED_OPT=/home/work/.venv/bin/triton-shared-opt\n"
                "BUDDY_OPT=/home/work/buddy-mlir/build/bin/buddy-opt\n"
            ),
        )
        with patch(
            "codex_agent.remote_executor._run_ssh_script",
            return_value=completed,
        ):
            result = check_remote_environment(config)

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.architecture, "riscv64")
        self.assertEqual(result.triton_version, "3.4.0")
        self.assertTrue(result.buddy_opt)

    def test_preflight_rejects_empty_tools_even_with_zero_exit(self):
        with patch("codex_agent.remote_executor._run_ssh_script", return_value=
                   subprocess.CompletedProcess([], 0, "ARCHITECTURE=riscv64\n")):
            result = check_remote_environment(RemoteValidationConfig("host", "/repo"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.exit_code, 78)

    def test_preflight_missing_ssh_is_structured_failure(self):
        with patch("codex_agent.remote_executor._run_ssh_script", side_effect=FileNotFoundError("ssh")):
            result = check_remote_environment(RemoteValidationConfig("host", "/repo"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.exit_code, 127)

    def test_ssh_arguments_are_quoted_and_noninteractive(self):
        with patch("codex_agent.remote_executor.run_bounded") as run:
            _run_ssh_script(RemoteValidationConfig("host", "/repo"), "script", ["x;touch bad", "a b"], 1)
        argv = run.call_args.args[0]
        self.assertIn("BatchMode=yes", argv)
        self.assertEqual(argv[-1], "bash -s -- 'x;touch bad' 'a b'")

    def test_cleanup_does_not_delete_an_active_workspace(self):
        with patch("codex_agent.remote_executor._run_ssh_script") as run:
            _cleanup_stage(RemoteValidationConfig("host", "/repo"), "/tmp/triton-riscv-agent/a")
        self.assertIn('test -f "$1/started" ||', run.call_args.args[1])

    def test_result_records_fresh_cache_and_unique_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "python/examples/flaggems"
            folder.mkdir(parents=True)
            (folder / "demo.py").write_text("x = 1\n")
            (folder / "test_demo.py").write_text("def test_demo(): pass\n")
            operator = dict(name="demo", implementation_file="python/examples/flaggems/demo.py",
                            test_files=["python/examples/flaggems/test_demo.py"],
                            test_nodes=["python/examples/flaggems/test_demo.py::test_demo"])
            completed = subprocess.CompletedProcess([], 0, "1 passed in 0.1s\n")
            def rpc(config, job, action, *args):
                raw = completed.stdout.encode()
                return {"state": "completed", "job_id": job["job_id"], "request_digest": job["request_digest"],
                        "exit_code": 0, "duration_seconds": 0.1, "log_bytes": len(raw),
                        "log_sha256": hashlib.sha256(raw).hexdigest(), "log_base64": base64.b64encode(raw).decode()}
            with patch("codex_agent.remote_executor.check_remote_environment", return_value=
                       RemotePreflightResult(configured=True, status="passed", architecture="riscv64")), \
                 patch("codex_agent.remote_executor._run_ssh_script", return_value=completed), \
                 patch("codex_agent.remote_executor.run_bounded", return_value=completed), \
                 patch("codex_agent.remote_executor._job_rpc", side_effect=rpc):
                results = [run_remote_operator(operator, repo_root=root, results_dir=root / "results",
                           timeout_seconds=10, config=RemoteValidationConfig("host", "/repo"))[0]
                           for _ in range(2)]
            self.assertTrue(all(item.fresh_compile and item.status == "passed" for item in results))
            self.assertNotEqual(results[0].log_path, results[1].log_path)
            self.assertEqual(results[0].isolation["enforcement"], "required")
            self.assertIn("rootless-namespace", results[0].command)

            with patch("codex_agent.remote_executor.check_remote_environment", return_value=
                       RemotePreflightResult(configured=True, status="passed", architecture="riscv64")), \
                 patch("codex_agent.remote_executor._run_ssh_script", return_value=completed), \
                 patch("codex_agent.remote_executor.run_bounded", return_value=completed), \
                 patch("codex_agent.remote_executor._job_rpc", side_effect=rpc), \
                 patch("codex_agent.remote_executor._collect_job", return_value=(
                     {"exit_code": 81, "duration_seconds": 30, "log_sha256": "fixture",
                      "admission": {"state": "not-admitted"}}, b"CAPACITY_ERROR=busy\n")):
                blocked, _ = run_remote_operator(operator, repo_root=root, results_dir=root / "results",
                           timeout_seconds=10, config=RemoteValidationConfig("host", "/repo"))
            self.assertEqual(blocked.failure_stage, "capacity")
            self.assertFalse(blocked.fresh_compile)
            self.assertFalse(blocked.diagnosis["repairable"])

            with patch("codex_agent.remote_executor.check_remote_environment", return_value=
                       RemotePreflightResult(configured=True, status="passed", architecture="riscv64")), \
                 patch("codex_agent.remote_executor._run_ssh_script", return_value=completed), \
                 patch("codex_agent.remote_executor.run_bounded", return_value=completed), \
                 patch("codex_agent.remote_executor._job_rpc", side_effect=rpc), \
                 patch("codex_agent.remote_executor._collect_job", return_value=(
                     {"exit_code": 130, "duration_seconds": 1, "log_sha256": "fixture",
                      "cancellation": {"confirmed": True}}, b"VALIDATION_CANCELLED=stopped\n")):
                cancelled, _ = run_remote_operator(operator, repo_root=root, results_dir=root / "results",
                           timeout_seconds=10, config=RemoteValidationConfig("host", "/repo"))
            self.assertEqual(cancelled.status, "cancelled")
            self.assertEqual(cancelled.failure_stage, "cancellation")
            self.assertFalse(cancelled.fresh_compile)
            self.assertFalse(cancelled.diagnosis["repairable"])

    def test_preflight_import_failure_cannot_be_hidden_by_printf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (".venv/bin", "scripts", "fake"):
                (root / name).mkdir(parents=True)
            (root / ".venv/bin/activate").write_text("")
            (root / "scripts/triton-riscv-env.sh").write_text("")
            for name, body in {"uname": "echo riscv64", "python": "exit 1"}.items():
                executable = root / "fake" / name
                executable.write_text("#!/bin/sh\n" + body + "\n")
                executable.chmod(0o700)
            result = subprocess.run(["bash", "-s", "--", str(root)], input=PREFLIGHT_SCRIPT,
                                    capture_output=True, text=True,
                                    env={**os.environ, "PATH": f"{root}/fake:{os.environ['PATH']}"})
            self.assertEqual(result.returncode, 75)
            self.assertIn("python-dependency-import-failed", result.stdout)

    def test_remote_payload_has_its_own_deadline_when_ssh_disconnects(self):
        from codex_agent.process_control import execution_budget
        with execution_budget(100), patch("codex_agent.remote_executor.run_bounded") as run:
            _run_ssh_script(RemoteValidationConfig("host", "/repo"), VALIDATION_SCRIPT, [], 300)
        remote_command = run.call_args.args[0][-1]
        self.assertIn("timeout --signal=TERM --kill-after=10s", remote_command)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 100)

    def test_disconnect_is_unknown_not_an_operator_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "python/examples/flaggems"
            folder.mkdir(parents=True)
            (folder / "demo.py").write_text("x = 1")
            (folder / "test_demo.py").write_text("def test_demo(): pass")
            operator = dict(name="demo", implementation_file="python/examples/flaggems/demo.py",
                            test_files=["python/examples/flaggems/test_demo.py"],
                            test_nodes=["python/examples/flaggems/test_demo.py::test_demo"])
            def ssh(config, script, args, timeout):
                return subprocess.CompletedProcess([], 255 if "remote_job.py" in script else 0, "connection lost")
            with patch("codex_agent.remote_executor.check_remote_environment", return_value=
                       RemotePreflightResult(configured=True, status="passed", architecture="riscv64")), \
                 patch("codex_agent.remote_executor._run_ssh_script", side_effect=ssh), \
                 patch("codex_agent.remote_executor.run_bounded", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(RuntimeError, "Remote outcome unknown"):
                    run_remote_operator(operator, repo_root=root, results_dir=root / "results",
                                        timeout_seconds=10, config=RemoteValidationConfig("host", "/repo"))
            self.assertEqual(len(list((root / "results/remote-jobs").glob("*.json"))), 1)
            self.assertFalse((root / "results/logs").exists())

    def test_real_shell_snapshot_success_failure_and_symlink_rejection(self):
        # Exercise the actual SSH payload locally; no SSH, compiler, or model.
        for mode in ("success", "failure", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in (".venv/bin", "scripts", "python/examples/flaggems", "fake"):
                    (root / name).mkdir(parents=True)
                (root / ".venv/bin/activate").write_text("")
                (root / "scripts/triton-riscv-env.sh").write_text("")
                original = root / "python/examples/flaggems/demo.py"
                original.write_text("ORIGINAL\n")
                if mode == "symlink":
                    (root / "python/examples/flaggems/link.py").symlink_to(original)
                (root / "fake/python").symlink_to(sys.executable)
                timeout = root / "fake/timeout"
                timeout.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
                timeout.chmod(0o700)
                parent = Path("/tmp/triton-riscv-agent")
                parent.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir=parent) as stage_name:
                    stage = Path(stage_name)
                    (stage / "payload").mkdir()
                    (stage / "payload/0").write_text("NEW\n")
                    # Only test shell snapshot orchestration on macOS. Real
                    # namespace enforcement is a separate Linux acceptance test.
                    (stage / "sandbox.py").write_text(
                        "import subprocess,sys\nraise SystemExit(subprocess.call(sys.argv[sys.argv.index('--')+1:]))\n"
                    )
                    check = ("from pathlib import Path; "
                             "assert Path('python/examples/flaggems/demo.py').read_text() == 'NEW\\n'; "
                             f"raise SystemExit({1 if mode == 'failure' else 0})")
                    result = subprocess.run(["bash", "-s", "--", str(root), stage_name, "10", "1",
                                             "python/examples/flaggems/demo.py", "python", "-c", check],
                                            input=VALIDATION_SCRIPT, text=True, capture_output=True,
                                            env={**os.environ, "PATH": f"{root}/fake:{os.environ['PATH']}"})
                    self.assertEqual(result.returncode == 0, mode == "success", result.stderr)
                    self.assertEqual(original.read_text(), "ORIGINAL\n")
                    self.assertFalse(stage.exists(), result.stderr)

    def test_missing_launcher_cannot_fall_back_to_pytest(self):
        self.assertIn('test -f "$stage/sandbox.py"', VALIDATION_SCRIPT)
        self.assertIn('python -I "$stage/sandbox.py"', VALIDATION_SCRIPT)
        self.assertNotIn('"${test_timeout}s" "${command[@]}"', VALIDATION_SCRIPT)

    def test_launcher_upload_failure_never_starts_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "python/examples/flaggems"
            folder.mkdir(parents=True)
            (folder / "demo.py").write_text("x = 1")
            (folder / "test_demo.py").write_text("def test_demo(): pass")
            operator = dict(name="demo", implementation_file="python/examples/flaggems/demo.py",
                test_files=["python/examples/flaggems/test_demo.py"],
                test_nodes=["python/examples/flaggems/test_demo.py::test_demo"])
            with patch("codex_agent.remote_executor.check_remote_environment", return_value=
                    RemotePreflightResult(configured=True, status="passed", architecture="riscv64")), \
                 patch("codex_agent.remote_executor._run_ssh_script", return_value=subprocess.CompletedProcess([], 0, "")) as ssh, \
                 patch("codex_agent.remote_executor.run_bounded", return_value=subprocess.CompletedProcess([], 1, "transfer failed")):
                with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                    run_remote_operator(operator, repo_root=root, results_dir=root / "results",
                                        timeout_seconds=10, config=RemoteValidationConfig("host", "/repo"))
            self.assertFalse(any("remote_job.py" in call.args[1] for call in ssh.call_args_list))


if __name__ == "__main__":
    unittest.main()
