import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

from codex_agent import remote_job as job


class RemoteJobTests(unittest.TestCase):
    def setUp(self):
        parent = Path("/tmp/triton-riscv-agent")
        parent.mkdir(exist_ok=True)
        self.stage = parent / uuid.uuid4().hex
        self.stage.mkdir(mode=0o700)
        (self.stage / "execution/payload").mkdir(parents=True)
        self.tools = tempfile.TemporaryDirectory()
        fake_timeout = Path(self.tools.name) / "timeout"
        fake_timeout.write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
        fake_timeout.chmod(0o700)
        # This fixture tests persistence/dispatch only, not GNU timeout or sandbox.
        self.environment = patch.dict(os.environ, {"PATH": self.tools.name + ":" + os.environ["PATH"]})
        self.environment.start()
        script = self.stage / "validation.sh"
        script.write_text('printf x >> "$2/../counter"\nsleep 0.2\nprintf "1 passed\\n"\nexit 0\n')
        self.request = {"version": 1, "job_id": self.stage.name, "repository": "/fixture",
                        "timeout_seconds": 10, "files": [], "command": ["python", "-m", "pytest"],
                        "input_sha256": {"validation.sh": job.file_digest(script)}}
        job.atomic_json(self.stage / "request.json", self.request)
        self.key = job.digest(self.request)

    def tearDown(self):
        self.environment.stop()
        self.tools.cleanup()
        shutil.rmtree(self.stage, ignore_errors=True)

    def finished(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = job.inspect_job(self.stage, self.key)
            if result["state"] == "completed":
                return result
            time.sleep(0.05)
        self.fail("fixture worker did not finish")

    def test_request_process_can_exit_and_duplicate_start_never_reruns(self):
        command = [sys.executable, "-I", str(Path(job.__file__).resolve()), "start", str(self.stage), self.key]
        first = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn(json.loads(first.stdout)["state"], {"accepted", "queued", "running", "completed"})
        second = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(second.returncode, 0, second.stderr)
        terminal = self.finished()
        self.assertEqual(terminal["exit_code"], 0)
        self.assertEqual(terminal["admission"]["state"], "admitted")
        self.assertEqual((self.stage / "counter").read_text(), "x")
        collected = job.inspect_job(self.stage, self.key, include_log=True)
        self.assertEqual(collected["log_sha256"], job.file_digest(self.stage / "output.log"))
        self.assertIn("log_base64", collected)
        again = job.start_job(self.stage, self.key)
        self.assertEqual(again, terminal)
        self.assertEqual((self.stage / "counter").read_text(), "x")

    def test_crash_without_terminal_evidence_is_unknown_and_cannot_restart(self):
        job.atomic_json(self.stage / "dispatch.json", {"created_at": time.time()})
        (self.stage / "worker-claimed").write_text(self.key)
        self.assertEqual(job.start_job(self.stage, self.key)["state"], "unknown")
        self.assertFalse((self.stage / "counter").exists())

    def test_absent_job_or_wrong_identity_is_never_assumed_success(self):
        self.assertEqual(job.inspect_job(self.stage, self.key)["state"], "not-started")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            job.inspect_job(self.stage, "wrong")
        (self.stage / "request.json").unlink()
        with self.assertRaises(FileNotFoundError):
            job.inspect_job(self.stage, self.key)

    def test_input_change_is_rejected_before_execution(self):
        (self.stage / "validation.sh").write_text("echo tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            job.run_worker(self.stage, self.key)
        self.assertFalse((self.stage / "counter").exists())
        self.assertEqual(job.inspect_job(self.stage, self.key)["state"], "unknown")

    def test_real_failing_command_still_has_a_completed_nonzero_result(self):
        script = self.stage / "validation.sh"
        script.write_text('printf "1 failed\\n"\nexit 1\n')
        self.request["input_sha256"]["validation.sh"] = job.file_digest(script)
        self.key = job.digest(self.request)
        job.atomic_json(self.stage / "request.json", self.request)
        result = job.run_worker(self.stage, self.key)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(job.inspect_job(self.stage, self.key)["exit_code"], 1)

    def test_corrupt_log_and_terminal_identity_are_rejected(self):
        job.start_job(self.stage, self.key)
        terminal = self.finished()
        (self.stage / "output.log").write_text("fabricated pass")
        with self.assertRaisesRegex(ValueError, "log does not match"):
            job.inspect_job(self.stage, self.key, include_log=True)
        terminal["job_id"] = "wrong"
        job.atomic_json(self.stage / "result.json", terminal)
        with self.assertRaisesRegex(ValueError, "different request"):
            job.inspect_job(self.stage, self.key)

    def test_ack_requires_exact_collected_log_digest(self):
        job.start_job(self.stage, self.key)
        terminal = self.finished()
        command = [sys.executable, "-I", str(Path(job.__file__).resolve()), "ack", str(self.stage), self.key]
        rejected = subprocess.run([*command, "wrong"], capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertTrue(self.stage.exists())
        accepted = subprocess.run([*command, terminal["log_sha256"]], capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stdout)
        self.assertFalse(self.stage.exists())

    def test_queue_timeout_is_terminal_without_running_test_or_claiming_compile(self):
        from contextlib import contextmanager
        from codex_agent import remote_capacity
        @contextmanager
        def unavailable(*args, **kwargs):
            self.assertEqual(job.inspect_job(self.stage, self.key)["state"], "queued")
            raise remote_capacity.CapacityUnavailable("busy")
            yield
        with patch.object(remote_capacity.CapacityPool, "acquire", unavailable):
            result = job.run_worker(self.stage, self.key)
        self.assertEqual(result["exit_code"], 81)
        self.assertEqual(result["admission"]["state"], "not-admitted")
        self.assertFalse((self.stage / "counter").exists())
        self.assertIn("CAPACITY_ERROR", (self.stage / "output.log").read_text())
        self.assertEqual(job.inspect_job(self.stage, self.key)["state"], "completed")

    def test_cancel_before_start_is_durable_and_repeated_start_cannot_execute(self):
        result = job.request_cancel(self.stage, self.key)
        self.assertTrue(result["cancellation"]["confirmed"])
        self.assertEqual(result["exit_code"], 130)
        self.assertEqual(job.start_job(self.stage, self.key), result)
        self.assertEqual(job.request_cancel(self.stage, self.key), result)
        self.assertFalse((self.stage / "counter").exists())

    def test_cancel_wrong_identity_is_rejected_before_writing_marker(self):
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            job.request_cancel(self.stage, "wrong")
        self.assertFalse((self.stage / "cancel.json").exists())

    def test_cancel_after_completion_preserves_the_original_result(self):
        original = job.run_worker(self.stage, self.key)
        self.assertEqual(job.request_cancel(self.stage, self.key), original)
        self.assertEqual(original["exit_code"], 0)
        self.assertFalse((self.stage / "cancel.json").exists())

    def test_cancel_while_queued_does_not_start_candidate(self):
        job.atomic_json(self.stage / "dispatch.json", {"created_at": time.time()})
        job.atomic_json(self.stage / "cancel.json", {"job_id": self.stage.name, "request_digest": self.key})
        result = job.run_worker(self.stage, self.key)
        self.assertTrue(result["cancellation"]["confirmed"])
        self.assertEqual(result["admission"]["state"], "cancelled-before-start")
        self.assertFalse((self.stage / "counter").exists())
