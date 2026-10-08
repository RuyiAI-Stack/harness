"""Deterministic fault fixtures, not operator correctness or live-server evidence."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

from codex_agent import remote_capacity as capacity, remote_job as job


class CapacityRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "slots"
        self.pool = capacity.CapacityPool(self.root, limit=1)
        parent = Path("/tmp/triton-riscv-agent")
        parent.mkdir(exist_ok=True)
        self.stage = parent / uuid.uuid4().hex
        self.stage.mkdir(mode=0o700)
        self.request = {"job_id": self.stage.name, "version": 1, "timeout_seconds": 5}
        self.key = job.digest(self.request)
        job.atomic_json(self.stage / "request.json", self.request)
        with capacity._open_lock(self.stage / "worker.lock"):
            pass
        self.server = patch.object(capacity, "server_root", return_value=self.root)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        shutil.rmtree(self.stage, ignore_errors=True)
        self.temp.cleanup()

    def terminal(self, admission, code=0):
        (self.stage / "output.log").write_text("unit fixture: command finished\n")
        return job.save_result(self.stage, self.request, self.key, code, time.monotonic(), admission,
                               shutdown={"version": 1, "descendants_stopped": True,
                                         "method": "linux-subreaper-pidfd"})

    def fenced(self, *, code=0, terminal=True):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with self.pool.acquire(self.stage.name, request_digest=self.key) as admission:
                if terminal:
                    self.terminal(admission, code)
                raise RuntimeError("injected supervisor exit after terminal write")
        return admission

    def inspect(self):
        return job.capacity_operation(self.stage, self.key)

    def release(self, token):
        return job.capacity_operation(self.stage, self.key, token)

    def test_readonly_absent_and_uninitialized_slots_create_no_files(self):
        missing = Path(self.temp.name) / "missing"
        self.assertEqual(capacity.inspect_existing(missing), [])
        self.assertFalse(missing.exists())
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(self.inspect()["state"], "no-reservation")
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)

    def test_exact_finished_task_releases_without_deleting_locks_or_test_evidence(self):
        self.fenced()
        inode = (self.root / "slot-0.lock").stat().st_ino
        original = (self.stage / "result.json").read_bytes()
        preview = self.inspect()
        self.assertTrue(preview["releasable"])
        self.assertEqual(self.release(preview["release_token"])["state"], "released")
        self.assertEqual(capacity.inspect_existing(self.root)[0]["state"], "free")
        self.assertEqual((self.root / "slot-0.lock").stat().st_ino, inode)
        self.assertEqual((self.stage / "result.json").read_bytes(), original)

    def test_no_terminal_means_blocked_even_if_slot_lock_is_free(self):
        self.fenced(terminal=False)
        self.assertEqual(self.inspect()["state"], "blocked")
        self.assertIn("no terminal", self.inspect()["reason"])
        self.assertNotIn("release_token", self.inspect())

    def test_active_slot_cannot_be_released_even_with_terminal_written(self):
        with self.pool.acquire(self.stage.name, request_digest=self.key) as admission:
            self.terminal(admission)
            self.assertFalse(self.inspect()["releasable"])
            record = json.loads((self.root / "slot-0.json").read_text())
            with self.assertRaises(BlockingIOError):
                self.release(capacity.reservation_token(0, record))

    def test_active_worker_prevents_release_of_a_fenced_slot(self):
        self.fenced()
        with capacity._open_lock(self.stage / "worker.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.assertFalse(self.inspect()["releasable"])

    def test_legacy_shutdown_record_cannot_be_inferred_from_exit_zero(self):
        self.fenced()
        result = json.loads((self.stage / "result.json").read_text())
        result.pop("shutdown")
        job.atomic_json(self.stage / "result.json", result)
        self.assertIn("shutdown proof", self.inspect()["reason"])

    def test_wrong_admission_or_unsupported_shutdown_are_rejected(self):
        self.fenced()
        original = json.loads((self.stage / "result.json").read_text())
        for field in ("reservation_id", "slot", "descendants_stopped", "method"):
            result = json.loads(json.dumps(original))
            target = result["admission"] if field in {"reservation_id", "slot"} else result["shutdown"]
            target[field] = "wrong"
            job.atomic_json(self.stage / "result.json", result)
            with self.subTest(field=field):
                self.assertFalse(self.inspect()["releasable"])

    def test_corrupt_or_missing_log_blocks_release_and_ack(self):
        self.fenced()
        digest = json.loads((self.stage / "result.json").read_text())["log_sha256"]
        for missing in (False, True):
            if missing:
                (self.stage / "output.log").unlink()
            else:
                (self.stage / "output.log").write_text("changed")
            with self.subTest(missing=missing):
                self.assertFalse(self.inspect()["releasable"])
                with self.assertRaises((ValueError, FileNotFoundError)):
                    job.acknowledge(self.stage, self.key, digest)
                self.assertTrue(self.stage.exists())

    def test_ack_retains_evidence_while_reservation_remains(self):
        self.fenced()
        digest = json.loads((self.stage / "result.json").read_text())["log_sha256"]
        with self.assertRaisesRegex(ValueError, "preserve terminal"):
            job.acknowledge(self.stage, self.key, digest)
        self.release(self.inspect()["release_token"])
        self.assertEqual(job.acknowledge(self.stage, self.key, digest)["state"], "removed")

    def test_explicit_release_rechecks_proof_after_preview(self):
        self.fenced()
        token = self.inspect()["release_token"]
        (self.stage / "output.log").write_text("changed since preview")
        with self.assertRaisesRegex(ValueError, "log does not match"):
            self.release(token)
        self.assertEqual(capacity.inspect_existing(self.root)[0]["state"], "fenced")

    def test_changed_reservation_requires_new_inspection(self):
        self.fenced()
        token = self.inspect()["release_token"]
        record = json.loads((self.root / "slot-0.json").read_text())
        record["reservation_id"] = uuid.uuid4().hex
        capacity._write(self.root / "slot-0.json", record)
        with self.assertRaisesRegex(ValueError, "confirm again"):
            self.release(token)
        self.assertEqual(json.loads((self.root / "slot-0.json").read_text()), record)

    def test_wrong_identity_and_legacy_reservations_are_blocked(self):
        self.fenced()
        original = json.loads((self.root / "slot-0.json").read_text())
        for value in (None, "other-request"):
            with self.subTest(value=value):
                capacity._write(self.root / "slot-0.json", {**original, "request_digest": value})
                self.assertFalse(self.inspect()["releasable"])

    def test_wrong_job_cannot_use_another_jobs_token(self):
        self.fenced()
        token = self.inspect()["release_token"]
        with self.assertRaisesRegex(ValueError, "identity"):
            capacity.release_job_capacity(self.root, "wrong-job", self.key, token, lambda *_: {})
        self.assertTrue(self.inspect()["releasable"])

    def test_duplicate_release_and_old_token_never_clear_a_new_owner(self):
        self.fenced()
        token = self.inspect()["release_token"]
        self.release(token)
        self.assertEqual(self.release(token)["state"], "already-released")
        with self.assertRaisesRegex(RuntimeError, "new owner"):
            with self.pool.acquire("different-job", request_digest="different-request"):
                raise RuntimeError("new owner left fenced")
        before = (self.root / "slot-0.json").read_bytes()
        self.assertEqual(self.release(token)["state"], "already-released")
        self.assertEqual((self.root / "slot-0.json").read_bytes(), before)

    def test_lost_confirmation_after_clear_is_reconciled_not_reexecuted(self):
        self.fenced()
        token = self.inspect()["release_token"]
        original = capacity._write
        def fail(path, value):
            if path.name.startswith("release-") and value["state"] == "released":
                raise OSError("injected lost completion write")
            return original(path, value)
        with patch.object(capacity, "_write", side_effect=fail), self.assertRaises(OSError):
            self.release(token)
        self.assertEqual(capacity.inspect_existing(self.root)[0]["state"], "free")
        self.assertEqual(self.release(token)["state"], "already-released")
        audit = json.loads((self.root / f"release-{token}.json").read_text())
        self.assertEqual(audit["state"], "released")

    def test_interruption_before_clear_does_not_claim_released(self):
        self.fenced()
        token = self.inspect()["release_token"]
        original = capacity._write
        def fail(path, value):
            if value is None:
                raise OSError("injected failed reservation clear")
            return original(path, value)
        with patch.object(capacity, "_write", side_effect=fail), self.assertRaises(OSError):
            self.release(token)
        self.assertEqual(capacity.inspect_existing(self.root)[0]["state"], "fenced")
        self.assertEqual(self.release(token)["state"], "released")

    def test_failed_test_can_release_capacity_without_changing_failure(self):
        self.fenced(code=1)
        self.release(self.inspect()["release_token"])
        self.assertEqual(job.inspect_job(self.stage, self.key)["exit_code"], 1)

    def test_malformed_token_and_symlink_pool_fail_closed(self):
        for token in ("../oops", "a" * 63, "z" * 64):
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.release(token)
        link = Path(self.temp.name) / "linked"
        link.symlink_to(self.root)
        with self.assertRaises(ValueError):
            capacity.inspect_existing(link)

    def test_two_real_release_processes_cannot_split_the_slot_lock(self):
        self.fenced()
        token = self.inspect()["release_token"]
        script = """import json,sys,time
from pathlib import Path
from codex_agent.remote_capacity import release_job_capacity
def verify(*args):
    time.sleep(0.1)
    return {'fixture': 'unit proof callback'}
try:
    result=release_job_capacity(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4],verify)
except BlockingIOError:
    result={'state':'busy'}
print(json.dumps(result))
"""
        command = [sys.executable, "-c", script, str(self.root), self.stage.name, self.key, token]
        children = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        results = []
        try:
            for child in children:
                output, error = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, error)
                results.append(json.loads(output)["state"])
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)
        self.assertEqual(results.count("released"), 1)
        self.assertTrue(set(results) <= {"released", "already-released", "busy"})
        self.assertEqual(self.release(token)["state"], "already-released")

    def test_corrupt_release_audit_is_not_treated_as_a_successful_retry(self):
        self.fenced()
        token = self.inspect()["release_token"]
        capacity._write(self.root / f"release-{token}.json", {"state": "released", "job_id": "other"})
        with self.assertRaisesRegex(ValueError, "audit identity"):
            self.release(token)
        self.assertEqual(capacity.inspect_existing(self.root)[0]["state"], "fenced")

    def test_oversized_or_shared_state_is_rejected_without_replacing_it(self):
        self.fenced()
        path = self.root / "slot-0.json"
        path.write_text("x" * 65537)
        with self.assertRaisesRegex(ValueError, "small, private"):
            self.inspect()
        path.write_text("{}")
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "small, private"):
            self.inspect()
        self.assertEqual(path.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
