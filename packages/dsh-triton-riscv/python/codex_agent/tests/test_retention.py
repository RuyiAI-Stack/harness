"""Retention fixtures exercise policy and evidence, not real operator success."""
import fcntl
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

from codex_agent.execution_guard import ROOT, approval_digest, atomic_json, digest, file_snapshot, journal_path
from codex_agent import remote_capacity as capacity, remote_job as job
from codex_agent.remote_maintenance import cleanup_collected_validation
from codex_agent.retention import RetentionPolicy, preview_retention

DAY = 86400
NOW = 1800000000


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.env = patch.dict(os.environ, {"RISCV_HOST": "host", "RISCV_REPO": "/repo"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def fixture(self, name="first", status="passed", age=8):
        run_id, result_id = "run-" + name, "result-" + name
        folder = self.root / "agent-results/operator-lifecycle/receipts"
        folder.mkdir(parents=True, exist_ok=True)
        log = self.root / f"{name}.log"
        log.write_text("fixture evidence, not a real validation\n")
        checksum = job.file_digest(log)
        plan = {"run_id": run_id, "operator": name, "status": "planned",
                "approval": {"status": "approved", "execution_run_id": result_id}}
        plan["approval_digest"] = approval_digest(self.root, "validation", plan)
        atomic_json(folder / f"{run_id}.json", plan)
        receipt_path = folder / f"{result_id}.json"
        result = {"run_id": result_id, "approved_run_id": run_id, "operator": name, "status": status,
                  "exit_code": 0 if status == "passed" else 1,
                  "receipt_path": str(receipt_path), "log_path": str(log)}
        atomic_json(receipt_path, result)
        job_id = uuid.uuid4().hex
        terminal = {"state": "completed", "job_id": job_id, "request_digest": "b" * 64,
                    "log_sha256": checksum, "exit_code": result["exit_code"], "finished_at": NOW - age * DAY}
        handle = {"host": "host", "repository": "/repo", "job_id": job_id, "request_digest": "b" * 64,
                  "stage": "/tmp/triton-riscv-agent/" + job_id, "terminal": terminal, "log_sha256": checksum}
        journal = {"kind": "validation", "id": run_id, "state": "completed", "completed_at": NOW - age * DAY,
                   "result": result, "execution": {"remote_job": handle},
                   "result_files": file_snapshot(self.root, [str(log), str(receipt_path)])}
        atomic_json(journal_path(self.root, "validation", run_id), journal)
        return plan, journal

    def save(self, journal):
        atomic_json(journal_path(self.root, journal["kind"], journal["id"]), journal)

    def preview(self, **kwargs):
        return preview_retention(self.root, now=NOW, **kwargs)

    def response(self, journal, **extra):
        handle = journal["execution"]["remote_job"]
        return {"state": "eligible", "job_id": handle["job_id"], "request_digest": handle["request_digest"],
                "terminal_sha256": digest(handle["terminal"]), "log_sha256": handle["log_sha256"], **extra}

    def tree(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None for p in self.root.rglob("*")}

    def test_empty_repo_is_readonly_and_does_not_create_state_dirs(self):
        with patch("codex_agent.retention._job_rpc") as rpc:
            self.assertEqual(self.preview()["items"], [])
            rpc.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_default_policy_distinguishes_success_failure_and_cancelled(self):
        self.fixture("pass", age=8)
        self.fixture("fail", status="failed", age=8)
        self.fixture("cancel", status="cancelled", age=8)
        before = self.tree()
        with patch("codex_agent.retention._job_rpc") as rpc:
            result = self.preview()
        self.assertEqual(result["counts"], {"local-candidate": 1, "retained": 2})
        self.assertEqual(before, self.tree())
        rpc.assert_not_called()
        self.assertTrue(all(i["local_action"] == "retain" for i in result["items"]))

    def test_period_boundary_and_custom_policy(self):
        self.fixture(age=7)
        self.assertEqual(self.preview()["items"][0]["state"], "local-candidate")
        self.assertEqual(self.preview(policy=RetentionPolicy(8, 30))["items"][0]["state"], "retained")
        self.assertEqual(self.preview(policy=RetentionPolicy(0, 0))["items"][0]["state"], "local-candidate")

    def test_clock_future_missing_nan_never_uses_file_mtime(self):
        _, journal = self.fixture(age=100)
        for value in (None, float("nan"), -1, NOW + DAY, "yesterday", True):
            with self.subTest(value=value):
                journal["completed_at"] = value
                self.save(journal)
                self.assertEqual(self.preview()["items"][0]["state"], "protected")

    def test_running_unknown_and_rejected_are_never_age_cleanup_candidates(self):
        _, journal = self.fixture(age=100)
        with patch("codex_agent.retention._job_rpc") as rpc:
            for state in ("running", "unknown", "rejected"):
                with self.subTest(state=state):
                    journal["state"] = state
                    self.save(journal)
                    self.assertEqual(self.preview(remote=True)["items"][0]["reason"], "execution-not-durably-completed")
            rpc.assert_not_called()

    def test_changed_or_missing_log_is_protected_without_network(self):
        _, journal = self.fixture()
        log = Path(journal["result"]["log_path"])
        with patch("codex_agent.retention._job_rpc") as rpc:
            log.write_text("changed")
            self.assertEqual(self.preview(remote=True)["items"][0]["state"], "protected")
            log.unlink()
            self.assertEqual(self.preview(remote=True)["items"][0]["state"], "protected")
            rpc.assert_not_called()

    def test_changed_approval_and_contradictory_outcome_are_protected(self):
        plan, journal = self.fixture()
        path = self.root / "agent-results/operator-lifecycle/receipts" / f"{plan['run_id']}.json"
        atomic_json(path, {**plan, "operator": "changed"})
        self.assertEqual(self.preview()["items"][0]["state"], "protected")
        atomic_json(path, plan)
        journal["result"]["status"] = "failed"
        self.save(journal)
        self.assertIn("outcomes differ", self.preview()["items"][0]["reason"])

    def test_missing_receipt_directory_is_not_created(self):
        _, journal = self.fixture()
        shutil.rmtree(self.root / "agent-results/operator-lifecycle")
        self.assertEqual(self.preview()["items"][0]["state"], "protected")
        self.assertFalse((self.root / "agent-results/operator-lifecycle").exists())

    def test_remote_check_only_queries_mature_valid_candidates(self):
        _, journal = self.fixture()
        self.fixture("young", age=1)
        before = self.tree()
        with patch("codex_agent.retention._job_rpc", return_value=self.response(journal)) as rpc:
            result = self.preview(remote=True)
        self.assertEqual(result["counts"], {"reviewable": 1, "retained": 1})
        self.assertEqual(rpc.call_count, 1)
        self.assertEqual(rpc.call_args.args[2], "cleanup-preview")
        self.assertEqual(self.tree(), before)

    def test_remote_checks_are_bounded_and_unchecked_records_stay_protected(self):
        self.fixture("one")
        self.fixture("two")
        with patch("codex_agent.retention._job_rpc", side_effect=RuntimeError("network failure")) as rpc:
            result = self.preview(remote=True, remote_limit=1)
        self.assertEqual(rpc.call_count, 1)
        self.assertEqual(result["remote_checks_attempted"], 1)
        self.assertEqual(result["counts"], {"protected": 2})
        self.assertTrue(any("budget-exhausted" in i["reason"] for i in result["items"]))

    def test_remote_unavailable_or_old_protocol_remains_protected_not_removed(self):
        self.fixture()
        with patch("codex_agent.retention._job_rpc", side_effect=RuntimeError("SSH or old protocol failure")):
            result = self.preview(remote=True)["items"][0]
        self.assertEqual(result["state"], "protected")
        self.assertEqual(result["remote_action"], "retain")

    def test_changed_target_blocks_network(self):
        self.fixture()
        with patch.dict(os.environ, {"RISCV_HOST": "other"}), patch("codex_agent.retention._job_rpc") as rpc:
            self.assertEqual(self.preview(remote=True)["items"][0]["state"], "protected")
            rpc.assert_not_called()

    def test_remote_identity_hash_and_terminal_changes_are_rejected(self):
        _, journal = self.fixture()
        for field in ("job_id", "request_digest", "log_sha256", "terminal_sha256"):
            with self.subTest(field=field), patch("codex_agent.retention._job_rpc", return_value=self.response(journal, **{field: "wrong"})):
                self.assertEqual(self.preview(remote=True)["items"][0]["state"], "protected")

    def test_remote_reserved_or_busy_remains_protected(self):
        _, journal = self.fixture()
        with patch("codex_agent.retention._job_rpc", return_value=self.response(journal, state="protected", reason="capacity-still-reserved")):
            item = self.preview(remote=True)["items"][0]
        self.assertEqual(item["state"], "protected")
        self.assertIn("capacity-still-reserved", item["reason"])

    def test_prior_cleanup_receipt_is_historical_not_a_new_remote_observation(self):
        plan, journal = self.fixture()
        handle = journal["execution"]["remote_job"]
        atomic_json(self.root / ROOT / "cleanup" / f"{handle['job_id']}.json", {
            "run_id": plan["run_id"], "job_id": handle["job_id"], "log_sha256": handle["log_sha256"],
            "state": "removed", "checked_at": NOW - DAY})
        with patch("codex_agent.retention._job_rpc") as rpc:
            item = self.preview(remote=True)["items"][0]
            rpc.assert_not_called()
        self.assertEqual(item["state"], "previously-cleaned")
        self.assertEqual(item["observation"], "local-only")

    def test_unknown_cleanup_receipt_is_not_treated_as_deletion_confirmation(self):
        plan, journal = self.fixture()
        handle = journal["execution"]["remote_job"]
        atomic_json(self.root / ROOT / "cleanup" / f"{handle['job_id']}.json", {
            "run_id": plan["run_id"], "job_id": handle["job_id"], "log_sha256": handle["log_sha256"], "state": "unknown"})
        self.assertEqual(self.preview()["items"][0]["state"], "local-candidate")

    def test_pagination_has_no_silent_history_cutoff(self):
        for name in ("a", "b", "c"):
            self.fixture(name)
        first = self.preview(limit=2)
        second = self.preview(limit=2, after=first["page"]["next_after"])
        ids = [i["run_id"] for i in first["items"] + second["items"]]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(first["page"]["has_more"])
        self.assertFalse(second["page"]["has_more"])

    def test_corrupt_journal_does_not_hide_other_records(self):
        _, journal = self.fixture()
        atomic_json(self.root / ROOT / "operations" / ("f" * 64 + ".json"), [])
        result = self.preview()
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["counts"], {"local-candidate": 1, "protected": 1})

    def test_parent_symlink_is_not_scanned(self):
        with tempfile.TemporaryDirectory() as external:
            (self.root / "agent-results").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.preview()

    def test_outside_evidence_and_symlink_evidence_are_protected(self):
        _, journal = self.fixture()
        log = Path(journal["result"]["log_path"])
        with tempfile.NamedTemporaryFile() as external:
            external_path = Path(external.name)
            log.unlink()
            log.symlink_to(external_path)
            self.assertEqual(self.preview()["items"][0]["state"], "protected")
            log.unlink()
            journal["result_files"] = {str(external_path): "fake"}
            self.save(journal)
            self.assertEqual(self.preview()["items"][0]["state"], "protected")

    def test_nonvalidation_records_and_rag_are_not_cleanup_targets(self):
        self.fixture()
        atomic_json(journal_path(self.root, "repair", "repair-1"), {"id": "repair-1", "kind": "repair", "state": "completed"})
        (self.root / "memory.sqlite3").write_bytes(b"not-a-real-database")
        before = self.tree()
        result = self.preview()
        self.assertEqual(result["counts"], {"local-candidate": 1, "protected": 1})
        self.assertEqual(self.tree(), before)

    def test_explicit_cleanup_still_rechecks_after_a_successful_preview(self):
        plan, journal = self.fixture()
        with patch("codex_agent.retention._job_rpc", return_value=self.response(journal)):
            self.assertEqual(self.preview(remote=True)["items"][0]["state"], "reviewable")
        Path(journal["result"]["log_path"]).write_text("changed after preview")
        with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
            with self.assertRaises(PermissionError):
                cleanup_collected_validation(self.root, plan["run_id"])
            rpc.assert_not_called()

    def test_invalid_policy_cursor_and_limit_fail_fast(self):
        for value in (-1, True, float("nan"), 3651):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RetentionPolicy(value, 30)
        for options in ({"limit": 0}, {"limit": 1001}, {"after": "../../etc/passwd"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.preview(**options)

    def test_cli_refuses_to_overwrite_report_or_existing_evidence(self):
        target = self.root / "keep.json"
        target.write_text("KEEP")
        command = [sys.executable, "-m", "codex_agent.retention", "--repo", str(self.root), "--output", str(target)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_text(), "KEEP")


class RemoteCleanupPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.capacity_root = Path(self.temp.name) / "capacity"
        parent = Path("/tmp/triton-riscv-agent")
        parent.mkdir(exist_ok=True)
        self.stage = parent / uuid.uuid4().hex
        self.stage.mkdir(mode=0o700)
        self.request = {"job_id": self.stage.name, "version": 1, "timeout_seconds": 5}
        self.key = job.digest(self.request)
        job.atomic_json(self.stage / "request.json", self.request)
        self.server = patch.object(capacity, "server_root", return_value=self.capacity_root)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        shutil.rmtree(self.stage, ignore_errors=True)
        self.temp.cleanup()

    def terminal(self):
        with capacity._open_lock(self.stage / "worker.lock"):
            pass
        (self.stage / "output.log").write_text("unit terminal fixture\n")
        return job.save_result(self.stage, self.request, self.key, 0, time.monotonic(), {"state": "admitted"},
                               shutdown={"version": 1, "descendants_stopped": True, "method": "linux-subreaper-pidfd"})

    def test_missing_terminal_does_not_create_worker_lock_or_capacity_pool(self):
        before = {p.name: p.read_bytes() for p in self.stage.iterdir()}
        self.assertEqual(job.preview_cleanup(self.stage, self.key)["state"], "protected")
        self.assertEqual({p.name: p.read_bytes() for p in self.stage.iterdir()}, before)
        self.assertFalse(self.capacity_root.exists())

    def test_valid_terminal_yields_exact_digest_without_deletion(self):
        terminal = self.terminal()
        result = job.preview_cleanup(self.stage, self.key)
        self.assertEqual(result["state"], "eligible")
        self.assertEqual(result["terminal_sha256"], digest(terminal))
        self.assertTrue((self.stage / "output.log").exists())
        self.assertFalse(self.capacity_root.exists())

    def test_missing_or_busy_worker_is_not_cleanup_ready(self):
        self.terminal()
        with capacity._open_lock(self.stage / "worker.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.assertEqual(job.preview_cleanup(self.stage, self.key)["reason"], "worker-still-active")
        (self.stage / "worker.lock").unlink()
        self.assertEqual(job.preview_cleanup(self.stage, self.key)["reason"], "worker-state-missing")
        self.assertFalse((self.stage / "worker.lock").exists())

    def test_reserved_slot_remains_protected(self):
        self.terminal()
        with capacity.CapacityPool(self.capacity_root).acquire(self.stage.name, request_digest=self.key):
            self.assertEqual(job.preview_cleanup(self.stage, self.key)["reason"], "capacity-still-reserved")

    def test_legacy_shutdown_and_changed_log_remain_protected(self):
        terminal = self.terminal()
        terminal.pop("shutdown")
        job.atomic_json(self.stage / "result.json", terminal)
        self.assertEqual(job.preview_cleanup(self.stage, self.key)["reason"], "shutdown-proof-missing-or-legacy")
        (self.stage / "output.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "log does not match"):
            job.preview_cleanup(self.stage, self.key)

    def test_prestart_cancel_needs_no_created_worker_lock(self):
        result = job.request_cancel(self.stage, self.key)
        self.assertEqual(result["exit_code"], 130)
        self.assertEqual(job.preview_cleanup(self.stage, self.key)["state"], "eligible")
        self.assertFalse((self.stage / "worker.lock").exists())


if __name__ == "__main__":
    unittest.main()
