"""Batch fault injection with real journals/receipts and fake SSH, not business results."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codex_agent import batch_recovery, project_tools
from codex_agent.execution_guard import ROOT, atomic_json, cancel_path, digest, inspect_execution, journal_path
from codex_agent.harness.native_bridge import review_artifact
from codex_agent.remote_executor import RemotePreflightResult
from codex_agent.tests.test_operator_lifecycle import ORIGINAL_SOURCE, TEST_SOURCE


class BatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        folder = self.root / "python/examples/flaggems"
        folder.mkdir(parents=True)
        for name in ("alpha", "beta", "gamma"):
            (folder / f"{name}.py").write_text(ORIGINAL_SOURCE.replace("demo", name))
            (folder / f"test_{name}.py").write_text(TEST_SOURCE.replace("demo", name))
        environment = patch.dict(os.environ, {"RISCV_HOST": "fixture", "RISCV_REPO": "/repo",
            "TRITON_RISCV_ALLOW_VALIDATION": "1", "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "1",
            "TRITON_RISCV_REQUIRE_REMOTE": "1", "TRITON_RISCV_EMBEDDING_PROVIDER": "none"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.actions, self.starts, self.broken, self.first_exit = [], [], True, 0
        patches = {
            "preflight": patch("codex_agent.remote_executor.check_remote_environment", return_value=
                RemotePreflightResult(configured=True, status="passed", architecture="riscv64")),
            "ssh": patch("codex_agent.remote_executor._run_ssh_script", return_value=subprocess.CompletedProcess([], 0, "")),
            "upload": patch("codex_agent.remote_executor.run_bounded", return_value=subprocess.CompletedProcess([], 0, "")),
            "rpc": patch("codex_agent.remote_executor._job_rpc", side_effect=self.rpc),
            "memory": patch("codex_agent.operator_lifecycle.remember_validation", return_value={"status": "not-recorded"}),
        }
        self.mocks = {}
        for key, entry in patches.items():
            self.mocks[key] = entry.start()
            self.addCleanup(entry.stop)
        self.job = project_tools.prepare_validation_job(self.root, ["alpha", "beta", "gamma"], timeout_seconds=20)
        self.id = self.job["job_id"]
        project_tools.decide_validation_job(self.root, self.id, approve=True, reviewer="native-harness:fixture-session")

    def rpc(self, config, job, action, *extra):
        self.actions.append((action, job["job_id"]))
        if action == "start":
            self.starts.append(job["job_id"])
        if action == "inspect" and len(self.starts) == 2 and self.broken:
            self.broken = False
            raise RuntimeError("Remote outcome unknown; persistent connection loss fixture")
        code = self.first_exit if job["job_id"] == self.starts[0] else 0
        raw = b"1 failed in 0.1s\n" if code else b"1 passed in 0.1s\n"
        cancelled = getattr(self, "cancel_second", False) and len(self.starts) == 2
        if cancelled:
            code, raw = 130, b"VALIDATION_CANCELLED=supervisor confirmed shutdown\n"
        return {"job_id": job["job_id"], "request_digest": job["request_digest"],
                "state": "removed" if action == "ack" else "completed", "exit_code": code,
                "cancellation": {"confirmed": cancelled},
                "duration_seconds": 0.1, "log_bytes": len(raw), "log_sha256": hashlib.sha256(raw).hexdigest(),
                "log_base64": base64.b64encode(raw).decode()}

    def journal(self):
        return inspect_execution(self.root, "job", self.id)

    def pause(self):
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            project_tools.execute_validation_job(self.root, self.id)
        self.assertEqual(self.journal()["state"], "unknown")
        self.assertEqual([entry["phase"] for entry in self.journal()["execution"]["batch_progress"]["items"]],
                         ["completed", "active", "pending"])
        self.assertEqual(project_tools.load_job(self.root, self.id)["status"], "interrupted")

    def assert_blocked(self, pattern):
        previous = list(self.actions)
        uploads = self.mocks["upload"].call_count
        with self.assertRaisesRegex(PermissionError, pattern):
            project_tools.execute_validation_job(self.root, self.id)
        self.assertEqual(self.actions, previous)
        self.assertEqual(self.mocks["upload"].call_count, uploads)

    def test_resume_replays_first_collects_second_starts_only_third(self):
        self.first_exit = 1
        self.pause()
        deadline = self.journal()["execution"]["batch_progress"]["deadline_at"]
        second_job = self.starts[1]
        actions_before = len(self.actions)
        uploads_before = self.mocks["upload"].call_count
        review = review_artifact(self.root, "job", self.id, "fixture-session")
        self.assertTrue(review["recovery"])
        result = project_tools.execute_validation_job(self.root, self.id)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([item["status"] for item in result["results"]], ["failed", "passed", "passed"])
        self.assertEqual(len(self.starts), 3)
        self.assertEqual(len(set(self.starts)), 3)
        self.assertEqual(self.actions[actions_before:actions_before+3],
                         [("inspect", second_job), ("collect", second_job), ("ack", second_job)])
        self.assertEqual(self.mocks["preflight"].call_count, 3)
        self.assertEqual(self.mocks["upload"].call_count - uploads_before, uploads_before // 2)
        self.assertEqual(self.journal()["execution"]["batch_progress"]["deadline_at"], deadline)
        self.assertEqual(len({item["run_id"] for item in result["results"]}), 3)
        count = len(self.actions)
        self.assertEqual(project_tools.execute_validation_job(self.root, self.id), result)
        self.assertEqual(len(self.actions), count)

    def test_child_committed_before_parent_checkpoint_is_not_reexecuted(self):
        self.broken = False
        original = batch_recovery.mark
        def crash(index, phase, result=None):
            if index == 1 and phase == "completed":
                raise RuntimeError("crash after child commit")
            return original(index, phase, result)
        with patch.object(batch_recovery, "mark", side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, "child commit"):
                project_tools.execute_validation_job(self.root, self.id)
        self.assertEqual(len(self.starts), 2)
        self.assertEqual(project_tools.execute_validation_job(self.root, self.id)["status"], "passed")
        self.assertEqual(len(self.starts), 3)

    def test_modified_completed_log_blocks_entire_batch_before_network(self):
        self.pause()
        child = inspect_execution(self.root, "validation", self.job["items"][0]["plan"]["run_id"])
        Path(child["result"]["log_path"]).write_text("tampered")
        self.assert_blocked("evidence changed")

    def test_changed_pending_plan_blocks_before_network(self):
        self.pause()
        path = self.root / self.job["items"][2]["plan"]["receipt_path"]
        plan = json.loads(path.read_text())
        plan["timeout_seconds"] += 1
        atomic_json(path, plan)
        self.assert_blocked("child plan changed")

    def test_changed_source_blocks_all_resume(self):
        self.pause()
        path = self.root / "python/examples/flaggems/gamma.py"
        path.write_text(path.read_text() + "\n# changed\n")
        self.assert_blocked("Files changed during disconnect")

    def test_missing_active_journal_is_not_permission_to_start_again(self):
        self.pause()
        journal_path(self.root, "validation", self.job["items"][1]["plan"]["run_id"]).unlink()
        self.assert_blocked("no matching durable journal")

    def test_lost_handle_blocks_and_keeps_third_pending(self):
        self.pause()
        identifier = self.job["items"][1]["plan"]["run_id"]
        child = inspect_execution(self.root, "validation", identifier)
        child["execution"].pop("remote_job")
        atomic_json(journal_path(self.root, "validation", identifier), child)
        self.assert_blocked("no recoverable remote handle")

    def test_checkpoint_identity_tampering_is_blocked(self):
        self.pause()
        journal = self.journal()
        journal["execution"]["batch_progress"]["items"][1]["run_id"] = "run-wrong"
        atomic_json(journal_path(self.root, "job", self.id), journal)
        self.assert_blocked("identity mismatch")

    def test_original_deadline_cannot_be_extended(self):
        self.pause()
        journal = self.journal()
        journal["execution"]["batch_progress"]["deadline_at"] += 900
        atomic_json(journal_path(self.root, "job", self.id), journal)
        self.assert_blocked("original deadline changed")

    def test_expired_budget_blocks_new_work(self):
        self.pause()
        deadline = self.journal()["execution"]["batch_progress"]["deadline_at"]
        with patch("codex_agent.batch_recovery.time.time", return_value=deadline + 1):
            self.assert_blocked("budget expired")

    def test_cancelled_batch_requires_host_not_auto_resume(self):
        self.pause()
        atomic_json(cancel_path(self.root, "job", self.id), {"requested": True})
        self.assert_blocked("Batch was cancelled")

    def test_foreign_native_session_cannot_resume(self):
        self.pause()
        with self.assertRaisesRegex(PermissionError, "another host/session"):
            review_artifact(self.root, "job", self.id, "foreign-session")

    def test_missing_checkpoint_with_effects_is_not_restartable(self):
        self.pause()
        journal = self.journal()
        journal["execution"].pop("batch_progress")
        atomic_json(journal_path(self.root, "job", self.id), journal)
        self.assert_blocked("progress is missing")

    def test_readonly_progress_does_not_restart_or_discard_blocked_records(self):
        self.pause()
        actions = list(self.actions)
        status = project_tools.get_validation_job(self.root, self.id)
        self.assertIn("resume-same-job", status["recovery"]["next_action"])
        self.assertEqual(status["recovery"]["observation"], "local-record-only")
        self.assertEqual(len(status["results"]), 1)
        atomic_json(cancel_path(self.root, "job", self.id), {"requested": True})
        status = project_tools.get_validation_job(self.root, self.id)
        self.assertEqual(status["recovery"]["next_action"], "host-inspection-required")
        self.assertEqual(self.actions, actions)

    def test_stale_running_journals_recover_with_same_child(self):
        self.pause()
        for kind, identifier in (("job", self.id), ("validation", self.job["items"][1]["plan"]["run_id"])):
            journal = inspect_execution(self.root, kind, identifier)
            journal["state"] = "running"
            atomic_json(journal_path(self.root, kind, identifier), journal)
        self.assertEqual(project_tools.execute_validation_job(self.root, self.id)["status"], "passed")
        self.assertEqual(len(self.starts), 3)

    def test_crash_after_intent_before_child_journal_fails_closed(self):
        with patch("codex_agent.project_tools.lifecycle.decide_validation_plan", side_effect=RuntimeError("crash before decision")):
            with self.assertRaisesRegex(RuntimeError, "before decision"):
                project_tools.execute_validation_job(self.root, self.id)
        self.assert_blocked("no matching durable journal")
        self.assertEqual(self.starts, [])

    def test_pending_child_cannot_hide_an_execution_journal(self):
        self.pause()
        identifier = self.job["items"][2]["plan"]["run_id"]
        atomic_json(journal_path(self.root, "validation", identifier), {"kind": "validation", "id": identifier, "state": "unknown"})
        self.assert_blocked("Pending batch item has execution")

    def test_child_cancel_marker_blocks_batch_resume(self):
        self.pause()
        identifier = self.job["items"][1]["plan"]["run_id"]
        atomic_json(cancel_path(self.root, "validation", identifier), {"requested": True})
        self.assert_blocked("child was cancelled")

    def test_remote_cancelled_child_does_not_start_later_items(self):
        self.broken, self.cancel_second = False, True
        with self.assertRaisesRegex(RuntimeError, "child was cancelled"):
            project_tools.execute_validation_job(self.root, self.id)
        self.assertEqual(len(self.starts), 2)
        self.assert_blocked("child cancellation requires host inspection")

    def test_foreign_unfinished_claim_cannot_be_reclaimed_by_batch(self):
        self.pause()
        key = "file:" + str(self.root / "python/examples/flaggems/beta.py")
        atomic_json(self.root / ROOT / "claims" / (digest(key) + ".json"),
                    {"kind": "validation", "id": "run-foreign"})
        atomic_json(journal_path(self.root, "validation", "run-foreign"), {"state": "unknown"})
        self.assert_blocked("unresolved execution owns these files")

    def test_new_batch_cannot_bypass_the_unfinished_original(self):
        self.pause()
        other = project_tools.prepare_validation_job(self.root, ["alpha", "beta", "gamma"], timeout_seconds=20)
        project_tools.decide_validation_job(self.root, other["job_id"], approve=True, reviewer="native-harness:fixture-session")
        before = list(self.actions)
        with self.assertRaisesRegex(PermissionError, "unresolved execution owns these files"):
            project_tools.execute_validation_job(self.root, other["job_id"])
        self.assertEqual(self.actions, before)
