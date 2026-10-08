import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_agent.execution_guard import approval_digest, atomic_json, file_snapshot, journal_path
from codex_agent.remote_maintenance import capacity_maintenance, cleanup_collected_validation, validation_status


class RemoteMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.run_id = "run-fixture-approved"
        self.result_id = "run-fixture-result"
        self.folder = self.root / "agent-results/operator-lifecycle/receipts"
        self.folder.mkdir(parents=True)
        self.log = self.root / "test.log"
        self.log.write_text("unit fixture, not business execution\n")
        log_digest = hashlib.sha256(self.log.read_bytes()).hexdigest()
        self.plan = {"run_id": self.run_id, "status": "planned", "operator": "demo",
                     "approval": {"status": "approved", "execution_run_id": self.result_id}}
        self.plan["approval_digest"] = approval_digest(self.root, "validation", self.plan)
        atomic_json(self.folder / f"{self.run_id}.json", self.plan)
        self.receipt_path = self.folder / f"{self.result_id}.json"
        self.result = {"run_id": self.result_id, "approved_run_id": self.run_id, "status": "failed",
                       "receipt_path": str(self.receipt_path.relative_to(self.root)), "log_path": str(self.log)}
        atomic_json(self.receipt_path, self.result)
        self.job = {"host": "host", "repository": "/repo", "job_id": "a" * 32,
                    "stage": "/tmp/triton-riscv-agent/" + "a" * 32, "request_digest": "b" * 64,
                    "log_sha256": log_digest, "terminal": {"log_sha256": log_digest}}
        self.journal = {"kind": "validation", "id": self.run_id, "state": "completed",
                        "result": self.result, "execution": {"remote_job": self.job},
                        "result_files": file_snapshot(self.root, [str(self.receipt_path), str(self.log)])}
        self.save()
        self.env = patch.dict(os.environ, {"RISCV_HOST": "host", "RISCV_REPO": "/repo"})
        self.env.start()

    def save(self):
        atomic_json(journal_path(self.root, "validation", self.run_id), self.journal)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_cleanup_only_acknowledges_exact_committed_job_and_preserves_local_evidence(self):
        with patch("codex_agent.remote_maintenance._job_rpc", return_value={"state": "removed", "job_id": self.job["job_id"]}) as rpc:
            result = cleanup_collected_validation(self.root, self.run_id)
        self.assertEqual(result["state"], "removed")
        self.assertEqual(rpc.call_args.args[2:], ("ack", self.job["log_sha256"]))
        self.assertTrue(self.log.exists())
        self.assertTrue(self.receipt_path.exists())

    def test_running_unknown_and_missing_journal_never_clean(self):
        for state in ("running", "unknown", "rejected"):
            with self.subTest(state=state):
                self.journal["state"] = state
                self.save()
                with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
                    with self.assertRaises(PermissionError):
                        cleanup_collected_validation(self.root, self.run_id)
                    rpc.assert_not_called()

    def test_changed_log_or_receipt_is_preserved_without_network_access(self):
        for path in (self.log, self.receipt_path):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_text("tampered")
                with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
                    with self.assertRaisesRegex(PermissionError, "changed"):
                        cleanup_collected_validation(self.root, self.run_id)
                    rpc.assert_not_called()
                path.write_bytes(original)

    def test_wrong_job_log_hash_or_changed_host_blocks_cleanup(self):
        self.job["log_sha256"] = "wrong"
        self.save()
        with self.assertRaisesRegex(PermissionError, "do not match"):
            cleanup_collected_validation(self.root, self.run_id)
        self.job["log_sha256"] = hashlib.sha256(self.log.read_bytes()).hexdigest()
        self.job["host"] = "other"
        self.save()
        with self.assertRaisesRegex(PermissionError, "configuration differs"):
            cleanup_collected_validation(self.root, self.run_id)

    def test_lost_ack_records_unknown_without_changing_committed_test_outcome(self):
        before = journal_path(self.root, "validation", self.run_id).read_bytes()
        with patch("codex_agent.remote_maintenance._job_rpc", side_effect=RuntimeError("connection lost")):
            report = cleanup_collected_validation(self.root, self.run_id)
        self.assertEqual(report["state"], "unknown")
        self.assertEqual(journal_path(self.root, "validation", self.run_id).read_bytes(), before)

    def test_wrong_ack_job_is_not_reported_as_removed(self):
        with patch("codex_agent.remote_maintenance._job_rpc", return_value={"state": "removed", "job_id": "wrong"}):
            self.assertEqual(cleanup_collected_validation(self.root, self.run_id)["state"], "unknown")

    def test_local_status_distinguishes_finished_from_test_passed_without_ssh(self):
        with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
            result = validation_status(self.root, self.run_id)
        rpc.assert_not_called()
        self.assertEqual(result["execution_state"], "completed")
        self.assertEqual(result["test_status"], "failed")
        self.assertEqual(result["observation"], "local-record-only")

    def test_live_query_can_report_queue_and_completion_without_claiming_verified_test(self):
        self.journal["state"] = "unknown"
        self.save()
        for remote_state in ("queued", "running", "completed"):
            with self.subTest(state=remote_state), patch("codex_agent.remote_maintenance._job_rpc", return_value={"state": remote_state}) as rpc:
                result = validation_status(self.root, self.run_id, remote=True)
                self.assertEqual(result["remote_state"], remote_state)
                self.assertIsNone(result["test_status"])
                self.assertEqual(rpc.call_args.args[2], "inspect")

    def test_unreachable_remote_is_unknown_not_failed_operator(self):
        self.journal["state"] = "unknown"
        self.save()
        with patch("codex_agent.remote_maintenance._job_rpc", side_effect=RuntimeError("timeout")):
            result = validation_status(self.root, self.run_id, remote=True)
        self.assertEqual(result["remote_state"], "unknown")
        self.assertIsNone(result["test_status"])
        self.assertEqual(result["observation"], "remote-query-failed")

    def test_cancel_request_is_not_a_pass_or_permission_to_retry(self):
        from codex_agent.execution_guard import cancel_path
        self.journal["state"] = "unknown"
        self.save()
        atomic_json(cancel_path(self.root, "validation", self.run_id), {"requested": True})
        result = validation_status(self.root, self.run_id)
        self.assertIsNone(result["test_status"])
        self.assertIn("do not retry", result["next_action"])
        with patch("codex_agent.remote_maintenance._job_rpc", return_value={
            "state": "completed", "cancellation": {"confirmed": True}}):
            result = validation_status(self.root, self.run_id, remote=True)
        self.assertIsNone(result["test_status"])
        self.assertIn("remote-stop-confirmed", result["next_action"])

    def test_cancel_sidecar_from_other_job_is_rejected(self):
        atomic_json(journal_path(self.root, "validation", self.run_id).with_suffix(".remote-cancel.json"),
                    {"run_id": self.run_id, "job_id": "wrong", "remote_shutdown_confirmed": True})
        with self.assertRaisesRegex(ValueError, "cancellation evidence identity"):
            validation_status(self.root, self.run_id)

    def test_result_id_and_wrong_journal_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "original approved plan"):
            validation_status(self.root, self.result_id)
        self.journal["id"] = "wrong"
        self.save()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validation_status(self.root, self.run_id)

    def test_missing_seal_does_not_authorize_cleanup(self):
        self.plan.pop("approval_digest")
        atomic_json(self.folder / f"{self.run_id}.json", self.plan)
        with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
            with self.assertRaisesRegex(PermissionError, "seal changed"):
                cleanup_collected_validation(self.root, self.run_id)
            rpc.assert_not_called()

    def test_preflight_failure_without_remote_log_is_queryable_but_never_cleanable(self):
        self.result["log_path"] = None
        self.journal["execution"] = {}
        atomic_json(self.receipt_path, self.result)
        self.journal["result_files"] = file_snapshot(self.root, [str(self.receipt_path)])
        self.save()
        with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
            self.assertEqual(validation_status(self.root, self.run_id)["test_status"], "failed")
            with self.assertRaisesRegex(PermissionError, "log missing"):
                cleanup_collected_validation(self.root, self.run_id)
            rpc.assert_not_called()

    def test_capacity_inspect_does_not_write_local_state_or_release(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        response = {"state": "blocked", "job_id": self.job["job_id"], "request_digest": self.job["request_digest"]}
        with patch("codex_agent.remote_maintenance._job_rpc", return_value=response) as rpc:
            result = capacity_maintenance(self.root, self.run_id)
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(rpc.call_args.args[2:], ("capacity-inspect", None))
        self.assertEqual({str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_capacity_release_uses_original_job_and_never_rewrites_validation(self):
        before = journal_path(self.root, "validation", self.run_id).read_bytes()
        token = "c" * 64
        response = {"state": "released", "job_id": self.job["job_id"],
                    "request_digest": self.job["request_digest"], "release_token": token}
        with patch("codex_agent.remote_maintenance._job_rpc", return_value=response) as rpc:
            result = capacity_maintenance(self.root, self.run_id, release_token=token)
        self.assertEqual(result["state"], "released")
        self.assertEqual(rpc.call_args.args[2:], ("capacity-release", token))
        self.assertEqual(journal_path(self.root, "validation", self.run_id).read_bytes(), before)

    def test_capacity_lost_response_stays_unknown_and_is_not_retried(self):
        with patch("codex_agent.remote_maintenance._job_rpc", side_effect=RuntimeError("lost response")) as rpc:
            result = capacity_maintenance(self.root, self.run_id, release_token="c" * 64)
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(rpc.call_count, 1)

    def test_capacity_wrong_receipt_response_cannot_claim_release(self):
        with patch("codex_agent.remote_maintenance._job_rpc", return_value={"state": "released", "job_id": "wrong"}):
            result = capacity_maintenance(self.root, self.run_id, release_token="c" * 64)
        self.assertEqual(result["state"], "unknown")

    def test_capacity_invalid_token_or_changed_plan_blocks_network(self):
        with patch("codex_agent.remote_maintenance._job_rpc") as rpc:
            with self.assertRaises(ValueError):
                capacity_maintenance(self.root, self.run_id, release_token="../bad")
            self.plan["operator"] = "changed"
            atomic_json(self.folder / f"{self.run_id}.json", self.plan)
            with self.assertRaisesRegex(PermissionError, "seal changed"):
                capacity_maintenance(self.root, self.run_id)
            rpc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
