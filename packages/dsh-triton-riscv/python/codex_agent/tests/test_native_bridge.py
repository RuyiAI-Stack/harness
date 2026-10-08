from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_agent.harness.native_bridge import dispatch
from codex_agent.operator_development import prepare_operator_development, propose_operator_implementation, apply_operator_implementation, _load_proposal
from codex_agent.tests.test_operator_development import valid_spec, IMPLEMENTATION, TEST_SOURCE


class NativeBridgeTests(unittest.TestCase):
    def test_original_session_can_collect_unknown_remote_job_but_not_changed_sources(self):
        from codex_agent.execution_guard import atomic_json, journal_path
        from codex_agent.operator_lifecycle import validate_operator_target, decide_validation_plan, _load_receipt
        folder = self.root / "python/examples/flaggems"
        source = folder / "square_new.py"
        source.write_text(IMPLEMENTATION)
        (folder / "test_square_new.py").write_text(TEST_SOURCE)
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_VALIDATION": "1", "RISCV_HOST": "fixture", "RISCV_REPO": "/repo"}):
            plan = validate_operator_target(self.root, "square_new")
            decide_validation_plan(self.root, plan.run_id, approve=True, reviewer="native-harness:native-test")
            record = _load_receipt(self.root, plan.run_id)
            journal = {"state": "unknown", "files_before": record["source_snapshot"],
                       "execution": {"remote_job": {"job_id": "fixture"}}}
            path = journal_path(self.root, "validation", plan.run_id)
            atomic_json(path, journal)
            request = {"action": "review", "kind": "validation", "id": plan.run_id, "session_id": "native-test"}
            review = dispatch(self.root, request)
            self.assertTrue(review["recovery"])
            self.assertTrue(review["replay"])
            with self.assertRaisesRegex(PermissionError, "another host/session"):
                dispatch(self.root, {**request, "session_id": "other"})
            source.write_text(IMPLEMENTATION + "\n# changed\n")
            with self.assertRaisesRegex(PermissionError, "Sources changed"):
                dispatch(self.root, request)
            source.write_text(IMPLEMENTATION)
            journal["execution"]["remote_job"] = {}
            atomic_json(path, journal)
            with self.assertRaisesRegex(PermissionError, "unknown"):
                dispatch(self.root, request)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "python/examples/flaggems").mkdir(parents=True)
        plan = prepare_operator_development(self.root, valid_spec())
        proposal = propose_operator_implementation(self.root, plan.development_id, IMPLEMENTATION, TEST_SOURCE, "fixture")
        self.request = {"kind": "development", "id": proposal.proposal_id, "session_id": "native-test"}
        self.env = patch.dict(os.environ, {"TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def review(self):
        return dispatch(self.root, {"action": "review", **self.request})

    def decide(self, review, outcome="allowed-once"):
        return dispatch(self.root, {"action": "decide", **self.request,
                                   "fingerprint": review["fingerprint"], "outcome": outcome})

    def test_review_is_read_only_then_exact_grant_applies(self):
        review = self.review()
        self.assertIn("test_square_new", review["reason"])
        self.assertFalse((self.root / "python/examples/flaggems/square_new.py").exists())
        self.assertEqual(apply_operator_implementation(self.root, self.request["id"]).status, "not_approved")
        self.decide(review)
        self.assertEqual(apply_operator_implementation(self.root, self.request["id"]).status, "applied")

    def test_review_change_rejects_grant(self):
        review = self.review()
        path, record = _load_proposal(self.root, self.request["id"])
        record["rationale"] = "changed"
        path.write_text(json.dumps(record))
        self.assertEqual(self.decide(review)["status"], "review_changed")
        self.assertEqual(apply_operator_implementation(self.root, self.request["id"]).status, "not_approved")

    def test_changed_files_require_baseline_confirmation_then_new_execution_approval(self):
        folder = self.root / "python/examples/flaggems"
        (folder / "square_new.py").write_text(IMPLEMENTATION + "\n# other task\n")
        (folder / "test_square_new.py").write_text(TEST_SOURCE)
        before = (folder / "square_new.py").read_bytes()
        review = self.review()
        self.assertTrue(review["source_change"])
        self.assertIn("replan_from_latest_sources", review["reason"])
        refreshed = dispatch(self.root, {"action": "refresh", **self.request,
            "fingerprint": review["fingerprint"], "outcome": "allowed-once"})
        self.assertEqual(refreshed["status"], "replanned")
        from codex_agent.operator_lifecycle import _load_receipt
        plan = _load_receipt(self.root, refreshed["plan"]["run_id"])
        self.assertEqual(plan["status"], "planned")
        self.assertNotEqual(plan.get("approval", {}).get("status"), "approved")
        self.assertEqual((folder / "square_new.py").read_bytes(), before)
        self.assertEqual(apply_operator_implementation(self.root, self.request["id"]).status, "not_approved")
        with self.assertRaisesRegex(PermissionError, "superseded"):
            self.review()

    def test_change_during_source_confirmation_reprompts_without_refreshing(self):
        source = self.root / "python/examples/flaggems/square_new.py"
        source.write_text(IMPLEMENTATION)
        review = self.review()
        source.write_text(IMPLEMENTATION + "\n# changed again\n")
        result = dispatch(self.root, {"action": "refresh", **self.request,
            "fingerprint": review["fingerprint"], "outcome": "allowed-once"})
        self.assertEqual(result["status"], "review_changed")
        self.assertEqual(_load_proposal(self.root, self.request["id"])[1]["status"], "pending_approval")

    def test_contract_change_during_review_is_detected(self):
        review = self.review()
        from codex_agent.operator_development import _load_request
        _, proposal = _load_proposal(self.root, self.request["id"])
        path, request = _load_request(self.root, proposal["development_id"])
        request["specification"]["semantics"] += " changed"
        path.write_text(json.dumps(request))
        self.assertEqual(self.decide(review)["status"], "review_changed")

    def test_validation_refresh_handles_a_plan_with_no_previous_approval(self):
        from codex_agent.operator_lifecycle import validate_operator_target, _load_receipt
        folder = self.root / "python/examples/flaggems"
        source = folder / "square_new.py"
        source.write_text(IMPLEMENTATION)
        (folder / "test_square_new.py").write_text(TEST_SOURCE)
        plan = validate_operator_target(self.root, "square_new")
        source.write_text(IMPLEMENTATION + "\n# new version\n")
        request = {"kind": "validation", "id": plan.run_id, "session_id": "native-test"}
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_VALIDATION": "1"}):
            review = dispatch(self.root, {"action": "review", **request})
            result = dispatch(self.root, {"action": "refresh", **request,
                "fingerprint": review["fingerprint"], "outcome": "allowed-once"})
        self.assertNotEqual(result["plan"]["run_id"], plan.run_id)
        self.assertEqual(_load_receipt(self.root, plan.run_id)["approval"]["status"], "superseded")

    def test_batch_refresh_creates_a_new_unapproved_snapshot(self):
        from codex_agent.project_tools import prepare_validation_job, load_job
        folder = self.root / "python/examples/flaggems"
        source = folder / "square_new.py"
        source.write_text(IMPLEMENTATION)
        (folder / "test_square_new.py").write_text(TEST_SOURCE)
        job = prepare_validation_job(self.root, ["square_new"], source_env=False)
        source.write_text(IMPLEMENTATION + "\n# A changed\n")
        request = {"kind": "job", "id": job["job_id"], "session_id": "native-test"}
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_VALIDATION": "1"}):
            review = dispatch(self.root, {"action": "review", **request})
            result = dispatch(self.root, {"action": "refresh", **request,
                "fingerprint": review["fingerprint"], "outcome": "allowed-once"})
        self.assertEqual(result["plan"]["approval"]["status"], "pending_approval")
        self.assertEqual(result["plan"]["results"], [])
        self.assertEqual(load_job(self.root, job["job_id"])["approval"]["status"], "superseded")

    def test_repair_conflict_does_not_apply_old_replacement(self):
        from codex_agent.operator_lifecycle import (validate_operator_target, propose_operator_repair,
                                                   apply_operator_repair, _load_receipt)
        folder = self.root / "python/examples/flaggems"
        source = folder / "square_new.py"
        source.write_text(IMPLEMENTATION)
        (folder / "test_square_new.py").write_text(TEST_SOURCE)
        plan = validate_operator_target(self.root, "square_new")
        path = self.root / plan.receipt_path
        receipt = json.loads(path.read_text())
        receipt.update(status="failed", exit_code=1, failure_stage="correctness")
        path.write_text(json.dumps(receipt))
        proposal = propose_operator_repair(self.root, plan.run_id, IMPLEMENTATION + "\n# B old replacement\n", "fixture")
        source.write_text(IMPLEMENTATION + "\n# A new baseline\n")
        request = {"kind": "repair", "id": proposal.proposal_id, "session_id": "native-test"}
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}):
            review = dispatch(self.root, {"action": "review", **request})
            result = dispatch(self.root, {"action": "refresh", **request,
                "fingerprint": review["fingerprint"], "outcome": "allowed-once"})
            self.assertEqual(apply_operator_repair(self.root, proposal.proposal_id).status, "not_approved")
        self.assertIn("A new baseline", source.read_text())
        self.assertNotIn("B old replacement", source.read_text())
        refreshed = _load_receipt(self.root, result["plan"]["run_id"])
        self.assertEqual(refreshed["status"], "planned")
        self.assertNotEqual(refreshed.get("approval", {}).get("status"), "approved")

    def test_rejection_cannot_apply(self):
        self.decide(self.review(), "rejected")
        self.assertEqual(apply_operator_implementation(self.root, self.request["id"]).status, "not_approved")

    def test_gate_disabled_and_unsupported_outcome_fail_closed(self):
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": "0"}):
            with self.assertRaises(PermissionError): self.review()
        with self.assertRaises(PermissionError): self.decide(self.review(), "model-said-yes")

    def test_another_session_cannot_reuse_approval(self):
        self.decide(self.review())
        self.request["session_id"] = "other"
        with self.assertRaisesRegex(PermissionError, "another"):
            self.review()

    def test_memory_delegates_to_existing_retrieval_and_excludes_current_run(self):
        with patch("codex_agent.harness.native_bridge.retrieve_operator_memory") as retrieve:
            retrieve.return_value.model_dump.return_value = {"status": "empty", "items": []}
            self.assertEqual(dispatch(self.root, {"action": "memory", "query": {"operator_name": "square", "run_id": "run-current"}})["status"], "empty")
            retrieve.assert_called_once_with(self.root, operator_name="square", run_id="run-current")

    def test_completed_apply_can_be_replayed_without_granting_new_authority(self):
        self.decide(self.review())
        first = apply_operator_implementation(self.root, self.request["id"])
        self.assertTrue(self.review()["replay"])
        second = apply_operator_implementation(self.root, self.request["id"])
        self.assertEqual(first.model_dump(), second.model_dump())
        (self.root / first.created_files[0]).write_text("# external change")
        with self.assertRaisesRegex(PermissionError, "Files changed"):
            apply_operator_implementation(self.root, self.request["id"])

    def test_approved_source_tampering_is_rejected(self):
        self.decide(self.review())
        path, record = _load_proposal(self.root, self.request["id"])
        record["implementation_source"] += "\n# changed after approval\n"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(PermissionError, "Approved content changed"):
            apply_operator_implementation(self.root, self.request["id"])

    def test_cancel_is_host_scoped_and_does_not_claim_remote_shutdown(self):
        self.decide(self.review())
        with self.assertRaisesRegex(PermissionError, "approving host"):
            dispatch(self.root, {"action": "cancel", **self.request, "session_id": "other"})
        result = dispatch(self.root, {"action": "cancel", **self.request})
        self.assertFalse(result["remote_shutdown_confirmed"])
        from codex_agent.process_control import ExecutionCancelled
        with self.assertRaises(ExecutionCancelled):
            apply_operator_implementation(self.root, self.request["id"])
        self.assertFalse((self.root / "python/examples/flaggems/square_new.py").exists())

    def test_unknown_write_requires_explicit_reconciliation_and_never_reuses_old_id(self):
        self.decide(self.review())
        with patch("codex_agent.operator_development._write_files_atomically", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                apply_operator_implementation(self.root, self.request["id"])
        with self.assertRaisesRegex(PermissionError, "outcome is unknown"):
            self.review()
        request = {"action": "reconcile", **self.request}
        with self.assertRaisesRegex(PermissionError, "explicit host confirmation"):
            dispatch(self.root, request)
        result = dispatch(self.root, {**request, "confirmed_stopped": True,
                                     "note": "Fixture: verified no processes and no target files exist."})
        self.assertEqual(result["status"], "abandoned")
        with self.assertRaisesRegex(PermissionError, "abandoned"):
            apply_operator_implementation(self.root, self.request["id"])
