"""Local lifecycle integration; model replies and execution output are synthetic."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from codex_agent import operator_development as development
from codex_agent import operator_lifecycle as domain
from codex_agent.orchestration.api import router
from codex_agent.orchestration.lifecycle import WorkflowLifecycle
from codex_agent.orchestration.roles import operator_paths
from codex_agent.tasks.context import defer_memory
from codex_agent.tasks.schema import artifacts, jobs, outbox
from codex_agent.tasks.store import Conflict, encode
from codex_agent.tests.test_operator_development import valid_spec, IMPLEMENTATION, TEST_SOURCE
from codex_agent.tests import test_orchestration_worker as worker_fixture
from codex_agent.validate_operator import OperatorValidationResult
from codex_agent.workers.handlers import execute_job


class WorkflowLifecycleTests(unittest.TestCase):
    submit = worker_fixture.WorkflowWorkerTests.submit
    execute = worker_fixture.WorkflowWorkerTests.execute
    task = worker_fixture.WorkflowWorkerTests.task

    def setUp(self):
        worker_fixture.WorkflowWorkerTests.setUp(self)
        self.cfg["permissions"] = dict(development=True, repair=True, validation=True)
        settings = patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg)})
        settings.start()
        self.addCleanup(settings.stop)
        self.request.update(operator="square_new", requirement="Compute elementwise square using the supplied specification.")
        self.backend.decision = "plan_tests"
        self.backend.change = self.reply
        self.lifecycle = WorkflowLifecycle(self.root)

    @staticmethod
    def reply(request, result):
        if request["phase"] == "planning_tests" and request["kind"] == "create":
            result["test_candidate"]["content"] = TEST_SOURCE
        if request["role"] == "implementer":
            result["candidate"]["content"] = IMPLEMENTATION

    def claim(self, job):
        event = self.jobs.db.rows(outbox, outbox.c.job_id == job["id"], order=(outbox.c.generation.desc(),))[0]
        return self.jobs.claim(self.jobs.envelope(event), job["kind"])

    def proposal(self):
        job = self.submit(specification=valid_spec())
        result = self.execute(job)
        self.assertEqual(result["status"], "succeeded", result)
        self.workflow_id = self.task(job)["id"]
        self.assertEqual(self.current()["state"], "waiting_approval")
        return self.action("publish")

    def current(self):
        return self.store.get(self.workflow_id)

    def action(self, action):
        return self.lifecycle.execute(self.workflow_id, action, expected_revision=self.current()["revision"])

    def approve(self):
        link = self.current()["result"]["lifecycle"]
        decide = development.decide_operator_development_proposal if link["proposal_kind"] == "development" else domain.decide_repair_proposal
        decide(self.root, link["proposal_id"], approve=True, reviewer="synthetic-test-human")

    def queued(self):
        self.proposal()
        self.approve()
        self.action("apply")
        self.action("plan-validation")
        plan = self.current()["result"]["lifecycle"]["validation_plan_id"]
        domain.decide_validation_plan(self.root, plan, approve=True, reviewer="synthetic-test-human")
        task = self.action("enqueue-validation")
        return self.jobs.get(task["result"]["lifecycle"]["validation_job_id"])

    def synthetic_execution(self, plan, *, passed=True):
        log = self.root / ("synthetic-" + plan.run_id + ".log")
        log.write_text("Synthetic execution boundary, NOT a hardware test.\n" + ("1 passed\n" if passed else "AssertionError: mismatched elements\n"))
        return OperatorValidationResult(operator="square_new", implementation_file=operator_paths("square_new")[0],
            test_files=[operator_paths("square_new")[1]], command=plan.command, dry_run=False,
            exit_code=0 if passed else 1, status="passed" if passed else "failed",
            failure_stage=None if passed else "correctness", likely_reason=None if passed else "reference mismatch",
            error_excerpt=[] if passed else ["AssertionError: mismatched elements"], duration_seconds=.1, log_path=str(log))

    def validation(self, job, *, passed=True, collect=True):
        plan_path = self.root / ("agent-results/operator-lifecycle/receipts/" + job["payload"]["approved_run_id"] + ".json")
        plan = SimpleNamespace(**json.loads(plan_path.read_text()))
        claim = self.claim(job)
        context = patch("codex_agent.orchestration.lifecycle.collect_finished_validation") if not collect else None
        if context:
            context.start()
        run_operator = domain.run_operator
        simulated = self.synthetic_execution(plan, passed=passed)
        def run(*args, **kwargs):
            return run_operator(*args, **kwargs) if kwargs["dry_run"] else simulated
        try:
            with patch("codex_agent.operator_lifecycle.run_operator", side_effect=run) as execution:
                execute_job(self.root, job["id"], claim["owner"])
            self.assertEqual(sum(not call.kwargs["dry_run"] for call in execution.call_args_list), 1)
        finally:
            if context:
                context.stop()
        return self.jobs.get(job["id"])

    def seed_failure(self):
        for relative, content in zip(operator_paths("square_new"), (IMPLEMENTATION.replace("values * values", "values + values"), TEST_SOURCE)):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        plan = domain.validate_operator_target(self.root, "square_new", execute=False)
        domain.decide_validation_plan(self.root, plan.run_id, approve=True, reviewer="synthetic-test-human")
        token = defer_memory.set(True)
        try:
            with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.synthetic_execution(plan, passed=False)):
                return domain.validate_operator_target(self.root, "square_new", execute=True, approved_run_id=plan.run_id)
        finally:
            defer_memory.reset(token)

    def test_create_review_approve_apply_queue_and_evidence_success(self):
        job = self.queued()
        self.assertEqual(self.current()["state"], "waiting_validation")
        self.assertEqual(self.action("collect-validation")["state"], "waiting_validation")
        result = self.validation(job)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(self.current()["state"], "succeeded", self.current())
        self.assertEqual(self.current()["result"]["lifecycle"]["validation_run_id"], result["result"]["run_id"])
        self.assertEqual(len(self.jobs.db.rows(artifacts, artifacts.c.job_id == job["id"])), 3)
        for relative in operator_paths("square_new"):
            self.assertTrue((self.root / relative).is_file())
        self.assertTrue((self.root / "tasks/operators/square_new.md").is_file())
        self.assertEqual(self.action("collect-validation")["state"], "succeeded")

    def test_failed_acceptance_is_failed_not_model_claimed_success(self):
        result = self.validation(self.queued(), passed=False)
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(self.current()["state"], "failed")
        self.assertEqual(self.current()["result"]["lifecycle"]["failure_stage"], "correctness")

    def test_repair_binds_failure_and_preserves_original_tests(self):
        failure = self.seed_failure()
        self.request["kind"] = "repair"
        job = self.submit(contract=worker_fixture.CONTRACT, failure_run_id=failure.run_id)
        self.execute(job)
        self.workflow_id = self.task(job)["id"]
        self.action("publish")
        self.approve()
        self.action("apply")
        source, test = operator_paths("square_new")
        self.assertEqual((self.root / source).read_text(), IMPLEMENTATION)
        self.assertEqual((self.root / test).read_text(), TEST_SOURCE)
        self.assertEqual(self.current()["result"]["lifecycle"]["proposal_kind"], "repair")
        self.assertTrue(any(e["id"] == "workflow_failure" for e in self.backend.calls[0]["evidence"]))

    def test_model_receives_exact_structured_specification(self):
        self.proposal()
        for call in self.backend.calls:
            spec = next(e for e in call["evidence"] if e["id"] == "workflow_specification")
            self.assertEqual(json.loads(spec["text"]), valid_spec())

    def test_unapproved_proposal_cannot_apply(self):
        self.proposal()
        with self.assertRaises(PermissionError):
            self.action("apply")
        self.assertFalse((self.root / operator_paths("square_new")[0]).exists())

    def test_permission_revocation_blocks_approved_proposal(self):
        self.proposal()
        self.approve()
        self.cfg["permissions"]["development"] = False
        with patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg)}), self.assertRaises(PermissionError):
            self.action("apply")

    def test_source_appearing_while_user_reviews_requires_new_proposal(self):
        self.proposal()
        self.approve()
        target = self.root / operator_paths("square_new")[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# changed by the user\n")
        with self.assertRaisesRegex(Conflict, "changed"):
            self.action("apply")
        self.assertEqual(target.read_text(), "# changed by the user\n")

    def test_task_document_changes_after_apply_block_validation(self):
        self.proposal()
        self.approve()
        self.action("apply")
        (self.root / "tasks/operators/square_new.md").write_text("# changed\n")
        with self.assertRaisesRegex(Conflict, "changed"):
            self.action("plan-validation")

    def test_duplicate_actions_do_not_reapply_and_stale_revision_is_rejected(self):
        self.proposal()
        self.approve()
        revision = self.current()["revision"]
        self.action("apply")
        with patch.object(development, "apply_operator_implementation", side_effect=AssertionError("must not repeat")):
            repeated = self.lifecycle.execute(self.workflow_id, "apply", expected_revision=revision)
        self.assertEqual(repeated["state"], "waiting_validation_approval")
        with self.assertRaisesRegex(Conflict, "revision changed"):
            self.lifecycle.execute(self.workflow_id, "plan-validation", expected_revision=revision)

    def test_write_completed_but_checkpoint_failed_is_fenced(self):
        self.proposal()
        self.approve()
        with patch.object(self.lifecycle, "_complete", side_effect=RuntimeError("lost checkpoint")), self.assertRaises(RuntimeError):
            self.action("apply")
        self.assertTrue((self.root / operator_paths("square_new")[0]).exists())
        self.assertEqual(self.current()["state"], "needs_reconciliation")
        self.assertEqual(self.current()["recovery_action"], "inspect_lifecycle_artifacts")
        with patch.object(development, "apply_operator_implementation", side_effect=AssertionError("must not repeat")), self.assertRaisesRegex(Conflict, "reconciliation"):
            self.action("apply")

    def test_validation_requires_separate_approval(self):
        self.proposal()
        self.approve()
        self.action("apply")
        self.action("plan-validation")
        with self.assertRaises(PermissionError):
            self.action("enqueue-validation")
        self.assertFalse(self.jobs.db.rows(jobs, jobs.c.kind == "validation"))

    def test_queue_and_parent_binding_roll_back_together(self):
        self.proposal()
        self.approve()
        self.action("apply")
        self.action("plan-validation")
        plan = self.current()["result"]["lifecycle"]["validation_plan_id"]
        domain.decide_validation_plan(self.root, plan, approve=True, reviewer="synthetic-test-human")
        with patch.object(self.lifecycle, "_checkpoint", side_effect=RuntimeError("rollback")), self.assertRaises(RuntimeError):
            self.action("enqueue-validation")
        self.assertFalse(self.jobs.db.rows(jobs, jobs.c.kind == "validation"))
        self.assertEqual(self.current()["state"], "waiting_validation_approval")
        self.action("enqueue-validation")
        self.assertEqual(len(self.jobs.db.rows(jobs, jobs.c.kind == "validation")), 1)

    def test_collection_retry_does_not_repeat_validation(self):
        job = self.queued()
        with patch("codex_agent.orchestration.lifecycle.collect_finished_validation", side_effect=RuntimeError("database unavailable")), self.assertWarns(RuntimeWarning):
            result = self.validation(job)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(self.current()["state"], "waiting_validation")
        with patch.object(domain, "run_operator", side_effect=AssertionError("must not repeat")):
            self.assertEqual(self.action("collect-validation")["state"], "succeeded")

    def test_edited_log_cannot_be_collected_as_success(self):
        result = self.validation(self.queued(), collect=False)
        (self.root / result["result"]["log_path"]).write_text("altered evidence\n")
        self.assertEqual(self.action("collect-validation")["state"], "needs_reconciliation")

    def test_missing_worker_receipt_artifact_blocks_success(self):
        job = self.queued()
        self.validation(job, collect=False)
        with self.jobs.db.transaction() as conn:
            conn.execute(artifacts.delete().where(artifacts.c.job_id == job["id"], artifacts.c.name == "receipt.json"))
        self.assertEqual(self.action("collect-validation")["state"], "needs_reconciliation")

    def test_another_tasks_job_cannot_be_collected(self):
        job = self.queued()
        payload = {**job["payload"], "workflow_id": "another-task"}
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, jobs, {"payload_json": encode(payload)}, jobs.c.id == job["id"])
        with self.assertRaisesRegex(Conflict, "does not belong"):
            self.action("collect-validation")

    def test_bare_passed_result_is_not_an_execution_receipt(self):
        job = self.queued()
        claim = self.claim(job)
        self.jobs.finish(job["id"], claim["owner"], "succeeded", {"status": "passed", "run_id": "run-made-up"})
        self.assertEqual(self.action("collect-validation")["state"], "needs_reconciliation")

    def test_failed_worker_without_receipt_requires_reconciliation(self):
        job = self.queued()
        claim = self.claim(job)
        self.jobs.finish(job["id"], claim["owner"], "failed", {"reason": "environment unavailable"})
        self.assertEqual(self.action("collect-validation")["state"], "needs_reconciliation")

    def test_repair_rejects_a_planned_or_mutated_failure_receipt(self):
        failure = self.seed_failure()
        self.request["kind"] = "repair"
        plan = domain.validate_operator_target(self.root, "square_new", execute=False)
        with self.assertRaises(Conflict):
            self.submit(contract=worker_fixture.CONTRACT, failure_run_id=plan.run_id)
        job = self.submit(contract=worker_fixture.CONTRACT, failure_run_id=failure.run_id)
        self.execute(job)
        self.workflow_id = self.task(job)["id"]
        path = self.root / failure.receipt_path
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(Conflict, "receipt changed"):
            self.action("publish")

    def test_no_spec_legacy_proposal_cannot_publish_or_guess(self):
        job = self.submit(contract=worker_fixture.CONTRACT)
        self.execute(job)
        self.workflow_id = self.task(job)["id"]
        with self.assertRaisesRegex(Conflict, "structured specification"):
            self.action("publish")
        self.assertNotIn("lifecycle", self.current()["result"])

    def test_spec_for_another_operator_is_rejected(self):
        spec = {**valid_spec(), "name": "another_operator"}
        with self.assertRaisesRegex(Conflict, "different task"):
            self.submit(specification=spec)
        self.assertFalse(self.jobs.db.rows(jobs))

    def test_semantic_role_cannot_replace_the_frozen_contract(self):
        self.backend.decision = "request_semantics"
        result = self.execute(self.submit(specification=valid_spec()))
        self.assertEqual(result["result"]["workflow_state"], "blocked")
        self.assertFalse((self.root / operator_paths("square_new")[0]).exists())

    def test_missing_role_artifact_blocks_publication(self):
        job = self.submit(specification=valid_spec())
        self.execute(job)
        self.workflow_id = self.task(job)["id"]
        step = self.current()["steps"][-1]
        with self.jobs.db.transaction() as conn:
            conn.execute(artifacts.delete().where(artifacts.c.id == step["result"]["artifact_id"]))
        with self.assertRaisesRegex(Conflict, "artifact"):
            self.action("publish")

    def test_reviewed_proposal_tampering_blocks_apply(self):
        self.proposal()
        self.approve()
        link = self.current()["result"]["lifecycle"]
        path = development._proposal_path(self.root, link["proposal_id"])
        record = json.loads(path.read_text())
        record["implementation_source"] += "\n# unreviewed change\n"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(Conflict, "changed"):
            self.action("apply")

    def test_post_validation_source_change_cannot_be_reported_as_success(self):
        self.validation(self.queued(), collect=False)
        (self.root / operator_paths("square_new")[0]).write_text("# changed after test\n")
        self.assertEqual(self.action("collect-validation")["state"], "needs_reconciliation")

    def test_another_plan_cannot_substitute_for_bound_validation(self):
        result = self.validation(self.queued(), collect=False)
        path = self.root / result["result"]["receipt_path"]
        receipt = json.loads(path.read_text())
        receipt["approved_run_id"] = "run-another-plan"
        path.write_text(json.dumps(receipt))
        self.assertEqual(self.action("collect-validation")["state"], "needs_reconciliation")

    def test_default_gate_rejects_http_submission_without_model_call(self):
        self.cfg["queue"].pop("proposalWorkflows")
        app = FastAPI()
        app.include_router(router(self.root))
        with patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg)}), TestClient(app) as client:
            body = dict(request=self.request, request_id="disabled", specification=valid_spec())
            self.assertEqual(client.post("/api/operator-workflows", json=body).status_code, 403)
        self.assertFalse(self.jobs.db.rows(jobs))

    def test_published_lifecycle_cannot_be_orphaned_by_generic_state_controls(self):
        self.proposal()
        task = self.current()
        with self.assertRaises(Conflict):
            self.store.revise_inputs(task["id"], {**task["versions"], "source_sha256": "0" * 64},
                reason="changed", expected_revision=task["revision"], event_id="change")
        with self.assertRaises(Conflict):
            self.store.close(task["id"], "cancelled", reason="changed", expected_revision=task["revision"], event_id="close")
        self.assertEqual(self.current()["revision"], task["revision"])

    def test_host_http_submit_and_revision_only_actions(self):
        from codex_agent.harness import HarnessAgent, HarnessSettings
        from codex_agent.platform.api import create_app
        from codex_agent.tests.test_platform_api import FakeHarnessBackend
        app = create_app(self.root, harness_agent=HarnessAgent(HarnessSettings.from_env(self.root, {}), FakeHarnessBackend()))
        with TestClient(app) as client:
            body = dict(request=self.request, request_id="http-fixture", specification=valid_spec())
            response = client.post("/api/operator-workflows", json=body)
            self.assertEqual(response.status_code, 202, response.text)
            job = response.json()
            self.execute(job)
            task = self.task(job)
            path = "/api/operator-workflows/" + task["id"]
            self.assertEqual(client.get(path).json()["state"], "waiting_approval")
            response = client.post(path + "/publish", json={"expected_revision": task["revision"]})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("proposal_id", response.json()["result"]["lifecycle"])
            self.assertEqual(client.post(path + "/apply", json={"expected_revision": response.json()["revision"]}).status_code, 403)
            self.assertEqual(client.post(path + "/succeed", json={"expected_revision": task["revision"]}).status_code, 422)
            self.assertEqual(client.post(path + "/apply", json={"expected_revision": task["revision"], "approved": True}).status_code, 422)
            self.assertEqual(client.post("/api/operator-workflows", json={"request": self.request, "request_id": "missing-spec"}).status_code, 400)
            proposal_id = response.json()["result"]["lifecycle"]["proposal_id"]
            self.assertEqual(client.get("/api/development-proposals/" + proposal_id).status_code, 200)
            approved = client.post("/api/development-proposals/" + proposal_id + "/decision",
                json={"approve": True, "reviewer": "synthetic-test-human"})
            self.assertEqual(approved.status_code, 200, approved.text)
            for action in ("apply", "plan-validation"):
                response = client.post(path + "/" + action, json={"expected_revision": response.json()["revision"]})
                self.assertEqual(response.status_code, 200, response.text)
            plan_id = response.json()["result"]["lifecycle"]["validation_plan_id"]
            approved = client.post("/api/validation-plans/" + plan_id + "/decision",
                json={"approve": True, "reviewer": "synthetic-test-human"})
            self.assertEqual(approved.status_code, 200, approved.text)
            response = client.post(path + "/enqueue-validation", json={"expected_revision": response.json()["revision"]})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["state"], "waiting_validation")


if __name__ == "__main__":
    unittest.main()
