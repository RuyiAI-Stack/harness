"""Host-only bridge from role proposals to existing approval and execution services.

File operations and MySQL cannot commit together. Persist intent first; an
interrupted operation is fenced for reconciliation instead of blindly replayed.
"""
import json
from pathlib import Path

from codex_agent import operator_development as development
from codex_agent import operator_lifecycle as lifecycle
from codex_agent.artifacts import ArtifactStore
from codex_agent.execution_guard import (approval_digest, artifact, file_snapshot,
                                        inspect_execution, resource_locks)
from codex_agent.runtime_config import permission_enabled, runtime_config
from codex_agent.tasks.schema import artifacts
from codex_agent.tasks.service import TaskService
from codex_agent.tasks.store import Conflict, JobStore, digest, encode
from codex_agent.validation_evidence import audit_validation_receipt, workspace_artifact_sha256
from .evidence import load_role_output
from .models import Identifier, TERMINAL
from .roles import operator_paths, sha256
from .service import WorkflowSeed, contract_from_spec, snapshot_files
from .service import WorkflowService
from .schema import tasks
from .store import WorkflowStore


class WorkflowLifecycle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.jobs = JobStore(self.root)
        self.store = WorkflowStore(self.root, database=self.jobs.db)

    def view(self, task_id):
        task = self.store.get(task_id)
        link = task["result"].get("lifecycle", {})
        task.update(proposal=None, validation_plan=None, acceptance=None, artifact_error=None)
        try:
            if link.get("proposal_id"):
                task["proposal"] = (development.get_operator_development_proposal(self.root, link["proposal_id"])
                    if link.get("proposal_kind") == "development" else lifecycle.get_repair_proposal(self.root, link["proposal_id"]))
                seed, *_ = self._bundle(task, check_snapshot=False)
                task["acceptance"] = (seed.specification or seed.contract).model_dump(mode="json")
            if link.get("validation_plan_id"):
                task["validation_plan"] = lifecycle.get_validation_plan(self.root, link["validation_plan_id"])
        except (ValueError, OSError, Conflict) as error:
            task["artifact_error"] = "Cannot verify saved review artifacts: " + type(error).__name__
        job_id = task["result"].get("agent_job_id") or (task["steps"][-1]["job_id"] if task["steps"] else None)
        task["job"] = self.jobs.get(job_id) if job_id else None
        return task

    def _checkpoint(self, task, action, data, *, state=None, connection=None):
        def change(conn, row):
            result = json.loads(row["result_json"])
            result["lifecycle"] = {**result.get("lifecycle", {}), **data}
            return {"result_json": encode(result), "state": state or row["state"]}, {}
        return self.store._mutate(task["id"], task["revision"],
            f"lifecycle:{action}:{task['revision']}", "lifecycle", {"action": action, **data}, change, connection=connection)

    def execute(self, task_id, action, *, expected_revision):
        actions = {"publish": self._publish, "apply": self._apply, "plan-validation": self._plan,
                   "enqueue-validation": self._enqueue, "collect-validation": self._collect}
        if action not in actions:
            raise ValueError("unknown workflow lifecycle action")
        if not runtime_config().queue.enabled or not runtime_config().queue.proposalWorkflows:
            raise PermissionError("proposal workflows are disabled")
        # IDs only identify server-owned records; callers never provide a proposal,
        # command or receipt ID to these actions.
        from pydantic import TypeAdapter
        TypeAdapter(Identifier).validate_python(task_id, strict=True)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be positive")
        with resource_locks(self.root, ["workflow:" + task_id], wait_seconds=5):
            task = self.store.get(task_id)
            link = task["result"].get("lifecycle", {})
            if link.get("operation"):
                raise Conflict("interrupted lifecycle operation requires reconciliation; do not replay")
            done = link.get("completed_actions", {})
            if action in done and expected_revision == done[action]:
                return task
            if task["revision"] != expected_revision:
                raise Conflict("workflow revision changed; reload and review again")
            if task["state"] in TERMINAL:
                if action == "collect-validation":
                    return task
                raise Conflict("workflow is closed")
            return actions[action](task)

    @staticmethod
    def _state(task, expected):
        if task["state"] != expected:
            raise Conflict("workflow is not in " + expected)

    def _begin(self, task, action, *, state=None):
        return self._checkpoint(task, action + "_started", {"operation": {
            "action": action, "requested_revision": task["revision"], "status": "started"}}, state=state)

    def _complete(self, task, action, data, *, state):
        link = task["result"]["lifecycle"]
        done = {**link.get("completed_actions", {}), action: link["operation"]["requested_revision"]}
        return self._checkpoint(task, action + "_completed",
            {**data, "operation": None, "completed_actions": done}, state=state)

    def _unknown(self, task, error):
        operation = {**task["result"]["lifecycle"]["operation"], "status": "unknown",
                     "error_type": type(error).__name__}
        self._checkpoint(task, "reconciliation_required", {"operation": operation,
            "reason": "Inspect domain artifacts before authorizing any new operation."}, state="needs_reconciliation")

    def _bundle(self, task, *, check_snapshot=True):
        seed = WorkflowSeed.model_validate(self.store.history(task["id"], limit=1)[0]["payload"]["context"])
        outputs = {}
        for step in task["steps"]:
            if step["status"] != "completed":
                raise Conflict("all proposal steps must be completed and current")
            job = self.jobs.get(step["job_id"])
            if job["status"] != "succeeded" or job["payload"].get("workflow_id") != task["id"]:
                raise Conflict("proposal job is not completed or belongs to another workflow")
            result = load_role_output(self.jobs.db, task["id"], step)
            if result is None or result.status != "completed" or result.unknowns:
                raise Conflict("proposal evidence is incomplete")
            outputs[step["role"], step["state"]] = result
        review = outputs.get(("reviewer", "reviewing"))
        semantic = outputs.get(("semantic", "preparing"))
        if seed.specification is None and semantic and semantic.specification is not None:
            seed = seed.model_copy(update={"specification": semantic.specification,
                "contract": contract_from_spec(semantic.specification)})
        elif seed.contract is None and semantic:
            seed = seed.model_copy(update={"contract": semantic.contract})
        if seed.contract is None or digest(seed.contract.model_dump(mode="json")) != task["versions"]["contract_sha256"]:
            raise Conflict("review contract does not match the frozen workflow version")
        implementation = outputs.get(("implementer", "implementing"))
        tests = outputs.get(("reviewer", "planning_tests"))
        if tests is None and task["result"].get("preparation_reuse", {}).get("status") == "reused":
            from .reuse import load_reused_tests
            tests = load_reused_tests(self.jobs.db, task, seed)
        if not review or review.verdict != "acceptable" or not implementation or not tests:
            raise Conflict("candidate needs completed implementation, fixed tests and independent review")
        files = {implementation.candidate.path: implementation.candidate.content,
                 tests.test_candidate.path: tests.test_candidate.content}
        source, test = operator_paths(task["operator"])
        if (set(files) != {source, test} or sha256(files[source]) != task["versions"]["source_sha256"]
                or sha256(files[test]) != task["versions"]["tests_sha256"]):
            raise Conflict("candidate hashes do not match workflow versions")
        if check_snapshot and snapshot_files(self.root, task["operator"]) != seed.files:
            raise Conflict("source or tests changed since proposal generation; submit a revised task")
        return seed, files, implementation.summary

    def _publish(self, task):
        self._state(task, "waiting_approval")
        if task["result"].get("lifecycle", {}).get("proposal_id"):
            self._proposal(task)
            return task
        seed, files, rationale = self._bundle(task)
        source, test = operator_paths(task["operator"])
        if task["kind"] == "create":
            if seed.specification is None:
                raise Conflict("new-operator publication requires a structured specification frozen at submission")
            if digest(contract_from_spec(seed.specification).model_dump(mode="json")) != task["versions"]["contract_sha256"]:
                raise Conflict("structured specification no longer matches the contract")
        else:
            if not seed.failure_run_id:
                raise Conflict("repair publication requires an audited failure bound at submission")
            checked = audit_validation_receipt(self.root, {"run_id": seed.failure_run_id, "operator": task["operator"]})
            if (checked.verdict != "verified-failed" or workspace_artifact_sha256(self.root, checked.receipt_path)
                    != seed.failure_receipt_sha256):
                raise Conflict("failure receipt changed or no longer matches source")
            if not lifecycle.diagnose_failure_run(self.root, seed.failure_run_id).source_repair_allowed:
                raise Conflict("failure is not eligible for source repair")
            if files[test] != next(f.content for f in seed.files if f.path == test):
                raise Conflict("repair cannot change acceptance tests")
        task = self._begin(task, "publish")
        try:
            if task["kind"] == "create":
                prepared = development.prepare_operator_development(self.root, seed.specification)
                proposal = development.propose_operator_implementation(self.root, prepared.development_id,
                    files[source], files[test], rationale)
                kind = "development"
            else:
                proposal = lifecycle.propose_operator_repair(self.root, seed.failure_run_id, files[source], rationale)
                kind = "repair"
            record = artifact(self.root, kind, proposal.proposal_id)
            return self._complete(task, "publish", {"proposal_id": proposal.proposal_id, "proposal_kind": kind,
                "proposal_digest": approval_digest(self.root, kind, record),
                "candidate_snapshot": {path: sha256(content) for path, content in files.items()}}, state="waiting_approval")
        except Exception as error:
            self._unknown(task, error)
            raise

    def _proposal(self, task):
        link = task["result"].get("lifecycle", {})
        if not link.get("proposal_id"):
            raise Conflict("publish a proposal before approval or application")
        kind = "development" if task["kind"] == "create" else "repair"
        if link["proposal_kind"] != kind:
            raise Conflict("proposal type differs from task")
        record = artifact(self.root, kind, link["proposal_id"])
        if record["operator"] != task["operator"] or approval_digest(self.root, kind, record) != link["proposal_digest"]:
            raise Conflict("proposal content or contract changed; request a new review")
        return link, record

    def _apply(self, task):
        self._state(task, "waiting_approval")
        link, record = self._proposal(task)
        if record["status"] != "approved" or record.get("approval_digest") != link["proposal_digest"]:
            raise PermissionError("the bound proposal must be approved by the host")
        if not permission_enabled(link["proposal_kind"]):
            raise PermissionError("host has not enabled this write capability")
        self._bundle(task)
        task = self._begin(task, "apply", state="applying")
        try:
            apply = development.apply_operator_implementation if task["kind"] == "create" else lifecycle.apply_operator_repair
            result = apply(self.root, link["proposal_id"])
            if result.status not in {"applied", "already_applied"}:
                raise Conflict("domain service did not apply the candidate")
            if file_snapshot(self.root, list(link["candidate_snapshot"])) != link["candidate_snapshot"]:
                raise Conflict("applied code differs from the reviewed candidate")
            return self._complete(task, "apply", {"application": result.model_dump(mode="json")},
                                  state="waiting_validation_approval")
        except Exception as error:
            self._unknown(task, error)
            raise

    def _applied(self, task):
        link, record = self._proposal(task)
        journal = inspect_execution(self.root, link["proposal_kind"], link["proposal_id"])
        if record["status"] != "applied" or not journal or journal["state"] != "completed":
            raise Conflict("candidate application has no completed execution journal")
        if (file_snapshot(self.root, list(link["candidate_snapshot"])) != link["candidate_snapshot"]
                or file_snapshot(self.root, list(journal["files_after"])) != journal["files_after"]):
            raise Conflict("reviewed code, test or task document changed after application")
        return link

    def _plan(self, task):
        self._state(task, "waiting_validation_approval")
        link = self._applied(task)
        if link.get("validation_plan_id"):
            self._plan_record(task)
            return task
        task = self._begin(task, "plan-validation")
        try:
            plan = lifecycle.validate_operator_target(self.root, task["operator"], execute=False)
            record = artifact(self.root, "validation", plan.run_id)
            checked = audit_validation_receipt(self.root, plan.model_dump())
            if checked.verdict != "planned" or record["source_snapshot"] != link["candidate_snapshot"]:
                raise Conflict("validation plan does not match the reviewed source and tests")
            return self._complete(task, "plan-validation", {"validation_plan_id": plan.run_id,
                "validation_plan_digest": approval_digest(self.root, "validation", record)}, state="waiting_validation_approval")
        except Exception as error:
            self._unknown(task, error)
            raise

    def _plan_record(self, task):
        link = task["result"]["lifecycle"]
        if not link.get("validation_plan_id"):
            raise Conflict("create a validation plan before requesting execution")
        record = artifact(self.root, "validation", link["validation_plan_id"])
        if (record["operator"] != task["operator"] or record["source_snapshot"] != link["candidate_snapshot"]
                or approval_digest(self.root, "validation", record) != link["validation_plan_digest"]):
            raise Conflict("validation plan changed or belongs to different code")
        return record

    def _enqueue(self, task):
        self._state(task, "waiting_validation_approval")
        link = self._applied(task)
        self._plan_record(task)
        # The queue job, Outbox event and workflow binding commit in one transaction.
        with self.jobs.db.transaction() as conn:
            job = TaskService(self.root).submit_validation(task["operator"], link["validation_plan_id"],
                "workflow-validation:" + task["id"], connection=conn, workflow_id=task["id"])
            return self._checkpoint(task, "validation_queued", {"validation_job_id": job["id"],
                "completed_actions": {**link["completed_actions"], "enqueue-validation": task["revision"]}},
                state="waiting_validation", connection=conn)

    def _collect(self, task):
        self._state(task, "waiting_validation")
        link = task["result"]["lifecycle"]
        job = self.jobs.get(link["validation_job_id"])
        if (job["kind"] != "validation" or job["payload"].get("workflow_id") != task["id"]
                or job["payload"].get("approved_run_id") != link["validation_plan_id"]):
            raise Conflict("validation job does not belong to this workflow and plan")
        if job["status"] in {"queued", "running"}:
            return task
        if job["status"] not in {"succeeded", "failed"} or not job["result"].get("run_id"):
            return self._checkpoint(task, "validation_unresolved", {"validation_job_status": job["status"],
                "reason": "No completed validation evidence; inspect the worker result before retrying."},
                state="cancelled" if job["status"] == "cancelled" else "needs_reconciliation")
        try:
            self._applied(task)
            plan = self._plan_record(task)
            reference = job["result"]
            checked = audit_validation_receipt(self.root, reference)
            receipt = artifact(self.root, "validation", reference["run_id"])
            records = self.jobs.db.rows(artifacts, artifacts.c.job_id == job["id"], artifacts.c.name == "receipt.json",
                                       artifacts.c.attempt == job["attempt"])
            if (checked.verdict not in {"verified-passed", "verified-failed"} or not records
                    or checked.operator != task["operator"]
                    or receipt.get("approved_run_id") != link["validation_plan_id"]
                    or plan.get("approval", {}).get("execution_run_id") != reference["run_id"]
                    or receipt["source_snapshot"] != link["candidate_snapshot"]
                    or records[0]["sha256"] != workspace_artifact_sha256(self.root, checked.receipt_path)):
                raise Conflict("validation receipt, log, source or plan binding failed")
            ArtifactStore(self.root).read(records[0])
            if (job["status"] == "succeeded") != checked.success:
                raise Conflict("job status disagrees with execution evidence")
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            return self._checkpoint(task, "validation_evidence_rejected", {"reason": str(error)[:2000]},
                                    state="needs_reconciliation")
        data = {"validation_evidence": checked.model_dump(mode="json"),
            "validation_run_id": checked.run_id, "failure_stage": receipt.get("failure_stage"),
            "reason": "Bound acceptance tests passed." if checked.success else "Acceptance tests failed; inspect the recorded failure."}
        if not checked.success and task["request"].get("auto_repair"):
            return self._repair_followup(task, checked.run_id, data)
        return self._checkpoint(task, "validation_collected", data, state="succeeded" if checked.success else "failed")

    def _repair_followup(self, task, failure_run_id, data):
        if task["repair_count"] >= task["max_repairs"]:
            return self._checkpoint(task, "repair_stopped", {**data, "stop_reason": "repair_budget_exhausted",
                "reason": "Repair budget exhausted. Inspect the latest failure and confirm requirements before starting a new task."}, state="failed")
        diagnosis = lifecycle.diagnose_failure_run(self.root, failure_run_id)
        if not diagnosis.source_repair_allowed:
            return self._checkpoint(task, "repair_stopped", {**data, "stop_reason": "not_source_repairable",
                "reason": diagnosis.recommended_action}, state="failed")
        seed = WorkflowSeed.model_validate(self.store.history(task["id"], limit=1)[0]["payload"]["context"])
        contract = seed.contract
        for step in task["steps"]:
            if step["role"] == "semantic":
                output = load_role_output(self.jobs.db, task["id"], step)
                if output:
                    contract = contract_from_spec(output.specification) if output.specification else output.contract
        if contract is None:
            raise Conflict("repair cannot proceed without the frozen contract")
        request = {**task["request"], "kind": "repair", "versions": {}, "auto_publish": True}
        # Each attempted repair has its own immutable workflow. A failed attempt
        # stays failed; the child, its queue message and parent link commit together.
        with self.jobs.db.transaction() as conn:
            job = WorkflowService(self.root).submit(request, "workflow-repair:" + task["id"],
                contract=contract, failure_run_id=failure_run_id, connection=conn,
                exclude_source_runs=seed.exclude_source_runs)
            child_id = job["payload"]["workflow_id"]
            self.jobs.db.update(conn, tasks, {"repair_count": task["repair_count"] + 1,
                "result_json": encode({"parent_workflow_id": task["id"], "agent_job_id": job["id"]})}, tasks.c.id == child_id)
            return self._checkpoint(task, "repair_queued", {**data, "child_workflow_id": child_id,
                "reason": "Validation failed. A bounded repair proposal is queued; future writes still require approval."},
                state="failed", connection=conn)


def collect_finished_validation(root, workflow_id, job_id):
    service = WorkflowLifecycle(root)
    task = service.store.get(workflow_id)
    if task["result"].get("lifecycle", {}).get("validation_job_id") != job_id:
        raise Conflict("cannot collect a different workflow's validation job")
    return service.execute(workflow_id, "collect-validation", expected_revision=task["revision"])
