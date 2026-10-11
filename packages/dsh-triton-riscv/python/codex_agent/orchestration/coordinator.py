"""Checkpointed proposal orchestration with an explicitly injected role adapter.

There is deliberately no default model adapter or API configuration lookup here.
"""
from dataclasses import asdict
import json

from codex_agent.artifacts import ArtifactStore
from codex_agent.storage.schema import runs
from codex_agent.tasks.store import Conflict, LeaseLost, digest, encode
from .models import InputVersions, NEXT_STATE, TERMINAL, WAITING
from .prompts import system_prompt, task_prompt
from .roles import CandidateFile, Evidence, FileSnapshot, RoleRequest, operator_paths, sha256
from .evidence import load_role_output
from .service import WorkflowSeed, contract_from_spec, snapshot_files
from codex_agent.develop_operator import parse_operator_spec
from .retrieval import role_evidence
from .store import WorkflowStore


class ProposalCoordinator:
    def __init__(self, jobs, job, owner, *, agent, retriever=None):
        self.jobs, self.job, self.owner, self.agent = jobs, job, owner, agent
        self.root = jobs.db.root
        self.store = WorkflowStore(self.root, database=jobs.db)
        self.artifacts = ArtifactStore(self.root)
        self.payload = json.loads(job["payload_json"])
        self.task_id = self.payload["workflow_id"]
        self.retriever = retriever or role_evidence

    def guard(self, conn):
        job = self.jobs.require_owner(conn, self.job["id"], self.owner)
        if job["cancel_requested"]:
            raise LeaseLost("workflow cancellation requested")

    def _bind(self, task):
        run = self.jobs.db.rows(runs, runs.c.id == self.job["run_id"])
        if (self.job["kind"] != "agent" or self.payload.get("mode") != "operator-workflow"
                or self.payload.get("session_id") != task["session_id"]
                or self.payload.get("run_id") != self.job["run_id"] or not run
                or run[0]["session_id"] != task["session_id"] or run[0]["operator"] != task["operator"]
                or json.loads(run[0]["request_json"]).get("workflow_id") != task["id"]):
            raise Conflict("job, run and workflow bindings do not match")
        created = self.store.history(task["id"], limit=1)[0]["payload"]
        if self.payload.get("fingerprint") != digest([created["request"], created["context"]]):
            raise Conflict("workflow input fingerprint changed")
        return WorkflowSeed.model_validate(created["context"])

    def _stop(self, task, reason):
        return self.store.close(task["id"], "blocked", reason=reason[:2000],
            expected_revision=task["revision"], event_id="stop-" + str(task["revision"]))

    @staticmethod
    def _blocked_reason(result, fallback):
        if result is None:
            return fallback
        if result.unknowns:
            return "Unresolved requirements: " + "; ".join(result.unknowns)
        return result.summary

    def _load_output(self, step):
        if step["job_id"] != self.job["id"]:
            raise Conflict("role step belongs to another job")
        return load_role_output(self.jobs.db, self.task_id, step)

    def _call(self, task, role, seed, contract, files):
        previous = [s for s in task["steps"] if s["input_epoch"] == task["input_epoch"]
                    and s["state"] == task["state"] and s["role"] == role]
        if previous:
            if previous[-1]["status"] == "running":
                raise Conflict("unfinished model call requires reconciliation, not automatic replay")
            return task, self._load_output(previous[-1])
        evidence = list(seed.evidence)
        if seed.specification:
            evidence.append(Evidence(id="workflow_specification", kind="contract", source="frozen development specification",
                text=encode(seed.specification.model_dump(mode="json")), verified=False))
        if seed.failure_context:
            evidence.append(seed.failure_context)
        if contract:
            evidence.append(Evidence(id="workflow_contract", kind="contract", source="workflow contract proposal",
                text=encode(contract.model_dump(mode="json")), verified=False))
        historical, retrieval = self.retriever(self.root, task, seed, role, task["state"])
        evidence.extend(historical)
        if role == "implementer" and task["result"].get("review_feedback"):
            evidence.append(Evidence(id="workflow_review", kind="requirement", source="previous candidate review",
                text=encode(task["result"]["review_feedback"])))
        candidate, baseline = None, None
        if task["state"] == "reviewing":
            proposed = files[operator_paths(task["operator"])[0]]
            baseline = next((f for f in seed.files if f.path == proposed.path), None)
            candidate = CandidateFile(path=proposed.path, content=proposed.content,
                                      base_sha256=baseline.sha256 if baseline else None)
        request = RoleRequest(task_id=task["id"], step_id="preflight", input_epoch=task["input_epoch"],
            role=role, phase=task["state"], kind=task["kind"], operator=task["operator"],
            requirement=task["request"]["requirement"], versions=InputVersions.model_validate(task["versions"]),
            evidence=evidence, files=list(files.values()), candidate=candidate, review_baseline=baseline)
        task_prompt(request)
        task = self.store.start_step(task["id"], {"role": role, "instruction": task["request"]["requirement"],
            "run_id": self.job["run_id"], "job_id": self.job["id"]},
            expected_revision=task["revision"], event_id="start-" + str(task["revision"]))
        step = task["steps"][-1]
        request = request.model_copy(update={"step_id": step["id"]})
        audit = {"request": request.model_dump(mode="json"), "retrieval": retrieval, "system_prompt": system_prompt(role),
                 "task_prompt": task_prompt(request)}
        self.jobs.begin_effects(self.job["id"], self.owner)
        notifications, event_bytes, omitted = [], 0, 0
        def on_event(event):
            nonlocal event_bytes, omitted
            size = len(encode(event).encode())
            if event_bytes + size <= 1024 * 1024:
                notifications.append(event)
                event_bytes += size
            else:
                omitted += 1
        try:
            value = self.agent.run(request, on_event=on_event)
        except Exception as error:
            # An interrupted request has an unknown outcome; do not auto-repeat it.
            audit.update(outcome="unknown", error_type=type(error).__name__,
                         notifications=notifications, omitted_notifications=omitted)
            self.artifacts.put(self.jobs, self.job["id"], self.owner, step["id"] + ".json", encode(audit).encode())
            raise
        if (value.task_id, value.step_id, value.input_epoch, value.role) != (
                request.task_id, request.step_id, request.input_epoch, request.role):
            raise Conflict("role returned a foreign task or step binding")
        audit.update(outcome=value.outcome, raw=asdict(value.harness), prompt_version=value.prompt_version,
                     prompt_sha256=value.prompt_sha256, error=value.error)
        artifact_id = self.artifacts.put(self.jobs, self.job["id"], self.owner,
                                        step["id"] + ".json", encode(audit).encode())
        result = value.result if value.outcome == "validated" else None
        task = self.store.finish_step(task["id"], step["id"], {
            "status": result.status if result else "failed", "summary": result.summary if result else "Role output rejected",
            "evidence_refs": result.evidence_refs if result else [], "artifact_id": artifact_id},
            expected_revision=task["revision"], event_id="finish-" + step["id"])
        return task, result

    def run(self):
        previous_guard = self.jobs.db.task_guard
        self.jobs.db.task_guard = self.guard
        try:
            return self._run()
        finally:
            self.jobs.db.task_guard = previous_guard

    def _run(self):
        task = self.store.get(self.task_id)
        seed = self._bind(task)
        if any(s["status"] == "superseded" for s in task["steps"]):
            raise Conflict("workflow inputs were revised; submit a new frozen workflow")
        if any(s["status"] == "running" for s in task["steps"]):
            raise Conflict("unfinished model call requires reconciliation, not automatic replay")
        contract = seed.contract
        files = {f.path: f for f in seed.files}
        # Restore virtual files from durable role artifacts, not a sibling run.
        for step in task["steps"]:
            result = self._load_output(step)
            if result is None:
                continue
            if step["role"] == "semantic":
                contract = result.contract
                if result.specification is not None:
                    contract = contract_from_spec(result.specification)
                    seed = seed.model_copy(update={"specification": result.specification})
            candidate = getattr(result, "candidate", None) or getattr(result, "test_candidate", None)
            if candidate:
                files[candidate.path] = FileSnapshot(path=candidate.path, content=candidate.content,
                                                      sha256=sha256(candidate.content))
        # Fixed preparation/testing plus bounded implementation-review pairs.
        for _ in range(4 + 2 * task["max_repairs"]):
            if task["state"] in TERMINAL | WAITING:
                break
            if snapshot_files(self.root, task["operator"]) != seed.files:
                task = self._stop(task, "Source or acceptance tests changed since submission; submit a new plan.")
                break
            role = {"preparing": "main", "planning_tests": "reviewer",
                    "implementing": "implementer", "reviewing": "reviewer"}[task["state"]]
            task, result = self._call(task, role, seed, contract, files)
            if result is None or result.status != "completed" or result.unknowns:
                task = self._stop(task, self._blocked_reason(result, "Role output failed validation."))
                break
            if task["state"] == "preparing":
                if result.decision in {"ask_user", "blocked"}:
                    task = self._stop(task, " ".join(result.questions) or result.summary)
                    break
                if result.decision == "request_semantics" or contract is None:
                    task, result = self._call(task, "semantic", seed, contract, files)
                    if result is None or result.status != "completed" or result.unknowns:
                        task = self._stop(task, self._blocked_reason(result, "Semantic output rejected."))
                        break
                    contract = result.contract
                    if result.specification is not None:
                        if task["kind"] != "create" or result.specification.name != task["operator"]:
                            task = self._stop(task, "Specification targets another operator or a repair task.")
                            break
                        parse_operator_spec(result.specification.model_dump(), self.root)
                        if seed.specification is not None and result.specification != seed.specification:
                            task = self._stop(task, "The frozen specification cannot be rewritten.")
                            break
                        contract = contract_from_spec(result.specification)
                        seed = seed.model_copy(update={"specification": result.specification})
                if seed.contract is not None and contract != seed.contract:
                    task = self._stop(task, "The frozen contract cannot be rewritten; submit a revised task.")
                    break
                if not all((contract.formula, contract.inputs, contract.outputs, contract.dtypes,
                            contract.shapes, contract.boundaries, contract.tolerances)):
                    task = self._stop(task, "Contract incomplete; confirm formula, inputs/outputs, dtype, shape, boundaries and tolerance.")
                    break
                changed = {"contract_sha256": digest(contract.model_dump(mode="json"))}
            elif task["state"] == "planning_tests":
                candidate = result.test_candidate
                original = next((f for f in seed.files if f.path == candidate.path), None)
                if original and original.content != candidate.content:
                    task = self._stop(task, "Existing acceptance tests cannot be changed by a repair workflow.")
                    break
                files[candidate.path] = FileSnapshot(path=candidate.path, content=candidate.content, sha256=sha256(candidate.content))
                changed = {"tests_sha256": sha256(candidate.content)}
            elif task["state"] == "implementing":
                candidate = result.candidate
                if task["result"].get("rework_count") and sha256(candidate.content) == task["versions"]["source_sha256"]:
                    task = self._stop(task, "Rework repeated the same implementation. Inspect findings: " + " ".join(task["result"].get("review_feedback", [])))
                    break
                if task["kind"] == "repair" and any(f.path == candidate.path and f.sha256 == sha256(candidate.content) for f in seed.files):
                    task = self._stop(task, "Repair produced unchanged source. Inspect the failure and requirements before another attempt.")
                    break
                files[candidate.path] = FileSnapshot(path=candidate.path, content=candidate.content, sha256=sha256(candidate.content))
                changed = {"source_sha256": sha256(candidate.content)}
            else:
                if result.verdict != "acceptable":
                    if (result.verdict == "changes_requested" and result.findings
                            and task["result"].get("rework_count", 0) < task["max_repairs"]):
                        task = self.store.rework(task["id"], expected_revision=task["revision"],
                            event_id="rework-" + str(task["revision"]), findings=result.findings)
                        continue
                    task = self._stop(task, "Review did not accept the proposal: " + " ".join(result.findings or [result.summary]))
                    break
                changed = {}
            if snapshot_files(self.root, task["operator"]) != seed.files:
                task = self._stop(task, "Files changed during role work; the candidate needs a new plan.")
                break
            task = self.store.advance(task["id"], NEXT_STATE[task["state"]], {**task["versions"], **changed},
                expected_revision=task["revision"], event_id="advance-" + str(task["revision"]))
        return {"workflow_id": task["id"], "workflow_state": task["state"], "revision": task["revision"],
                "validation": "not_started", "files_applied": False, "reason": task["result"].get("reason",
                "Candidate proposal is ready for approval; no code was applied or tested."),
                "artifact_ids": [s["result"].get("artifact_id") for s in task["steps"]]}


def execute_workflow(jobs, job, owner, *, agent):
    result = ProposalCoordinator(jobs, job, owner, agent=agent).run()
    ArtifactStore(jobs.db.root).put(jobs, job["id"], owner, "result.json", encode(result).encode())
    status = "succeeded" if result["workflow_state"] == "waiting_approval" else "failed"
    jobs.finish(job["id"], owner, status, result)
    if result["workflow_state"] == "waiting_approval":
        try:
            from .lifecycle import WorkflowLifecycle
            lifecycle = WorkflowLifecycle(jobs.db.root)
            task = lifecycle.store.get(result["workflow_id"])
            if task["request"].get("auto_publish"):
                lifecycle.execute(task["id"], "publish", expected_revision=task["revision"])
        except Exception:
            # Publication failure must not replay a completed model job.
            import warnings
            warnings.warn("Proposal job saved; inspect workflow publication state before continuing.", RuntimeWarning)
