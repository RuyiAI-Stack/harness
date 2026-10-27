"""Explicit proposal-workflow submission; normal chat routing stays unchanged."""
import json
from pathlib import Path
import uuid
from contextlib import nullcontext

from pydantic import Field

from codex_agent.operator_development import NewOperatorSpec
from codex_agent.develop_operator import parse_operator_spec
from codex_agent.platform.store import timestamp
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.schema import runs
from codex_agent.tasks.service import request_key
from codex_agent.tasks.store import Conflict, JobStore, digest, encode
from .models import Contract, InputVersions, TaskRequest, Sha256
from .roles import Evidence, FileSnapshot, OperatorContract, operator_paths, sha256
from .store import WorkflowStore, bounded
from .schema import tasks, events


class WorkflowSeed(Contract):
    files: list[FileSnapshot] = Field(default_factory=list, max_length=2)
    evidence: list[Evidence] = Field(default_factory=list, max_length=24)
    contract: OperatorContract | None = None
    specification: NewOperatorSpec | None = None
    failure_run_id: str | None = None
    failure_receipt_sha256: Sha256 | None = None
    failure_context: Evidence | None = None
    exclude_source_runs: list[str] = Field(default_factory=list, max_length=24)


def contract_from_spec(spec):
    return OperatorContract(formula=spec.semantics,
        inputs=[item.name + ": " + item.description for item in spec.inputs], outputs=[spec.output],
        dtypes=spec.dtypes, shapes=[encode(shape) for shape in spec.shape_cases],
        boundaries=["Use the exact supplied shape cases and input_shape_cases."],
        tolerances=encode(spec.tolerances.model_dump()))


def snapshot_files(root, operator):
    root = Path(root).resolve()
    result = []
    for relative in operator_paths(operator):
        path = root / relative
        path.resolve().relative_to(root)
        if path.is_symlink():
            raise ValueError("operator files must not be symlinks")
        if not path.exists():
            continue
        with path.open("rb") as stream:
            raw = stream.read(192001)
        if len(raw) > 192000:
            raise ValueError("operator file exceeds snapshot budget")
        content = raw.decode("utf-8")
        result.append(FileSnapshot(path=relative, content=content, sha256=sha256(content)))
    return result


class WorkflowService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.jobs = JobStore(self.root)
        self.store = WorkflowStore(self.root, database=self.jobs.db)

    def submit(self, request, key, *, evidence=(), contract=None, specification=None, failure_run_id=None,
               connection=None, exclude_source_runs=()):
        if not runtime_config().queue.enabled or not runtime_config().queue.proposalWorkflows:
            raise PermissionError("proposal workflow submission is disabled")
        request = TaskRequest.model_validate(request)
        evidence, exclude_source_runs = list(evidence), list(exclude_source_runs)
        # Compare a replay to the frozen submission, not files the task has since created.
        # Hold the same workspace lock through fresh submission to avoid a check/create race.
        with (self.jobs.db.transaction() if connection is None else nullcontext(connection)) as conn:
            self.jobs.db.lock(conn, "task-submit")
            old = self.jobs.find_key(conn, request_key("operator-workflow", key))
            if old:
                payload = json.loads(old["payload_json"])
                records = self.jobs.db.rows(events, events.c.task_id == payload.get("workflow_id"),
                    events.c.revision == 1, connection=conn)
                if payload.get("mode") != "operator-workflow" or not records:
                    raise Conflict("workflow submission key belongs to another operation")
                created = json.loads(records[0]["payload_json"])
                if payload.get("fingerprint") != digest([created["request"], created["context"]]):
                    raise Conflict("stored workflow submission is inconsistent")
                spec = NewOperatorSpec.model_validate(specification) if specification is not None else None
                proposed_contract = OperatorContract.model_validate(contract) if contract is not None else None
                if spec is not None:
                    derived = contract_from_spec(spec)
                    if proposed_contract is not None and proposed_contract != derived:
                        raise Conflict("contract differs from the structured development specification")
                    proposed_contract = derived
                excluded = list(exclude_source_runs)
                if failure_run_id:
                    excluded.append(f"operator-lifecycle:{failure_run_id}")
                supplied = WorkflowSeed.model_validate(dict(evidence=evidence, contract=proposed_contract,
                    specification=spec, failure_run_id=failure_run_id,
                    exclude_source_runs=list(dict.fromkeys(excluded)))).model_dump(mode="json")
                incoming = request.model_dump(mode="json")
                original = dict(created["request"])
                versions, observed = incoming.pop("versions"), original.pop("versions")
                fields = ("evidence", "contract", "specification", "failure_run_id", "exclude_source_runs")
                if (incoming != original or any(value is not None and value != observed[name]
                        for name, value in versions.items()) or any(supplied[name] != created["context"].get(name,
                        [] if name == "exclude_source_runs" else None) for name in fields)):
                    raise Conflict("workflow submission key reused with different inputs")
                return self.jobs.public(old)
            return self._submit(request, key, evidence=evidence, contract=contract, specification=specification,
                failure_run_id=failure_run_id, connection=conn, exclude_source_runs=exclude_source_runs)

    def _submit(self, request, key, *, evidence=(), contract=None, specification=None, failure_run_id=None,
                connection=None, exclude_source_runs=()):
        if not runtime_config().queue.enabled or not runtime_config().queue.proposalWorkflows:
            raise PermissionError("proposal workflow submission is disabled")
        request = TaskRequest.model_validate(request)
        if len(request.requirement) > 8000:
            raise ValueError("role requirement exceeds 8000 characters")
        if specification is not None:
            specification = NewOperatorSpec.model_validate(specification)
            parse_operator_spec(specification.model_dump(), self.root)
            if request.kind != "create" or specification.name != request.operator:
                raise Conflict("development specification belongs to a different task")
            derived = contract_from_spec(specification)
            if contract is not None and OperatorContract.model_validate(contract) != derived:
                raise Conflict("contract differs from the structured development specification")
            contract = derived
        receipt_hash, failure_context = None, None
        if failure_run_id is not None:
            from codex_agent.execution_guard import artifact
            from codex_agent.validation_evidence import audit_validation_receipt, workspace_artifact_sha256
            checked = audit_validation_receipt(self.root, {"run_id": failure_run_id, "operator": request.operator})
            if request.kind != "repair" or checked.verdict != "verified-failed":
                raise Conflict("repair must bind an audited failure for this operator and source version")
            receipt = artifact(self.root, "validation", failure_run_id)
            if set(receipt["source_snapshot"]) != set(operator_paths(request.operator)):
                raise Conflict("repair receipt must use exactly this operator and its acceptance test file")
            receipt_hash = workspace_artifact_sha256(self.root, checked.receipt_path)
            failure_context = Evidence(id="workflow_failure", kind="log", source=checked.receipt_path,
                text=encode({k: receipt.get(k) for k in ("run_id", "command", "failure_stage", "error_excerpt")}),
                verified=True)
        excluded = list(exclude_source_runs)
        if failure_run_id:
            excluded.append(f"operator-lifecycle:{failure_run_id}")
        seed = WorkflowSeed.model_validate({"files": snapshot_files(self.root, request.operator),
            "evidence": list(evidence), "contract": contract, "specification": specification,
            "failure_run_id": failure_run_id, "failure_receipt_sha256": receipt_hash,
            "failure_context": failure_context,
            "exclude_source_runs": list(dict.fromkeys(excluded))})
        if len({e.id for e in seed.evidence}) != len(seed.evidence) or any(
                e.id.startswith("workflow_") for e in seed.evidence):
            raise ValueError("evidence IDs must be unique and must not use workflow_ prefix")
        if request.kind == "create" and seed.files:
            raise Conflict("create workflow cannot overwrite an existing operator or test")
        if request.kind == "repair" and len(seed.files) != 2:
            raise ValueError("repair requires existing implementation and acceptance tests")
        by_path = {f.path: f.sha256 for f in seed.files}
        if failure_run_id and by_path != receipt["source_snapshot"]:
            raise Conflict("source changed while freezing the failure receipt")
        source, tests = operator_paths(request.operator)
        observed = InputVersions(contract_sha256=digest(seed.contract.model_dump(mode="json")) if seed.contract else None,
            source_sha256=by_path.get(source), tests_sha256=by_path.get(tests))
        for field, supplied in request.versions.model_dump().items():
            if supplied is not None and supplied != getattr(observed, field):
                raise Conflict("supplied version does not match observed inputs")
        request = request.model_copy(update={"versions": observed})
        context = seed.model_dump(mode="json")
        fingerprint = digest([request.model_dump(mode="json"), context])
        bounded(context)
        key = request_key("operator-workflow", key)
        db = self.jobs.db
        with (db.transaction() if connection is None else nullcontext(connection)) as conn:
            db.lock(conn, "task-submit")
            old = self.jobs.find_key(conn, key)
            if old:
                payload = json.loads(old["payload_json"])
                if payload.get("fingerprint") != fingerprint or payload.get("mode") != "operator-workflow":
                    raise Conflict("workflow submission key reused with different inputs")
                return self.jobs.public(old)
            task = self.store.create(request, request_id=key, context=context, connection=conn)
            run_id, now = uuid.uuid4().hex, timestamp()
            db.insert(conn, runs, dict(id=run_id, session_id=request.session_id,
                intent="operator-workflow", operator=request.operator, status="queued", phase="outbox-pending",
                request_json=encode({"action": "operator-workflow", "workflow_id": task["id"]}),
                result_json="{}", created_at=now, updated_at=now, next_sequence=0))
            job = self.jobs.submit("agent", {"mode": "operator-workflow", "workflow_id": task["id"],
                "run_id": run_id, "session_id": request.session_id, "fingerprint": fingerprint},
                key, run_id=run_id, connection=conn)
            db.update(conn, tasks, {"result_json": encode({"agent_job_id": job["id"]})}, tasks.c.id == task["id"])
            return job
