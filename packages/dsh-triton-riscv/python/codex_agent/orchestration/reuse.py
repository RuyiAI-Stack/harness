"""Host-only reuse of frozen preparation within a verified repair lineage."""
from importlib.metadata import PackageNotFoundError, version
import json

from codex_agent.execution_guard import artifact
from codex_agent.runtime_config import runtime_config
from codex_agent.tasks.store import Conflict, digest, encode
from codex_agent.tasks.schema import jobs
from codex_agent.validation_evidence import audit_validation_receipt, workspace_artifact_sha256
from . import prompts
from .evidence import load_role_audit, load_role_output
from .roles import operator_paths
from .store import WorkflowStore

POLICY_VERSION = "frozen-preparation-v1"


def preparation_scope(root, task, seed, contract, agent):
    settings = getattr(getattr(agent, "backend", None), "settings", None)
    if settings is None or not contract:
        return None
    identity = [getattr(settings, key, None) for key in ("provider", "model", "base_url")]
    if not all(isinstance(value, str) and value for value in identity):
        return None
    try:
        sdk = version("deepseek-harness-sdk")
    except PackageNotFoundError:
        return None
    script = root / "scripts/triton-riscv-env.sh"
    try:
        environment_script = workspace_artifact_sha256(root, "scripts/triton-riscv-env.sh") if script.exists() else None
        if script.is_symlink() or (script.exists() and environment_script is None):
            return None
    except (OSError, ValueError):
        return None
    return digest(dict(policy=POLICY_VERSION, prompt=prompts.PROMPT_VERSION, sdk=sdk, model=identity,
        workflow_policy=task["policy_version"], operator=task["operator"],
        requirement=task["request"]["requirement"], contract=contract.model_dump(mode="json"),
        evidence=[item.model_dump(mode="json") for item in seed.evidence],
        remote=runtime_config().remote.model_dump(mode="json"),
        environment_script=environment_script))


def _plan(db, task, seed, scope, *, connection=None):
    if not task["request"].get("reuse_preparation", True):
        return {"status": "not_reused", "reason": "disabled"}
    if task["kind"] != "repair" or scope is None or seed.contract is None:
        return {"status": "not_reused", "reason": "missing_frozen_context"}
    if not all((seed.contract.formula, seed.contract.inputs, seed.contract.outputs,
                seed.contract.dtypes, seed.contract.shapes, seed.contract.boundaries, seed.contract.tolerances)):
        return {"status": "not_reused", "reason": "incomplete_contract"}
    if (seed.failure_run_id is None or seed.failure_receipt_sha256 is None
            or seed.failure_context is None or not seed.failure_context.verified):
        return {"status": "not_reused", "reason": "missing_failure_evidence"}
    checked = audit_validation_receipt(db.root, {"run_id": seed.failure_run_id, "operator": task["operator"]})
    if (checked.verdict != "verified-failed"
            or workspace_artifact_sha256(db.root, checked.receipt_path) != seed.failure_receipt_sha256):
        return {"status": "not_reused", "reason": "failure_evidence_changed"}
    receipt = artifact(db.root, "validation", seed.failure_run_id)
    if receipt["source_snapshot"] != {f.path: f.sha256 for f in seed.files}:
        return {"status": "not_reused", "reason": "failure_source_changed"}
    store = WorkflowStore(db.root, database=db)
    child, seen = task, {task["id"]}
    tests_path = operator_paths(task["operator"])[1]
    frozen_test = next((f for f in seed.files if f.path == tests_path), None)
    if frozen_test is None:
        return {"status": "not_reused", "reason": "missing_fixed_tests"}
    # Only reciprocal ancestors qualify; a model-supplied ID is never consulted.
    for depth in range(20):
        parent_id = child["result"].get("parent_workflow_id")
        if not parent_id or parent_id in seen:
            return {"status": "not_reused", "reason": "missing_or_cyclic_lineage"}
        seen.add(parent_id)
        parent = (store._snapshot(connection, store._task(connection, parent_id, lock=True))
                  if connection is not None else store.get(parent_id))
        link = parent["result"].get("lifecycle", {})
        if (parent["state"] != "failed" or link.get("child_workflow_id") != child["id"]
                or parent["session_id"] != task["session_id"] or parent["operator"] != task["operator"]
                or parent["request"]["requirement"] != task["request"]["requirement"]
                or parent["policy_version"] != task["policy_version"]
                or parent["versions"]["contract_sha256"] != task["versions"]["contract_sha256"]
                or parent["versions"]["tests_sha256"] != frozen_test.sha256):
            return {"status": "not_reused", "reason": "lineage_or_contract_changed"}
        if depth == 0 and (link.get("validation_run_id") != seed.failure_run_id
                          or parent["versions"]["source_sha256"] != task["versions"]["source_sha256"]):
            return {"status": "not_reused", "reason": "different_failed_candidate"}
        candidates = [s for s in parent["steps"] if s["state"] == "planning_tests"]
        if not candidates:
            reused = parent["result"].get("preparation_reuse", {})
            if reused.get("status") != "reused" or reused.get("scope_sha256") != scope:
                return {"status": "not_reused", "reason": "missing_compatible_plan"}
            child = parent
            continue
        step = candidates[-1]
        audit, record = load_role_audit(db, parent["id"], step, connection=connection)
        result = load_role_output(db, parent["id"], step, connection=connection)
        job = db.rows(jobs, jobs.c.id == step["job_id"], connection=connection)
        if (audit.get("preparation_scope") != scope or audit.get("prompt_version") != prompts.PROMPT_VERSION
                or not job or job[0]["status"] != "succeeded"
                or json.loads(job[0]["payload_json"]).get("workflow_id") != parent["id"]
                or result is None or result.status != "completed" or result.unknowns
                or result.verdict != "tests_proposed" or result.test_candidate is None
                or result.test_candidate.path != tests_path or result.test_candidate.content != frozen_test.content):
            return {"status": "not_reused", "reason": "plan_changed_or_unverifiable"}
        return dict(status="reused", policy_version=POLICY_VERSION, scope_sha256=scope,
            parent_workflow_id=task["result"]["parent_workflow_id"], source_workflow_id=parent["id"],
            source_step_id=step["id"], source_artifact_id=record["id"], source_artifact_sha256=record["sha256"],
            contract_sha256=task["versions"]["contract_sha256"], tests_sha256=frozen_test.sha256,
            skipped_phases=["preparing", "planning_tests"], avoided_role_calls_minimum=2,
            reason="unchanged_frozen_preparation_in_same_repair_chain")
    return {"status": "not_reused", "reason": "lineage_limit"}


def reuse_preparation(store, task, seed, agent):
    if (task["state"] != "preparing" or task["steps"] or "preparation_reuse" in task["result"]
            or not task["result"].get("parent_workflow_id")):
        return task
    scope = preparation_scope(store.db.root, task, seed, seed.contract, agent)

    def change(conn, row):
        current = store._snapshot(conn, row)
        if current["state"] != "preparing" or current["steps"]:
            raise Conflict("preparation already started")
        try:
            decision = _plan(store.db, current, seed, scope, connection=conn)
        except (KeyError, ValueError, TypeError, OSError, Conflict):
            # Broken optional reuse evidence cannot authorize a shortcut.
            decision = {"status": "not_reused", "reason": "reuse_evidence_unavailable"}
        values = {"result_json": encode({**current["result"], "preparation_reuse": decision})}
        if decision["status"] == "reused":
            values.update(state="implementing", input_epoch=row["input_epoch"] + 1)
        return values, decision

    return store._mutate(task["id"], task["revision"], "preparation-reuse-" + str(task["revision"]),
                         "preparation_reuse_decided", {"scope_sha256": scope}, change)


def load_reused_tests(db, task, seed):
    """Recheck immutable preparation proof at publication, never reuse approval."""
    decision = task["result"].get("preparation_reuse", {})
    store = WorkflowStore(db.root, database=db)
    events = [e for e in store.history(task["id"], limit=10) if e["operation"] == "preparation_reuse_decided"]
    if (decision.get("status") != "reused" or decision.get("policy_version") != POLICY_VERSION
            or not events or events[0]["payload"] != decision
            or task["kind"] != "repair" or seed.contract is None
            or decision.get("contract_sha256") != digest(seed.contract.model_dump(mode="json"))
            or decision.get("tests_sha256") != task["versions"]["tests_sha256"]):
        raise Conflict("reused preparation checkpoint is missing or changed")
    child, seen = task, {task["id"]}
    for _ in range(20):
        parent_id = child["result"].get("parent_workflow_id")
        if not parent_id or parent_id in seen:
            break
        seen.add(parent_id)
        parent = store.get(parent_id)
        if (parent["result"].get("lifecycle", {}).get("child_workflow_id") != child["id"]
                or parent["state"] != "failed" or parent["session_id"] != task["session_id"]
                or parent["operator"] != task["operator"]
                or parent["request"]["requirement"] != task["request"]["requirement"]
                or parent["versions"]["contract_sha256"] != decision["contract_sha256"]
                or parent["versions"]["tests_sha256"] != decision["tests_sha256"]):
            break
        if parent["id"] == decision["source_workflow_id"]:
            steps = [s for s in parent["steps"] if s["id"] == decision["source_step_id"] and s["state"] == "planning_tests"]
            if len(steps) != 1:
                break
            audit, record = load_role_audit(db, parent["id"], steps[0])
            result = load_role_output(db, parent["id"], steps[0])
            job = db.rows(jobs, jobs.c.id == steps[0]["job_id"])
            test = next((f for f in seed.files if f.path == operator_paths(task["operator"])[1]), None)
            if (record["id"] == decision["source_artifact_id"] and record["sha256"] == decision["source_artifact_sha256"]
                    and audit.get("preparation_scope") == decision["scope_sha256"]
                    and audit.get("prompt_version") == prompts.PROMPT_VERSION
                    and job and job[0]["status"] == "succeeded"
                    and json.loads(job[0]["payload_json"]).get("workflow_id") == parent["id"]
                    and test and test.sha256 == decision["tests_sha256"]
                    and result is not None and result.status == "completed" and not result.unknowns
                    and result.verdict == "tests_proposed" and result.test_candidate is not None
                    and result.test_candidate.path == test.path and result.test_candidate.content == test.content):
                return result
            break
        child = parent
    raise Conflict("reused test plan no longer has matching lineage and artifact evidence")
