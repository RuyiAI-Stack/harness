"""Read-only, workspace-bound cost evidence for a workflow and its repair chain."""
import json
import math

from codex_agent.artifacts import ArtifactStore
from codex_agent.tasks.schema import artifacts
from codex_agent.tasks.store import Conflict
from .store import WorkflowStore
from .usage import count, summarize


def workflow_calls(db, task):
    steps = task["steps"]
    jobs = {step["job_id"] for step in steps if step.get("job_id")}
    names = {step["id"] + ".json" for step in steps}
    records = db.rows(artifacts, artifacts.c.job_id.in_(jobs), artifacts.c.name.in_(names)) if jobs else []
    output = []
    for step in steps:
        matches = [r for r in records if r["job_id"] == step["job_id"] and r["name"] == step["id"] + ".json"]
        for record in matches or [None]:
            usage, error = None, None
            try:
                if record is None:
                    error = "usage_not_recorded"
                else:
                    audit = json.loads(ArtifactStore(db.root).read(record))
                    request = audit["request"]
                    if ((request["task_id"], request["step_id"], request["input_epoch"], request["role"], request["phase"])
                            != (task["id"], step["id"], step["input_epoch"], step["role"], step["state"])
                            or request["versions"] != step["input"]["versions"]):
                        raise ValueError("foreign role artifact")
                    usage = audit.get("usage")
                    if not isinstance(usage, dict) or usage.get("schema_version") != 1:
                        usage, error = None, "usage_not_recorded"
                    elif (not isinstance(usage.get("totals"), dict)
                          or not isinstance(usage.get("attempts"), list)
                          or not count(usage.get("attempt_count"))
                          or usage["attempt_count"] != len(usage["attempts"])
                          or not all(isinstance(a, dict) and type(a.get("closed")) is bool for a in usage["attempts"])
                          or type(usage.get("elapsed_ms")) not in (int, float)
                          or not math.isfinite(usage["elapsed_ms"]) or usage["elapsed_ms"] < 0
                          or not isinstance(usage.get("outcome"), str)):
                        usage, error = None, "malformed_usage_record"
            except (KeyError, TypeError, ValueError, OSError):
                usage, error = None, "artifact_unavailable_or_unbound"
            output.append({"workflow_id": task["id"], "step_id": step["id"], "role": step["role"],
                           "phase": step["state"], "step_status": step["status"],
                           "job_id": step["job_id"], "job_attempt": record["attempt"] if record else None,
                           "input_epoch": step["input_epoch"], "artifact_id": record["id"] if record else None,
                           "usage": usage, "error": error})
    return output


def aggregate_calls(calls):
    usages = [call["usage"] for call in calls]
    rows = [value["totals"] if value else {} for value in usages]
    result = summarize(rows)
    result.update(role_invocations=len(calls),
                  recorded_invocations=sum(value is not None for value in usages),
                  observed_model_attempts=sum(value.get("attempt_count", 0) for value in usages if value),
                  model_attempt_count_complete=bool(usages) and all(value and value.get("attempt_count", 0) > 0
                      and not value.get("issues") and all(a["closed"] for a in value["attempts"]) for value in usages),
                  role_elapsed_ms=sum(value["elapsed_ms"] for value in usages) if usages and all(usages) else None,
                  known_role_elapsed_ms=sum(value["elapsed_ms"] for value in usages if value),
                  unsuccessful_invocations=sum(value["outcome"] != "validated" or value.get("role_status") != "completed"
                                               for value in usages if value),
                  cost=None, cost_status="pricing_not_configured")
    return result


def workflow_usage(db, workflow_id):
    store = WorkflowStore(db.root, database=db)
    task = store.get(workflow_id)
    seen = {task["id"]}
    # Follow only reciprocal links in the same workspace, session and operator.
    while task["result"].get("parent_workflow_id"):
        parent = store.get(task["result"]["parent_workflow_id"])
        _check_link(parent, task)
        if parent["id"] in seen or len(seen) >= 21:
            raise Conflict("invalid repair chain")
        seen.add(parent["id"])
        task = parent
    tasks, seen = [task], {task["id"]}
    while task["result"].get("lifecycle", {}).get("child_workflow_id"):
        child = store.get(task["result"]["lifecycle"]["child_workflow_id"])
        _check_link(task, child)
        if child["id"] in seen or len(seen) >= 21:
            raise Conflict("invalid repair chain")
        seen.add(child["id"])
        tasks.append(child)
        task = child
    calls = [call for task in tasks for call in workflow_calls(db, task)]
    return {"schema_version": 1, "requested_workflow_id": workflow_id, "root_workflow_id": tasks[0]["id"],
            "scope": "workflow_and_linked_repairs", "totals": aggregate_calls(calls),
            "by_role": {role: aggregate_calls([c for c in calls if c["role"] == role])
                        for role in sorted({c["role"] for c in calls})},
            "workflows": [{"id": task["id"], "state": task["state"], "created_ms": task["created_ms"],
                           "updated_ms": task["updated_ms"],
                           "preparation_reuse": task["result"].get("preparation_reuse"),
                           "validation_plan_id": task["result"].get("lifecycle", {}).get("validation_plan_id")}
                          for task in tasks], "calls": calls,
            "notes": ["SDK-reported usage, not a provider bill; absent counters are unknown, not zero.",
                      "Role elapsed time includes SDK startup and waiting, not just model inference.",
                      "Counts cover role calls only; embedding, ordinary chat, queue and validation time are excluded.",
                      "A validated role proposal is not proof that operator tests passed.",
                      "Reused preparation is a host checkpoint, not a model call or measured token saving."]}


def _check_link(parent, child):
    if (parent["session_id"] != child["session_id"] or parent["operator"] != child["operator"]
            or child["result"].get("parent_workflow_id") != parent["id"]
            or parent["result"].get("lifecycle", {}).get("child_workflow_id") != child["id"]):
        raise Conflict("repair chain identity mismatch")
