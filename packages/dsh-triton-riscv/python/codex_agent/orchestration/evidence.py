"""Read immutable role outputs with task, step, job and input-version bindings."""
import json

from codex_agent.artifacts import ArtifactStore
from codex_agent.tasks.schema import artifacts
from codex_agent.tasks.store import Conflict
from .roles import RoleRequest, parse_result


def load_role_audit(db, task_id, step, *, connection=None):
    rows = db.rows(artifacts, artifacts.c.id == step["result"].get("artifact_id"),
                   artifacts.c.job_id == step["job_id"], connection=connection)
    if not rows or rows[0]["name"] != step["id"] + ".json":
        raise Conflict("missing or foreign role artifact")
    value = json.loads(ArtifactStore(db.root).read(rows[0]))
    request = RoleRequest.model_validate(value["request"])
    if (request.task_id, request.step_id, request.input_epoch, request.role, request.phase) != (
            task_id, step["id"], step["input_epoch"], step["role"], step["state"]):
        raise Conflict("role artifact does not belong to this checkpoint")
    if request.versions.model_dump() != step["input"]["versions"]:
        raise Conflict("role artifact input versions do not match the checkpoint")
    return value, rows[0]


def load_role_output(db, task_id, step, *, connection=None):
    value, _ = load_role_audit(db, task_id, step, connection=connection)
    request = RoleRequest.model_validate(value["request"])
    if value["outcome"] != "validated" or step["status"] != "completed":
        return None
    return parse_result(request, value["raw"]["final_response"])
