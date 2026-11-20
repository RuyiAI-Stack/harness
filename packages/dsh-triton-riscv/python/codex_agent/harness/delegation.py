"""Host-approved, session-bound read-only replicas. Plans are not test receipts."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from codex_agent.accounts.guard import check_workspace
from codex_agent.execution_guard import atomic_json, digest, resource_locks

SCOPE = ContextVar("parallel_host_scope", default=None)
SCOPE_ENV = "TRITON_PARALLEL_SCOPE"
READ_TOOLS = ["discover_operator", "read_operator_file", "inspect_project", "retrieve_operator_memory"]
TERMINAL = {"completed", "partial", "cancelled", "rejected", "unknown"}


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    goal: str = Field(min_length=10, max_length=2000)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=10, max_length=2000)
    tasks: list[Subtask] = Field(min_length=2, max_length=4)
    context: str = Field(default="", max_length=12000)


@contextmanager
def host_scope(session_id, run_id, user_request):
    token = SCOPE.set({"session_id": session_id, "run_id": run_id, "user_request": user_request})
    try:
        yield
    finally:
        SCOPE.reset(token)


def trusted_scope():
    # This private file is supplied by the SDK host, never a model argument.
    path = os.environ.get(SCOPE_ENV)
    if not path:
        raise PermissionError("Parallel tasks require a host-bound session")
    scope = json.loads(Path(path).read_text())
    if not scope or not scope.get("session_id"):
        raise PermissionError("No active host turn")
    return scope


def base_settings(root, scope):
    from dataclasses import replace
    from codex_agent.harness.config import HarnessSettings
    settings = HarnessSettings.from_env(root)
    return replace(settings, model=scope.get("base_model") or settings.model)


class Plans:
    def __init__(self, root):
        self.root = Path(root).resolve()
        check_workspace(self.root)
        # Keep evidence inside the same account-owned workspace as lifecycle artifacts.
        self.folder = self.root / "agent-results/parallel-plans"

    def path(self, identifier):
        if not re.fullmatch(r"parallel-[a-f0-9]{32}", identifier):
            raise ValueError("Invalid parallel plan ID")
        return self.folder / (identifier + ".json")

    def load(self, identifier, session_id):
        check_workspace(self.root)
        row = json.loads(self.path(identifier).read_text())
        if row["session_id"] != session_id or row["workspace"] != str(self.root):
            raise PermissionError("Parallel plan belongs to another session or workspace")
        if digest(row["spec"]) != row["digest"]:
            raise PermissionError("Parallel plan changed; create a new plan")
        return row

    @staticmethod
    def public(row):
        return {**row["spec"], "cancel_requested": row.get("cancel_requested", False), **{key: row[key] for key in (
            "plan_id", "session_id", "status", "digest", "results", "created_at")}}

    @staticmethod
    def stale_reason(row, settings):
        if row["status"] not in {"pending", "approved"}:
            return None
        approved_at = row.get("approval", {}).get("at", row["created_at"])
        since = approved_at if row["status"] == "approved" else row["created_at"]
        if time.time() - since > 3600:
            return "approval_expired"
        if any(row["spec"][key] != getattr(settings, key) for key in ("model", "provider", "base_url")):
            return "model_configuration_changed"
        from codex_agent.harness.model_routing import replica_route
        if any("route" in task and task["route"] != replica_route(settings, row["spec"], task)
               for task in row["spec"]["tasks"]):
            return "routing_policy_changed"
        return None

    def prepare(self, scope, request, settings):
        value = PlanRequest.model_validate(request)
        if len({task.goal.strip() for task in value.tasks}) != len(value.tasks):
            raise ValueError("Subtasks must have distinct goals")
        # Config validation happens before approval; no API request is sent here.
        settings.validate_live_run()
        spec = {"reason": value.reason, "context": value.context,
                "user_request": scope["user_request"][:12000],
                "tasks": [{"task_id": f"child-{i + 1}", "goal": task.goal} for i, task in enumerate(value.tasks)],
                "provider": settings.provider, "model": settings.model, "base_url": settings.base_url,
                "max_parallel": 2, "timeout_seconds": 300, "max_tool_calls": 8,
                "tools": READ_TOOLS}
        from codex_agent.harness.model_routing import replica_route
        for task in spec["tasks"]:
            task["route"] = replica_route(settings, spec, task)
        spec["timeout_seconds"] = max(task["route"]["timeout_seconds"] for task in spec["tasks"])
        with resource_locks(self.root, ["parallel-session:" + scope["session_id"]]):
            for row in self.pending(scope["session_id"], statuses={"pending", "approved", "running"}):
                # Execution can race preparation. Recheck under its plan lock before
                # retiring an unstarted plan; never replace active or cancelling work.
                with resource_locks(self.root, [row["plan_id"]], wait_seconds=5):
                    row = self.load(row["plan_id"], scope["session_id"])
                    if row["status"] in TERMINAL:
                        continue
                    stale = self.stale_reason(row, settings)
                    if stale:
                        row.update(status="cancelled", retired_reason=stale, retired_at=time.time())
                        atomic_json(self.path(row["plan_id"]), row)
                        continue
                    if row["digest"] == digest(spec):
                        return self.public(row)
                    raise PermissionError("An undecided parallel plan already exists; reject it before replanning")
            row = {"plan_id": "parallel-" + uuid4().hex, "session_id": scope["session_id"],
                   "run_id": scope["run_id"], "workspace": str(self.root), "spec": spec,
                   "digest": digest(spec), "created_at": time.time(), "status": "pending", "results": []}
            atomic_json(self.path(row["plan_id"]), row)
        return self.public(row)

    def pending(self, session_id, statuses=None):
        if not self.folder.exists():
            return []
        rows = []
        for path in self.folder.glob("parallel-*.json"):
            row = json.loads(path.read_text())
            if row.get("session_id") == session_id and row.get("status") in (statuses or {"pending"}):
                rows.append(self.load(row["plan_id"], session_id))
        return rows

    def decide(self, identifier, session_id, fingerprint, approved, reviewer):
        if type(approved) is not bool:
            raise ValueError("approved must be a boolean")
        with resource_locks(self.root, [identifier], wait_seconds=5):
            row = self.load(identifier, session_id)
            if row["digest"] != fingerprint:
                raise PermissionError("Plan changed; review the current plan")
            target = "approved" if approved else "rejected"
            if row["status"] == target:
                return self.public(row)
            allowed = {"pending"} if approved else {"pending", "approved"}
            if row["status"] not in allowed or (approved and time.time() - row["created_at"] > 3600):
                raise PermissionError("Plan is no longer awaiting approval; create a new plan")
            row.update(status=target, approval={"reviewer": reviewer, "digest": fingerprint, "at": time.time()})
            atomic_json(self.path(identifier), row)
            return self.public(row)

    def cancel(self, identifier, session_id):
        with resource_locks(self.root, [identifier], wait_seconds=5):
            row = self.load(identifier, session_id)
            if row["status"] not in TERMINAL:
                row["cancel_requested"] = True
                if row["status"] != "running":
                    row["status"] = "cancelled"
                atomic_json(self.path(identifier), row)
            return self.public(row)

    def execute(self, identifier, session_id, settings, runner=None, *, stop=None):
        from dsh_multi_agent.parallel import run_parallel
        from codex_agent.harness.replica import run_replica
        with resource_locks(self.root, ["parallel-execute:" + session_id]):
            with resource_locks(self.root, [identifier], wait_seconds=5):
                row = self.load(identifier, session_id)
                if row["status"] in TERMINAL:
                    return self.public(row)
                if row["status"] == "running":
                    # The execution lock is free: an earlier process died. Never replay unknown paid work.
                    row["status"] = "unknown"
                    atomic_json(self.path(identifier), row)
                    return self.public(row)
                if row["status"] != "approved" or row.get("approval", {}).get("digest") != row["digest"]:
                    raise PermissionError("Human approval is required before replicas can start")
                if time.time() - row["approval"]["at"] > 3600:
                    raise PermissionError("Approval expired; create a new plan")
                for key in ("model", "provider", "base_url"):
                    if row["spec"][key] != getattr(settings, key):
                        raise PermissionError("Model configuration changed; create a new plan")
                from codex_agent.harness.model_routing import replica_route
                for task in row["spec"]["tasks"]:
                    if "route" in task and task["route"] != replica_route(settings, row["spec"], task):
                        raise PermissionError("Model routing policy changed; create a new plan for approval")
                settings.validate_live_run()
                row["status"] = "running"
                atomic_json(self.path(identifier), row)

            def cancelled():
                if stop is not None and stop():
                    self.cancel(identifier, session_id)
                    return True
                current = self.load(identifier, session_id)
                return current.get("cancel_requested", False) or current["status"] == "cancelled"

            def save(result):
                with resource_locks(self.root, [identifier], wait_seconds=5):
                    current = self.load(identifier, session_id)
                    current["results"].append(result)
                    atomic_json(self.path(identifier), current)

            execute = runner or (lambda task, stop: run_replica(settings, row["spec"], task, stop))
            run_parallel(row["spec"]["tasks"], execute, cancelled=cancelled, on_result=save)
            with resource_locks(self.root, [identifier], wait_seconds=5):
                current = self.load(identifier, session_id)
                if current.get("cancel_requested", False):
                    current["status"] = "cancelled"
                elif current["status"] != "cancelled":
                    current["status"] = "completed" if all(r["status"] == "completed" for r in current["results"]) else "partial"
                atomic_json(self.path(identifier), current)
                return self.public(current)
