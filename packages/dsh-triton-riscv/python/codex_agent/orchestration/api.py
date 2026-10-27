"""Opt-in host HTTP controls. No approval or lifecycle controls are MCP tools."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from codex_agent.operator_development import NewOperatorSpec
from codex_agent.tasks.store import Conflict
from codex_agent.runtime_config import runtime_config
from .lifecycle import WorkflowLifecycle
from .models import Contract, TaskRequest
from .roles import Evidence, OperatorContract
from .service import WorkflowService
from .usage_report import workflow_usage


class Submission(Contract):
    request: TaskRequest
    request_id: str = Field(min_length=1, max_length=160)
    contract: OperatorContract | None = None
    evidence: list[Evidence] = Field(default_factory=list, max_length=24)
    specification: NewOperatorSpec | None = None
    failure_run_id: str | None = None


class Revision(Contract):
    expected_revision: int = Field(ge=1)


def router(root):
    routes = APIRouter(prefix="/api/operator-workflows")
    def call(function):
        try:
            return function()
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except KeyError as error:
            raise HTTPException(404, "Unknown workflow or artifact") from error
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(409, "Operation incomplete; inspect workflow state and domain artifacts before retrying") from error

    @routes.get("/config")
    def config():
        queue = runtime_config().queue
        return {"enabled": queue.enabled and queue.proposalWorkflows, "default_max_repairs": 3}

    @routes.get("")
    def list_workflows(session_id: str = Query(min_length=1, max_length=64)):
        if not runtime_config().queue.enabled or not runtime_config().queue.proposalWorkflows:
            return []
        return call(lambda: WorkflowLifecycle(root).store.for_session(session_id))

    @routes.post("", status_code=202)
    def submit(body: Submission):
        def action():
            if body.request.kind == "create" and body.specification is None and not body.request.auto_publish:
                raise ValueError("create requires a structured specification or the explicit interactive flow")
            if body.request.kind == "repair" and body.failure_run_id is None:
                raise ValueError("repair requires an audited failure_run_id")
            return WorkflowService(root).submit(body.request, body.request_id, contract=body.contract,
                evidence=body.evidence, specification=body.specification, failure_run_id=body.failure_run_id)
        return call(action)

    @routes.get("/{workflow_id}")
    def get(workflow_id: str):
        return call(lambda: WorkflowLifecycle(root).view(workflow_id))

    @routes.get("/{workflow_id}/usage")
    def usage(workflow_id: str):
        return call(lambda: workflow_usage(WorkflowLifecycle(root).jobs.db, workflow_id))

    @routes.post("/{workflow_id}/{action}")
    def act(workflow_id: str, action: Literal["publish", "apply", "plan-validation", "enqueue-validation", "collect-validation"],
            body: Revision):
        return call(lambda: WorkflowLifecycle(root).execute(workflow_id, action, expected_revision=body.expected_revision))

    return routes
