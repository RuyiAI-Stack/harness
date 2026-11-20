"""Human-only parallel approval endpoints, behind the existing account gateway."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from codex_agent.accounts.guard import reviewer_identity
from codex_agent.harness.delegation import Plans


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    approved: bool
    digest: str


def router(root, store):
    api = APIRouter(prefix="/api/sessions/{session_id}/parallel-plans")

    def operate(session_id, action):
        try:
            store.get_session(session_id)
            return action(Plans(root))
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Parallel plan or session not found") from None
        except (PermissionError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @api.get("/{plan_id}")
    def detail(session_id: str, plan_id: str):
        return operate(session_id, lambda plans: plans.public(plans.load(plan_id, session_id)))

    @api.post("/{plan_id}/decision")
    def decide(session_id: str, plan_id: str, request: Decision):
        return operate(session_id, lambda plans: plans.decide(plan_id, session_id, request.digest,
                       request.approved, reviewer_identity("workbench-user")))

    @api.post("/{plan_id}/cancel")
    def cancel(session_id: str, plan_id: str):
        return operate(session_id, lambda plans: plans.cancel(plan_id, session_id))

    return api
