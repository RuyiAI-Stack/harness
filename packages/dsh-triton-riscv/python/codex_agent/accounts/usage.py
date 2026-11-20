"""Read-only personal usage; the authenticated account is the sole scope selector."""
import asyncio
from datetime import datetime, timezone

from fastapi import Request
from sqlalchemy import case, func, select
from sqlalchemy.exc import SQLAlchemyError

from codex_agent.storage.database import StorageError
from codex_agent.storage.schema import runs, sessions
from .schema import users, workspaces
from .store import AccountError


def personal_usage(accounts, user_id):
    """Query the registered workspace only; never create or enumerate workspaces."""
    with accounts.engine.connect() as conn:
        account = conn.execute(select(users.c.id, users.c.username, users.c.status)
            .where(users.c.id == user_id, users.c.status == "active")).mappings().first()
        if account is None:
            raise AccountError(401, "Account unavailable; please sign in")
        workspace = conn.execute(select(workspaces.c.workspace_id, workspaces.c.status)
            .where(workspaces.c.user_id == user_id)).mappings().first()
        if workspace and workspace["status"] != "ready":
            raise AccountError(409, "Usage is unavailable while the workspace requires reconciliation")

        totals = {"sessions": 0, "runs": 0, "failed_runs": 0}
        last_used_at, recent = None, []
        if workspace:
            scope = workspace["workspace_id"]
            sc = conn.execute(select(func.count().label("count"), func.max(sessions.c.updated_at).label("last"))
                .where(sessions.c.workspace_id == scope)).mappings().one()
            rc = conn.execute(select(func.count().label("count"),
                func.coalesce(func.sum(case((runs.c.status == "failed", 1), else_=0)), 0).label("failed"),
                func.max(runs.c.updated_at).label("last"))
                .where(runs.c.workspace_id == scope)).mappings().one()
            totals = {"sessions": int(sc["count"]), "runs": int(rc["count"]), "failed_runs": int(rc["failed"])}
            last_used_at = max(filter(None, [sc["last"], rc["last"]]), default=None)
            # Only lifecycle metadata leaves this endpoint. Intent, request_json,
            # result_json, operator, session text and workspace paths stay private.
            recent = [dict(row) for row in conn.execute(select(runs.c.id, runs.c.status,
                runs.c.created_at, runs.c.updated_at).where(runs.c.workspace_id == scope)
                .order_by(runs.c.updated_at.desc(), runs.c.id.desc()).limit(10)).mappings()]
        return {"account": dict(account), "totals": totals, "last_used_at": last_used_at,
                "recent_runs": recent, "as_of": datetime.now(timezone.utc).isoformat()}


def install_routes(app, accounts, user_for):
    @app.get("/api/account/usage")
    async def usage(request: Request):
        _, user = await user_for(request)
        # There is deliberately no admin/user_id override. Reject selectors rather
        # than silently implying a caller can choose whose statistics are returned.
        if request.query_params:
            raise AccountError(422, "This endpoint only accepts the current authenticated account")
        await asyncio.to_thread(accounts.limit, "account-usage", user["id"], 60, 60)
        try:
            result = await asyncio.wait_for(asyncio.to_thread(personal_usage, accounts, user["id"]), timeout=10)
        except TimeoutError:
            raise AccountError(503, "Usage statistics timed out; please retry later") from None
        except (StorageError, SQLAlchemyError):
            raise AccountError(503, "Usage statistics are temporarily unavailable") from None
        # Re-check revocation after the potentially slow read before releasing data.
        _, current = await user_for(request, touch=False)
        if current["id"] != user["id"]:
            raise AccountError(401, "Account changed; reload before continuing")
        return result
