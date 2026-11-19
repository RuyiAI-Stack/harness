"""Gateway-only management routes. Authentication is injected, not model supplied."""
import asyncio
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import case, func, select
from sqlalchemy.exc import SQLAlchemyError
from codex_agent.storage.database import StorageError

from codex_agent.accounts.schema import users, workspaces
from codex_agent.accounts.store import AccountError
from codex_agent.storage.schema import sessions, runs
from codex_agent.tasks.store import Conflict
from .store import DocumentInput, KnowledgeStore
from .vectors import search


class LifecycleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    catalog_revision: int = Field(default=0, ge=0)


class VersionInput(LifecycleInput):
    version: int = Field(ge=1)


class IndexInput(VersionInput):
    request_id: str = Field(min_length=1, max_length=100)


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    query: str = Field(min_length=1, max_length=6000)
    limit: int = Field(default=5, ge=1, le=10)


def usage_overview(accounts, offset=0):
    with accounts.engine.connect() as conn:
        sc = select(sessions.c.workspace_id, func.count().label("count"),
                    func.max(sessions.c.updated_at).label("last")).group_by(sessions.c.workspace_id).subquery()
        rc = select(runs.c.workspace_id, func.count().label("count"),
                    func.sum(case((runs.c.status == "failed", 1), else_=0)).label("failed"),
                    func.max(runs.c.updated_at).label("last")).group_by(runs.c.workspace_id).subquery()
        query = select(users.c.id, users.c.username, users.c.status, sc.c.count.label("sessions"),
            sc.c.last.label("session_last"), rc.c.count.label("runs"), rc.c.failed, rc.c.last.label("run_last"))
        query = query.select_from(users.outerjoin(workspaces, workspaces.c.user_id == users.c.id)
            .outerjoin(sc, sc.c.workspace_id == workspaces.c.workspace_id)
            .outerjoin(rc, rc.c.workspace_id == workspaces.c.workspace_id))
        rows = list(conn.execute(query.order_by(users.c.created_ms.desc(), users.c.id).offset(offset).limit(100)).mappings())
        totals = conn.execute(select(func.count().label("users"), func.coalesce(func.sum(sc.c.count), 0).label("sessions"),
            func.coalesce(func.sum(rc.c.count), 0).label("runs"), func.coalesce(func.sum(rc.c.failed), 0).label("failed_runs"))
            .select_from(users.outerjoin(workspaces, workspaces.c.user_id == users.c.id)
                         .outerjoin(sc, sc.c.workspace_id == workspaces.c.workspace_id)
                         .outerjoin(rc, rc.c.workspace_id == workspaces.c.workspace_id))).mappings().one()
    return {"users": [dict(id=row["id"], username=row["username"], status=row["status"],
            last_used_at=max(filter(None, [row["session_last"], row["run_last"]]), default=None),
            session_count=row["sessions"] or 0, run_count=row["runs"] or 0, failed_run_count=int(row["failed"] or 0))
            for row in rows], "totals": {k: int(v) for k, v in totals.items()}, "offset": offset}


def install_routes(app, config, accounts, user_for):
    root = Path(config.data_dir) / "knowledge"

    async def principal(request, *, admin=False, mutation=False):
        _, user = await user_for(request, mutation=mutation)
        if admin and user["id"] not in config.admin_user_ids:
            raise AccountError(403, "Administrator access required")
        await asyncio.to_thread(accounts.limit, "knowledge-admin" if admin else "knowledge-query", user["id"], 60, 60)
        return user

    def library():
        if not config.knowledge_enabled:
            raise AccountError(404, "Shared knowledge is disabled")
        return KnowledgeStore(root)

    async def body(request, schema):
        from codex_agent.accounts.api import limited_body
        try:
            return schema.model_validate_json(await limited_body(request, 8 * 1024 * 1024 if schema is DocumentInput else 300000))
        except ValueError:
            raise AccountError(422, "Invalid knowledge fields or document size") from None

    async def call(fn, *args, mutation=False):
        from .parsing import ParseError
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=35)
        except TimeoutError:
            raise AccountError(503, "Confirmation result is unknown; check the request receipt before retrying"
                if mutation else "Knowledge request timed out; reload current state before retrying") from None
        except KeyError:
            raise AccountError(404, "Knowledge document not found") from None
        except Conflict as error:
            raise AccountError(409, str(error)) from None
        except ParseError as error:
            raise AccountError(422, str(error)) from None
        except ValueError:
            raise AccountError(422, "Invalid document or embedding configuration") from None
        except (StorageError, SQLAlchemyError):
            raise AccountError(503, "Knowledge storage unavailable; check the operation receipt before retrying"
                if mutation else "Knowledge storage unavailable; retry later") from None

    @app.get("/api/admin/overview")
    async def overview(request: Request, offset: int = 0):
        user = await principal(request, admin=True)
        if not 0 <= offset <= 100000:
            raise AccountError(422, "Invalid page")
        def read():
            with accounts.engine.begin() as conn:
                accounts.log(conn, "admin-overview-read", user["id"])
            return usage_overview(accounts, offset)
        return await call(read)

    @app.get("/api/admin/knowledge")
    async def listing(request: Request, offset: int = 0, q: str = "", status: str = "all", view: str = "active"):
        user = await principal(request, admin=True)
        if not 0 <= offset <= 100000:
            raise AccountError(422, "Invalid page")
        if not config.knowledge_enabled:
            return {"documents": [], "total": 0, "enabled": False}
        store = library()
        return {**await call(lambda: store.catalog(offset=offset, query=q, status=status, view=view, actor=user["id"])), "enabled": True}

    @app.post("/api/admin/knowledge/confirm", status_code=202)
    async def confirm(request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, DocumentInput)
        return await call(lambda: library().confirm(value, user["id"], publisher_ids=config.admin_user_ids), mutation=True)

    @app.get("/api/admin/knowledge/operations/{request_id}")
    async def operation(request_id: str, request: Request):
        user = await principal(request, admin=True)
        return await call(library().operation, request_id, user["id"])

    @app.get("/api/admin/knowledge/{identifier}")
    async def detail(identifier: str, request: Request):
        user = await principal(request, admin=True)
        return await call(library().detail, identifier, user["id"])

    @app.post("/api/admin/knowledge", status_code=201)
    async def save(request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, DocumentInput)
        return await call(library().save, value, user["id"])

    @app.post("/api/admin/knowledge/files", status_code=202)
    async def upload_file(request: Request, filename: str, request_id: str, title: str = "", source: str = ""):
        from codex_agent.accounts.api import limited_body
        from .files import upload
        from .parsing import MAX_UPLOAD, ParseError
        user = await principal(request, admin=True, mutation=True)
        store = library()
        await asyncio.to_thread(accounts.limit, "knowledge-upload", user["id"], 10, 60)
        data = await limited_body(request, MAX_UPLOAD)
        try:
            return await call(upload, store, data, filename, title or Path(filename).stem, source,
                              request_id, user["id"])
        except ParseError as error:
            raise AccountError(422, str(error)) from None

    @app.get("/api/admin/knowledge/{identifier}/original")
    async def original(identifier: str, request: Request):
        from fastapi.responses import FileResponse
        from .files import file_path, source_row
        from .schema import documents
        await principal(request, admin=True)
        store = library()
        def read():
            with store.engine.connect() as conn:
                doc = store.row(conn, documents, identifier, allow_deleted=True)
                source = source_row(store, conn, identifier, doc["latest_version"])
                if not source:
                    raise KeyError("No original file")
                path = file_path(store, source["file_id"])
                if not path.is_file():
                    raise KeyError("Missing original file")
                return path, source["filename"]
        path, filename = await call(read)
        return FileResponse(path, filename=filename, media_type="application/octet-stream",
                            content_disposition_type="attachment", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/admin/knowledge/{identifier}/preview")
    async def image_preview(identifier: str, version: int, request: Request):
        from fastapi.responses import FileResponse
        from .files import preview_path, source_row
        from .schema import documents, versions
        await principal(request, admin=True)
        store = library()
        def read():
            if version < 1:
                raise ValueError("Invalid version")
            with store.engine.connect() as conn:
                store.row(conn, documents, identifier, allow_deleted=True)
                store.row(conn, versions, identifier, version)
                return preview_path(store, source_row(store, conn, identifier, version))
        path = await call(read)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.post("/api/admin/knowledge/{identifier}/index", status_code=202)
    async def index(identifier: str, request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, IndexInput)
        return await call(library().enqueue, identifier, value.version, value.request_id, user["id"], value.catalog_revision)

    @app.post("/api/admin/knowledge/{identifier}/publish")
    async def publish(identifier: str, request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, VersionInput)
        return await call(library().publish, identifier, value.version, user["id"], value.catalog_revision)

    @app.post("/api/admin/knowledge/{identifier}/unpublish")
    async def unpublish(identifier: str, request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, LifecycleInput)
        return await call(library().publish, identifier, None, user["id"], value.catalog_revision)

    @app.post("/api/admin/knowledge/{identifier}/delete")
    async def trash(identifier: str, request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, VersionInput)
        return await call(lambda: library().set_deleted(identifier, value.version, value.catalog_revision,
                                                       user["id"], deleted=True))

    @app.post("/api/admin/knowledge/{identifier}/restore")
    async def restore(identifier: str, request: Request):
        user = await principal(request, admin=True, mutation=True)
        value = await body(request, VersionInput)
        return await call(lambda: library().set_deleted(identifier, value.version, value.catalog_revision,
                                                       user["id"], deleted=False))

    @app.post("/api/knowledge/search")
    async def query(request: Request):
        await principal(request, mutation=True)
        value = await body(request, QueryInput)
        library()
        try:
            return await asyncio.to_thread(search, root, value.query, limit=value.limit)
        except Exception:
            return JSONResponse({"status": "unavailable", "items": [], "context": ""}, status_code=503)

    @app.get("/api/knowledge/documents/{identifier}")
    async def document(identifier: str, version: int, request: Request):
        await principal(request)
        return await call(library().published_document, identifier, version)
