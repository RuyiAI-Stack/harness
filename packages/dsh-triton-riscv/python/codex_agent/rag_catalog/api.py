"""Authenticated, version-aware RAG source administration. No model calls."""
import asyncio
import secrets
from typing import Literal

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError

from codex_agent.accounts.api import limited_body
from codex_agent.accounts.store import AccountError
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.database import StorageError
from codex_agent.tasks.store import Conflict, digest
from .store import CatalogStore
from .scopes import authorized_scopes
from . import actions
from . import sharing
from .sources import configured_sources, open_history, source_signing_key, HistoryUnavailable, HistoryForbidden


class ActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    action: Literal["archive", "correct"]
    expected_revision: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(default="管理员停用", max_length=1000)
    content: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def valid_action(self):
        if "\0" in self.request_id or "\0" in self.reason or "\0" in self.content:
            raise ValueError("Invalid content")
        if self.action == "correct" and not self.content.strip():
            raise ValueError("Correction content is required")
        if self.action == "archive" and (not self.reason.strip() or self.content):
            raise ValueError("Archive requires only a reason")
        return self


def install_routes(app, config, accounts, user_for):
    # Signing grants no data access: every subsequent request resolves and checks
    # its current authorized scope again. A restart deliberately invalidates cursors.
    cursor_key = secrets.token_bytes(32)

    async def principal(request, mutation=False):
        _, user = await user_for(request, mutation=mutation)
        if user["id"] not in config.admin_user_ids:
            raise AccountError(403, "Administrator access required")
        if not config.rag_catalog_enabled:
            raise AccountError(404, "RAG catalog is disabled")
        await asyncio.to_thread(accounts.limit, "rag-admin", user["id"], 120, 60)
        return user

    def store(user, source_id="current"):
        if source_id != "current":
            source, engine, scopes, key = open_history(config, source_id, cursor_key)
            return CatalogStore(scopes, cursor_key=key, database_engine=engine,
                memory_mode=source.memory_mode, knowledge_fingerprint=None)
        cfg = runtime_config()
        fingerprint = None
        if cfg.indexing.enabled and config.knowledge_enabled:
            try:
                from codex_agent.knowledge.vectors import index_definition
                fingerprint = digest(index_definition())
            except (ValueError, OSError):
                pass  # Read original documents even if indexing is unavailable.
        memory_mode = cfg.memory.retrievalMode
        if memory_mode == "qdrant" and not cfg.indexing.enabled:
            memory_mode = "unknown"
        return CatalogStore(authorized_scopes(config, accounts, user), cursor_key=source_signing_key(cursor_key, "current"),
            database_engine=accounts.engine, memory_mode=memory_mode,
            knowledge_fingerprint=fingerprint)

    async def call(fn, *, mutation=False):
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn), timeout=35)
        except TimeoutError:
            message = "操作结果尚未确认，请查询操作回执，不要重复提交" if mutation else "资料读取超时，请稍后刷新"
            raise AccountError(503, message) from None
        except KeyError:
            raise AccountError(404, "资料不存在或不在授权范围内") from None
        except HistoryForbidden:
            return JSONResponse(status_code=403, content={"detail": "历史资料源未获授权或授权已失效",
                "error_code": "rag_source_forbidden"})
        except PermissionError:
            raise AccountError(403, "无权执行此资料操作") from None
        except Conflict as error:
            raise AccountError(409, str(error)) from None
        except ValueError:
            raise AccountError(422, "资料参数或版本无效") from None
        except HistoryUnavailable:
            raise AccountError(503, "历史资料源暂不可读，请检查来源连接、工作空间登记及授权；未返回部分资料冒充完整结果") from None
        except (StorageError, SQLAlchemyError):
            raise AccountError(503, "资料服务不可用，请稍后重试") from None

    def original_file(catalog, item_id, version, *, preview=False):
        from codex_agent.knowledge.store import KnowledgeStore
        from codex_agent.knowledge.schema import documents, versions
        from codex_agent.knowledge.files import file_path, source_row, preview_path
        scope, kind, source_id = catalog.resolve(item_id)
        if kind != "knowledge_document" or not version or not version.isdecimal():
            raise KeyError("No original file")
        library = KnowledgeStore(scope["root"])
        with library.engine.connect() as conn:
            library.row(conn, documents, source_id, allow_deleted=True)
            library.row(conn, versions, source_id, int(version))
            source = source_row(library, conn, source_id, int(version))
            if not source:
                raise KeyError("No original file")
            path = preview_path(library, source) if preview else file_path(library, source["file_id"])
            if path.is_symlink() or not path.is_file():
                raise KeyError("Missing original file")
            return path, source["filename"]

    def detail(catalog, item_id, version=None, source_id="current"):
        result = catalog.detail(item_id, version=version)
        result["corrections"] = actions.corrections(catalog, item_id)
        if result.get("source_file"):
            try:
                original_file(catalog, item_id, result["selected_version"])
                result["source_file"]["download_available"] = True
            except (KeyError, ValueError, OSError):
                result["source_file"]["download_available"] = False
            try:
                original_file(catalog, item_id, result["selected_version"], preview=True)
                result["source_file"]["preview_available"] = True
            except (KeyError, ValueError, OSError):
                result["source_file"]["preview_available"] = False
        if result["source_kind"] == "knowledge_document":
            from codex_agent.knowledge.store import KnowledgeStore
            from codex_agent.knowledge.cleanup import cleanup_status
            scope, _, document_source_id = catalog.resolve(item_id)
            library = KnowledgeStore(scope["root"])
            with library.engine.connect() as conn:
                status = cleanup_status(library, conn, document_source_id)
            result["cleanup_state"] = status.get("state", "unknown")
            result["cleanup_verified_at"] = status.get("verified_at")
        return sharing.decorate(config, source_id, catalog, item_id, result)

    def filters(q, source_kind, scope_id, status, view, origin=""):
        return dict(q=q, source_kind=source_kind, scope_id=scope_id, status=status, view=view, origin=origin)

    @app.get("/api/admin/rag/sources")
    async def sources(request: Request):
        await principal(request)
        return configured_sources(config)

    @app.get("/api/admin/rag/catalog")
    async def listing(request: Request, q: str = "", source_kind: str = "all", scope_id: str = "all",
                      status: str = "all", view: str = "active", cursor: str | None = None, limit: int = 25, source_id: str = "current", origin: str = ""):
        user = await principal(request)
        return await call(lambda: store(user, source_id).catalog(**filters(q, source_kind, scope_id, status, view, origin), cursor=cursor, limit=limit))

    @app.get("/api/admin/rag/catalog/summary")
    async def summary(request: Request, q: str = "", source_kind: str = "all", scope_id: str = "all",
                      status: str = "all", view: str = "active", source_id: str = "current", origin: str = ""):
        user = await principal(request)
        return await call(lambda: store(user, source_id).summary(**filters(q, source_kind, scope_id, status, view, origin)))

    @app.get("/api/admin/rag/catalog/reconciliation")
    async def reconciliation(request: Request, scope_id: str = "all", cursor: str | None = None,
                             index_cursor: str | None = None, limit: int = 25, source_id: str = "current"):
        user = await principal(request)
        return await call(lambda: store(user, source_id).reconciliation(scope_id=scope_id, cursor=cursor,
            index_cursor=index_cursor, limit=limit))

    @app.get("/api/admin/rag/operations/{request_id}")
    async def receipt(request_id: str, item_id: str, request: Request, source_id: str = "current"):
        user = await principal(request)
        if not 1 <= len(request_id) <= 100:
            raise AccountError(422, "Invalid request ID")
        return await call(lambda: actions.operation(store(user, source_id), user["id"], request_id, item_id))

    @app.get("/api/admin/rag/catalog/{item_id}")
    async def read(item_id: str, request: Request, version: str | None = None, source_id: str = "current"):
        user = await principal(request)
        return await call(lambda: detail(store(user, source_id), item_id, version, source_id))

    @app.get("/api/admin/rag/catalog/{item_id}/versions")
    async def versions(item_id: str, request: Request, cursor: str | None = None, limit: int = 25, source_id: str = "current"):
        user = await principal(request)
        return await call(lambda: store(user, source_id).versions(item_id, cursor=cursor, limit=limit))

    @app.get("/api/admin/rag/catalog/{item_id}/versions/{version}")
    async def version(item_id: str, version: str, request: Request, source_id: str = "current"):
        user = await principal(request)
        return await call(lambda: detail(store(user, source_id), item_id, version, source_id))

    @app.get("/api/admin/rag/catalog/{item_id}/chunks")
    async def chunks(item_id: str, version: str, request: Request, cursor: str | None = None, limit: int = 25, source_id: str = "current"):
        user = await principal(request)
        return await call(lambda: store(user, source_id).chunks(item_id, version, cursor=cursor, limit=limit))

    @app.get("/api/admin/rag/catalog/{item_id}/vectors")
    async def vectors(item_id: str, version: str, request: Request, source_id: str = "current"):
        user = await principal(request)
        await asyncio.to_thread(accounts.limit, "rag-vector-check", user["id"], 20, 60)
        from .vectors import inspect_vectors
        return await call(lambda: inspect_vectors(config, source_id, store(user, source_id), item_id, version))

    @app.get("/api/admin/rag/catalog/{item_id}/original")
    async def original(item_id: str, version: str, request: Request, source_id: str = "current"):
        user = await principal(request)
        result = await call(lambda: original_file(store(user, source_id), item_id, version))
        if isinstance(result, JSONResponse):
            return result
        path, filename = result
        return FileResponse(path, filename=filename, media_type="application/octet-stream", content_disposition_type="attachment",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/admin/rag/catalog/{item_id}/preview")
    async def image_preview(item_id: str, version: str, request: Request, source_id: str = "current"):
        user = await principal(request)
        result = await call(lambda: original_file(store(user, source_id), item_id, version, preview=True))
        if isinstance(result, JSONResponse):
            return result
        return FileResponse(result[0], media_type="image/png",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.post("/api/admin/rag/catalog/{item_id}/actions")
    async def mutate(item_id: str, request: Request, source_id: str = "current"):
        user = await principal(request, mutation=True)
        if source_id != "current":
            return JSONResponse(status_code=403, content={"detail": "历史资料源仅供只读查看，不能在此修改、停用或写入纠错",
                "error_code": "rag_source_forbidden"})
        try:
            value = ActionInput.model_validate_json(await limited_body(request, 100000))
        except ValueError:
            raise AccountError(422, "操作参数无效") from None
        def write():
            catalog = store(user, source_id)
            actions.execute(catalog, item_id, value, user["id"])
            return detail(catalog, item_id)
        return await call(write, mutation=True)

    @app.post("/api/admin/rag/catalog/{item_id}/share")
    async def share(item_id: str, request: Request, source_id: str = "current"):
        user = await principal(request, mutation=True)
        try:
            value = sharing.ShareInput.model_validate_json(await limited_body(request, 10000))
        except ValueError:
            raise AccountError(422, "共享资料参数无效") from None
        def write():
            sharing.grant(config, source_id)
            return sharing.share(config, source_id, store(user, source_id), item_id, value, user["id"])
        return await call(write, mutation=True)
