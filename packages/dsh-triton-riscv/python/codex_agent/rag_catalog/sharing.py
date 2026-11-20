"""Copy an explicitly granted historical original to a shared draft, exactly once."""
import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.engine import make_url

from codex_agent.accounts.schema import users
from codex_agent.knowledge.source_links import source_links
from codex_agent.knowledge.store import DocumentInput, KnowledgeStore
from codex_agent.knowledge.schema import documents
from codex_agent.platform.store import timestamp
from codex_agent.storage.schema import meta
from codex_agent.tasks.store import Conflict, digest, encode
from .sources import HistoryForbidden, source_config, historical_scopes


class ShareInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    expected_revision: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def valid(self):
        if "\0" in self.expected_revision or "\0" in self.request_id:
            raise ValueError("Invalid sharing request")
        return self


def grant(config, source_id):
    if not config.knowledge_enabled or source_id == "current":
        raise PermissionError("Only explicitly granted historical sources can be shared")
    source = source_config(config, source_id)
    if not source.allow_share:
        raise HistoryForbidden("This source is readable but not granted for sharing")
    return source


def identity(config, source_id, catalog, item_id):
    source = grant(config, source_id)
    scope, kind, identifier = catalog.resolve(item_id)
    if kind != "memory_case" or scope["id"] not in source.workspace_ids:
        raise PermissionError("Unknown historical source kind or scope")
    # Credential rotation does not duplicate documents, while switching the
    # configured database cannot silently resolve an unrelated source's copy.
    url = make_url(os.environ[source.url_env])
    database_identity = [url.drivername, url.host, url.port, url.database, url.query.get("unix_socket")]
    key = digest([source.id, database_identity, scope["id"], kind, identifier])
    return source, scope, kind, identifier, key


def decorate(config, source_id, catalog, item_id, detail):
    if (source_id == "current" or not config.knowledge_enabled
            or not source_config(config, source_id).allow_share):
        return detail
    _, _, _, _, key = identity(config, source_id, catalog, item_id)
    library = KnowledgeStore(Path(config.data_dir) / "knowledge")
    with library.engine.connect() as conn:
        found = library.rows(conn, source_links, source_links.c.origin_key == key)
    if found:
        detail["shared_document_id"] = found[0]["document_id"]
    if (detail["content_state"] == "available" and detail["content"].strip()
            and detail["selected_version"] == detail["latest_version"]):
        detail["capabilities"] = [*detail["capabilities"], "share_to_shared"]
    return detail


def share(config, source_id, catalog, item_id, value, actor):
    source, scope, kind, identifier, origin_key = identity(config, source_id, catalog, item_id)
    # Re-read the latest original. The browser never supplies source content,
    # an owner, a root path or a destination database.
    current = catalog.detail(item_id)
    if current["selected_version"] != current["latest_version"]:
        current = catalog.detail(item_id, version=current["latest_version"])
    if current["revision"] != value.expected_revision:
        raise Conflict("历史资料已变化，请重新读取后再加入共享库")
    if current["content_state"] != "available" or not current["content"].strip():
        raise Conflict("没有可读的完整原文，不能创建共享资料")
    title = current.get("selected_title") or current["title"]
    provenance = (f"历史来源：{source.label}；来源编号：{identifier}；"
                  f"版本：{current['latest_version']}；原来源状态：{current['lifecycle_state']}。"
                  "这是一份管理员待审核的可读资料副本，原始任务记录未被改写。")
    evidence = current.get("provenance", {})
    context = "\n".join(f"{label}：{evidence[key]}" for key, label in (
        ("source_run", "来源任务"), ("confidence_grade", "原记录可信等级"),
        ("evidence_status", "证据说明")) if evidence.get(key))
    content = (context + "\n\n" if context else "") + current["content"]
    library = KnowledgeStore(Path(config.data_dir) / "knowledge")
    receipt_key = "shared-import:" + digest([actor, value.request_id])
    fingerprint = digest([origin_key, value.expected_revision])
    db = library.writer()
    with db.transaction() as conn:
        db.lock(conn, "knowledge-write")
        if actor not in config.admin_user_ids or not conn.execute(select(users.c.id).where(
                users.c.id == actor, users.c.status == "active")).scalar_one_or_none():
            raise PermissionError("Administrator access revoked")
        # Read permission and sharing permission must both still hold at commit.
        historical_scopes(source, catalog.engine)
        if catalog.detail(item_id, version=current["latest_version"])["revision"] != value.expected_revision:
            raise Conflict("历史资料在导入时发生变化，请重新读取")
        prior = library.rows(conn, meta, meta.c.key == receipt_key)
        if prior and prior[0]["value"] != fingerprint:
            raise Conflict("分享请求编号已用于其他资料，请重新读取")
        linked = library.rows(conn, source_links, source_links.c.origin_key == origin_key)
        if linked:
            # Never overwrite administrator edits or resurrect a deleted copy.
            doc = library.row(conn, documents, linked[0]["document_id"], allow_deleted=True)
            if not prior:
                db.insert(conn, meta, dict(key=receipt_key, value=fingerprint))
            return library.summary(conn, doc)
        draft = DocumentInput(title=title, source=provenance, content=content,
            request_id="historical-import:" + origin_key)
        result = library.save_transaction(draft, actor, db, conn)
        db.insert(conn, source_links, dict(origin_key=origin_key, source_alias=source.id,
            source_scope=scope["id"], source_kind=kind, source_id=identifier,
            source_version=current["latest_version"], source_revision=value.expected_revision,
            source_content_hash=hashlib.sha256(content.encode()).hexdigest(),
            document_id=result["id"], actor_id=actor, created_at=timestamp()))
        db.insert(conn, meta, dict(key=receipt_key, value=fingerprint))
        library.log(db, conn, actor, "historical-source-imported", result["id"], result["latest_version"])
        return result
