"""Read-only catalog projections. Source records, never vector payloads, are authority.

The caller supplies already-authorized scopes. Every query, including counts and
historical detail, binds those scopes again. No model or vector service is used.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re

from sqlalchemy import String, and_, case, cast, exists, func, literal, or_, select, union_all
from sqlalchemy.exc import SQLAlchemyError

from codex_agent.knowledge.file_schema import sources, locations
from codex_agent.knowledge.lifecycle import controls
from codex_agent.knowledge.source_links import source_links
from codex_agent.knowledge.schema import documents, versions, chunks as knowledge_chunks
from codex_agent.storage.database import engine, StorageError
from codex_agent.storage.schema import (
    memories, memory_versions, chunks as memory_chunks, memory_indexes,
    memory_index_entries, memory_index_chunks, counters,
)
from codex_agent.tasks.schema import jobs
from codex_agent.tasks.store import Conflict


KINDS = {"knowledge_document", "memory_case"}
MEMORY_FIELDS = ("memory_type", "operator", "semantics", "pytorch_reference", "summary",
                 "outcome", "confidence_grade", "source_run", "failure_stage", "error_signature")
SECTION_FIELDS = (("semantics", "问题与语义"), ("pytorch_reference", "参考代码"),
                  ("summary", "处理记录"), ("error_signature", "错误信息"),
                  ("outcome", "记录中的结果"))
FAILED_JOBS = {"failed", "dead", "cancelled", "needs-reconciliation"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _decoded(value, expected):
    try:
        result = json.loads(value)
        if not isinstance(result, expected):
            raise ValueError()
        return result
    except (ValueError, TypeError):
        raise StorageError("RAG source metadata is malformed") from None


def _safe_reference(value):
    value = str(value or "")
    return Path(value).name if value.startswith(("/", "~")) else value[:1000]


def _evidence(value, depth=0):
    """A bounded, explicit evidence display; no environment or embedding material."""
    if depth > 8:
        return "[内容层级过深]"
    if isinstance(value, dict):
        return {str(k): _evidence(v, depth + 1) for k, v in list(value.items())[:100]
                if not re.search(r"embedding|vector|password|secret|token|api.?key|credential|environment|^env$", str(k), re.I)}
    if isinstance(value, list):
        return [_evidence(v, depth + 1) for v in value[:100]]
    if isinstance(value, str):
        return _safe_reference(value) if value.startswith(("/", "~")) else value[:20000]
    return value if value is None or isinstance(value, (int, float, bool)) else str(value)


class CatalogStore:
    def __init__(self, scopes, *, cursor_key, database_engine=None, memory_mode="unknown",
                 knowledge_fingerprint=None):
        if not isinstance(cursor_key, bytes) or len(cursor_key) < 16:
            raise ValueError("A stable private catalog signing key is required")
        if not isinstance(scopes, (list, tuple)) or len(scopes) > 256:
            raise ValueError("Invalid authorized scopes")
        self.scopes = {}
        for scope in scopes:
            value = dict(scope)
            if (value.get("kind") not in KINDS or not re.fullmatch(r"[a-f0-9]{64}", value.get("id", ""))
                    or not isinstance(value.get("label"), str) or not isinstance(value.get("root"), (str, Path))):
                raise ValueError("Invalid authorized scope")
            value["root"] = Path(value["root"]).resolve()
            if hashlib.sha256(str(value["root"]).encode()).hexdigest() != value["id"]:
                raise ValueError("Scope root does not match its workspace")
            key = (value["kind"], value["id"])
            if key in self.scopes:
                raise ValueError("Duplicate authorized scope")
            value["writable"] = bool(value.get("writable", False))
            self.scopes[key] = value
        self.key = cursor_key
        self.engine = database_engine if database_engine is not None else engine()
        self.memory_mode = memory_mode
        self.knowledge_fingerprint = knowledge_fingerprint

    def _sign(self, domain, payload):
        body = base64.urlsafe_b64encode(_json(payload).encode()).decode().rstrip("=")
        signature = hmac.new(self.key, (domain + ":" + body).encode(), hashlib.sha256).hexdigest()
        return body + "." + signature

    def _unsign(self, domain, value):
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError("Invalid catalog token")
        try:
            body, signature = value.split(".")
            wanted = hmac.new(self.key, (domain + ":" + body).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, wanted):
                raise ValueError()
            return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        except (ValueError, TypeError, UnicodeError):
            raise ValueError("Invalid catalog token") from None

    def item_id(self, scope, kind, source_id):
        return self._sign("item", [kind, scope, str(source_id)])

    def resolve(self, item_id):
        try:
            kind, scope_id, source_id = self._unsign("item", item_id)
            scope = self.scopes[(kind, scope_id)]
            if ((kind == "knowledge_document" and not re.fullmatch(r"[a-f0-9]{32}", source_id))
                    or (kind == "memory_case" and not re.fullmatch(r"[1-9][0-9]{0,18}", source_id))):
                raise ValueError()
            return scope, kind, source_id
        except (ValueError, KeyError, TypeError):
            raise KeyError("Catalog item not found") from None

    def _scope_list(self):
        return [{k: value[k] for k in ("id", "label", "kind")} for value in self.scopes.values()]

    def _revision(self, conn):
        ids = sorted({scope[1] for scope in self.scopes})
        stmt = select(counters.c.workspace_id, counters.c.name, counters.c.value).where(
            counters.c.workspace_id.in_(ids),
            counters.c.name.in_(["cache-version:memory", "cache-version:rag-catalog"]))
        rows = [list(row) for row in conn.execute(stmt.order_by(counters.c.workspace_id, counters.c.name))]
        return _digest([[(k, v["writable"]) for k, v in sorted(self.scopes.items())], rows,
                        self.memory_mode, self.knowledge_fingerprint])

    @staticmethod
    def _field(name, value):
        return value.label(name)

    def _knowledge_query(self, scope):
        latest, published = versions.alias("latest"), versions.alias("published")
        joined = documents.outerjoin(latest, and_(latest.c.workspace_id == documents.c.workspace_id,
            latest.c.document_id == documents.c.id, latest.c.version == documents.c.latest_version))
        joined = joined.outerjoin(published, and_(published.c.workspace_id == documents.c.workspace_id,
            published.c.document_id == documents.c.id, published.c.version == documents.c.published_version))
        joined = joined.outerjoin(controls, and_(controls.c.workspace_id == documents.c.workspace_id,
            controls.c.document_id == documents.c.id))
        joined = joined.outerjoin(sources, and_(sources.c.workspace_id == documents.c.workspace_id,
            sources.c.document_id == documents.c.id, sources.c.version == documents.c.latest_version))
        joined = joined.outerjoin(jobs, and_(jobs.c.workspace_id == documents.c.workspace_id,
            jobs.c.id == latest.c.job_id))
        failed = and_(latest.c.state.in_(["pending", "parsing"]),
                      or_(jobs.c.status.in_(FAILED_JOBS), jobs.c.cancel_requested == 1))
        content = case((latest.c.document_id.is_(None), "missing"),
            (latest.c.state.in_(["parse_failed", "ocr_required", "ocr_empty"]), "parse_failed"),
            (and_(sources.c.document_id.is_not(None), sources.c.reviewed == 0), "review_required"),
            (and_(func.coalesce(latest.c.content, "") == "", latest.c.state != "parsing"), "missing"),
            else_="available")
        state = case((failed, "failed"), (latest.c.state == "ready", "ready"),
            (latest.c.state == "stale", "stale"),
            (latest.c.state == "pending", "queued"), (latest.c.state == "parsing", "building"),
            (latest.c.state.in_(["failed", "parse_failed", "ocr_required", "ocr_empty"]), "failed"),
            (latest.c.state == "draft", "not_indexed"), else_="unknown")
        if self.knowledge_fingerprint:
            state = case((and_(latest.c.state == "ready", or_(latest.c.fingerprint.is_(None), latest.c.fingerprint != self.knowledge_fingerprint)), "stale"), else_=state)
        lifecycle = case((controls.c.deleted_at.is_not(None), "trashed"), else_="active")
        retrieval = case((controls.c.deleted_at.is_not(None), "excluded"),
            (documents.c.published_version.is_(None), "excluded"),
            (or_(published.c.state.is_(None), published.c.state != "ready", func.coalesce(published.c.content, "") == ""), "unavailable"),
            *(([(or_(published.c.fingerprint.is_(None), published.c.fingerprint != self.knowledge_fingerprint), "unavailable")]) if self.knowledge_fingerprint else []),
            else_="eligible" if self.knowledge_fingerprint else "unknown")
        values = dict(source_kind=literal("knowledge_document"), scope_id=documents.c.workspace_id,
            source_id=documents.c.id, title=func.coalesce(latest.c.title, "原文记录缺失"),
            source=func.coalesce(latest.c.source, ""), content=func.coalesce(latest.c.content, ""),
            serving_content=func.coalesce(published.c.content, ""),
            format=func.coalesce(sources.c.format, "text"), filename=sources.c.filename,
            latest_version=cast(documents.c.latest_version, String),
            stored_serving_version=cast(documents.c.published_version, String),
            content_state=content, lifecycle_state=lifecycle, index_state=state,
            retrieval_state=retrieval, updated_at=documents.c.updated_at,
            lifecycle_revision=func.coalesce(controls.c.revision, 0),
            source_revision=func.coalesce(latest.c.source_hash, ""),
            chunk_count=func.coalesce(latest.c.chunk_count, 0),
            index_revision=func.coalesce(latest.c.fingerprint, ""),
            latest_state=func.coalesce(latest.c.state, "missing"))
        return select(*[self._field(k, v) for k, v in values.items()]).select_from(joined).where(documents.c.workspace_id == scope["id"])

    def _checked_memory_query(self, conn, scope):
        eligible_generation = None
        if self.memory_mode == "qdrant":
            # Share the retrieval gate without creating a provider or querying vectors.
            # It reads at most two active generation definitions, once per scope/query,
            # rather than hashing configuration once for every source record.
            from codex_agent.indexing.retrieval import _index, RetrievalUnavailable

            class ScopedReader:
                def rows(_self, table, *conditions, limit=None):
                    if table is not memory_indexes:
                        raise ValueError("Unexpected catalog index lookup")
                    query = select(table).where(table.c.workspace_id == scope["id"], *conditions).limit(limit)
                    return [dict(row) for row in conn.execute(query).mappings()]

            try:
                active, _definition = _index(ScopedReader(), None, False)
                eligible_generation = active["generation"]
            except (RetrievalUnavailable, ValueError, TypeError, KeyError):
                # Corrupt/incompatible index metadata must not hide readable source
                # material or imply that the source can currently be retrieved.
                pass
        return self._memory_query(scope, eligible_generation=eligible_generation)

    def _memory_query(self, scope, *, eligible_generation=None):
        active = select(memory_indexes.c.workspace_id, func.count().label("n"),
            func.min(memory_indexes.c.generation).label("generation")).where(memory_indexes.c.status == "active",
                memory_indexes.c.workspace_id == scope["id"]).group_by(memory_indexes.c.workspace_id).subquery()
        joined = memories.outerjoin(active, active.c.workspace_id == memories.c.workspace_id)
        joined = joined.outerjoin(memory_index_entries, and_(memory_index_entries.c.workspace_id == memories.c.workspace_id,
            memory_index_entries.c.memory_id == memories.c.id, memory_index_entries.c.generation == active.c.generation))
        entry = memory_index_entries.c
        metadata_ready = and_(active.c.n == 1, entry.status == "ready", entry.desired == "present",
                              entry.case_version == memories.c.case_version)
        definition_valid = active.c.generation == eligible_generation if eligible_generation else literal(False)
        ready = and_(metadata_ready, definition_valid) if self.memory_mode == "qdrant" else metadata_ready
        index = case((active.c.n > 1, "unknown"), (ready, "ready"),
            *(([(metadata_ready, "unknown")]) if self.memory_mode == "qdrant" else []),
            (and_(entry.status == "ready", entry.case_version != memories.c.case_version), "stale"),
            (entry.status.in_(["failed", "dead"]), "failed"),
            (entry.status.in_(["queued", "pending"]), "queued"),
            (entry.status == "running", "building"), (entry.memory_id.is_(None), "not_indexed"), else_="unknown")
        content = case((and_(func.coalesce(memories.c.summary, "") == "", func.coalesce(memories.c.semantics, "") == "",
            func.coalesce(memories.c.pytorch_reference, "") == ""), "missing"), else_="available")
        if self.memory_mode == "qdrant":
            available = case((ready, "eligible"), else_="unavailable")
        elif self.memory_mode in {"legacy", "jaccard"}:
            available = literal("eligible")
        else:
            available = literal("unknown")
        retrieval = case((memories.c.active != 1, "excluded"), (content == "missing", "unavailable"), else_=available)
        values = dict(source_kind=literal("memory_case"), scope_id=memories.c.workspace_id,
            source_id=cast(memories.c.id, String), title=memories.c.operator + literal(" · ") + memories.c.memory_type,
            source=memories.c.source_run, content=memories.c.searchable_text,
            serving_content=memories.c.searchable_text, format=literal("memory"), filename=literal(None, String),
            latest_version=func.coalesce(memories.c.case_version, "legacy"),
            stored_serving_version=func.coalesce(memories.c.case_version, "legacy"),
            content_state=content, lifecycle_state=case((memories.c.active == 1, "active"), else_="archived"),
            index_state=index, retrieval_state=retrieval, updated_at=memories.c.updated_at,
            lifecycle_revision=literal(0), source_revision=memories.c.fingerprint,
            chunk_count=func.coalesce(entry.point_count, 0), index_revision=func.coalesce(cast(entry.revision, String), ""),
            latest_state=literal("recorded"))
        return select(*[self._field(k, v) for k, v in values.items()]).select_from(joined).where(memories.c.workspace_id == scope["id"])

    def _views(self, conn):
        queries, health = [], []
        for scope in self.scopes.values():
            try:
                query = self._knowledge_query(scope) if scope["kind"] == "knowledge_document" else self._checked_memory_query(conn, scope)
                conn.execute(query.limit(0))
            except SQLAlchemyError:
                health.append(dict(scope_id=scope["id"], source_kind=scope["kind"], status="unavailable", reason="资料源暂不可读，请检查数据库结构或连接"))
                continue
            queries.append(query)
            health.append(dict(scope_id=scope["id"], source_kind=scope["kind"], status="ready"))
        return (union_all(*queries).subquery() if queries else None), health

    def _validate(self, q, source_kind, scope_id, status, view, limit):
        if (not isinstance(q, str) or len(q) > 200 or "\0" in q or source_kind not in KINDS | {"all"}
                or scope_id not in {"all"} | {s[1] for s in self.scopes}
                or status not in {"all", "eligible", "pending", "attention", "excluded"}
                or view not in {"active", "trash"} or type(limit) is not int or not 1 <= limit <= 50):
            raise ValueError("Invalid catalog filters")

    @staticmethod
    def _pending(table):
        return and_(table.c.lifecycle_state == "active", or_(table.c.index_state.in_(["queued", "building"]),
            and_(table.c.source_kind == "knowledge_document", table.c.index_state == "not_indexed")))

    @staticmethod
    def _attention(table):
        return or_(table.c.content_state.in_(["missing", "parse_failed"]),
                   table.c.index_state.in_(["failed", "stale", "unknown"]), table.c.retrieval_state == "unavailable")

    def _conditions(self, table, q, source_kind, scope_id, status, view, origin=""):
        conditions = [table.c.lifecycle_state == "trashed" if view == "trash" else table.c.lifecycle_state != "trashed"]
        if origin:
            if origin != "uploaded" and not re.fullmatch(r"history:[a-z][a-z0-9_-]{0,47}", origin):
                raise ValueError("Invalid source category")
            linked = select(source_links.c.origin_key).where(source_links.c.workspace_id == table.c.scope_id,
                source_links.c.document_id == table.c.source_id)
            if origin != "uploaded":
                linked = linked.where(source_links.c.source_alias == origin.removeprefix("history:"))
            conditions.extend([table.c.source_kind == "knowledge_document",
                               ~exists(linked) if origin == "uploaded" else exists(linked)])
        if source_kind != "all": conditions.append(table.c.source_kind == source_kind)
        if scope_id != "all": conditions.append(table.c.scope_id == scope_id)
        if q.strip():
            conditions.append(or_(*[func.lower(getattr(table.c, name)).contains(q.strip().lower(), autoescape=True)
                                    for name in ("title", "source", "content", "serving_content")]))
        if status == "eligible": conditions.append(table.c.retrieval_state == "eligible")
        elif status == "pending": conditions.append(self._pending(table))
        elif status == "attention": conditions.append(self._attention(table))
        elif status == "excluded": conditions.append(table.c.retrieval_state != "eligible")
        return conditions

    def _project(self, row):
        scope = self.scopes[(row["source_kind"], row["scope_id"])]
        serving = row["stored_serving_version"] if row["retrieval_state"] == "eligible" else None
        reason = {"eligible": "资料当前符合配置的检索条件；不表示已在某次任务命中", "excluded": "资料未发布、已停用或已移入回收站",
                  "unavailable": "原文或当前索引版本不可用", "unknown": "尚未确认当前检索配置与索引是否一致"}[row["retrieval_state"]]
        capabilities = ["view", "versions", "chunks"]
        if scope["writable"]:
            if row["source_kind"] == "memory_case":
                capabilities += ["correct"] + (["archive"] if row["lifecycle_state"] == "active" else [])
            elif row["lifecycle_state"] == "trashed": capabilities += ["restore"]
            else:
                capabilities += ["delete"]
                if row["content_state"] in {"available", "review_required"}: capabilities += ["edit"]
                if row["content_state"] == "available" and row["latest_state"] in {"draft", "failed"}: capabilities += ["index"]
                if row["index_state"] == "ready" and row["content_state"] == "available": capabilities += ["publish"]
                if row["stored_serving_version"]: capabilities += ["unpublish"]
        revision = _digest({k: row[k] for k in ("latest_version", "stored_serving_version", "updated_at", "lifecycle_revision",
            "source_revision", "index_revision", "index_state", "lifecycle_state", "content_state")})
        return dict(item_id=self.item_id(scope["id"], scope["kind"], row["source_id"]), source_kind=scope["kind"],
            source_id=str(row["source_id"]), scope={k: scope[k] for k in ("id", "label", "kind")},
            title=row["title"], excerpt=(row.get("excerpt") or "")[:240], source=_safe_reference(row["source"]),
            format=row["format"], filename=row["filename"], latest_version=row["latest_version"], serving_version=serving,
            content_state=row["content_state"], lifecycle_state=row["lifecycle_state"], index_state=row["index_state"],
            retrieval_state=row["retrieval_state"], retrieval_reason=reason,
            cleanup_state="unknown" if row["lifecycle_state"] in {"archived", "trashed"} else "not_requested",
            updated_at=row["updated_at"], verified_at=None, revision=revision,
            catalog_revision=int(row["lifecycle_revision"]), chunk_count=int(row["chunk_count"]), capabilities=capabilities)

    def catalog(self, *, q="", source_kind="all", scope_id="all", status="all", view="active", cursor=None, limit=25, origin=""):
        self._validate(q, source_kind, scope_id, status, view, limit)
        filters = [q.strip(), source_kind, scope_id, status, view, limit, origin]
        try:
            with self.engine.connect() as conn:
                revision = self._revision(conn)
                relation, health = self._views(conn)
                complete = all(h["status"] == "ready" for h in health)
                after = None
                if cursor:
                    token = self._unsign("catalog", cursor)
                    if not isinstance(token, dict): raise ValueError("Invalid catalog cursor")
                    if token.get("revision") != revision or token.get("filters") != filters or token.get("health") != _digest(health):
                        raise Conflict("资料或权限已更新，请刷新目录后继续")
                    after = token.get("after")
                    if not isinstance(after, list) or len(after) != 4 or not all(isinstance(x, str) for x in after):
                        raise ValueError("Invalid catalog cursor")
                rows, total, next_cursor = [], 0, None
                if relation is not None:
                    cond = self._conditions(relation, q, source_kind, scope_id, status, view, origin)
                    total = conn.execute(select(func.count()).select_from(relation).where(*cond)).scalar_one()
                    c = relation.c
                    if after:
                        stamp, kind, sid, source = after
                        cond.append(or_(c.updated_at < stamp,
                            and_(c.updated_at == stamp, c.source_kind > kind),
                            and_(c.updated_at == stamp, c.source_kind == kind, c.scope_id > sid),
                            and_(c.updated_at == stamp, c.source_kind == kind, c.scope_id == sid, c.source_id > source)))
                    columns = [column for column in relation.c if column.name not in {"content", "serving_content"}]
                    stmt = select(*columns, func.substr(c.content, 1, 240).label("excerpt")).where(*cond).order_by(
                        c.updated_at.desc(), c.source_kind, c.scope_id, c.source_id).limit(limit + 1)
                    found = list(conn.execute(stmt).mappings())
                    rows = [self._project(dict(row)) for row in found[:limit]]
                    if len(found) > limit:
                        last = found[limit - 1]
                        next_cursor = self._sign("catalog", dict(revision=revision, filters=filters, health=_digest(health),
                            after=[last[k] for k in ("updated_at", "source_kind", "scope_id", "source_id")]))
                if revision != self._revision(conn):
                    raise Conflict("资料在读取时发生变化，请刷新")
                return dict(items=rows, total=total, next_cursor=next_cursor, catalog_revision=revision,
                    complete=complete, source_health=health, scopes=self._scope_list(), as_of=_now())
        except SQLAlchemyError:
            raise StorageError("RAG catalog storage unavailable") from None

    def summary(self, *, q="", source_kind="all", scope_id="all", status="all", view="active", origin=""):
        self._validate(q, source_kind, scope_id, status, view, 25)
        try:
            with self.engine.connect() as conn:
                revision = self._revision(conn)
                relation, health = self._views(conn)
                counts = dict(total=0, eligible=0, pending=0, attention=0, excluded=0)
                if relation is not None:
                    c = relation.c
                    query = select(func.count().label("total"),
                        func.coalesce(func.sum(case((c.retrieval_state == "eligible", 1), else_=0)), 0).label("eligible"),
                        func.coalesce(func.sum(case((self._pending(relation), 1), else_=0)), 0).label("pending"),
                        func.coalesce(func.sum(case((self._attention(relation), 1), else_=0)), 0).label("attention"),
                        func.coalesce(func.sum(case((c.retrieval_state != "eligible", 1), else_=0)), 0).label("excluded"))
                    counts = {k: int(v) for k, v in conn.execute(query.select_from(relation).where(
                        *self._conditions(relation, q, source_kind, scope_id, status, view, origin))).mappings().one().items()}
                if revision != self._revision(conn): raise Conflict("资料在读取时发生变化，请刷新")
                return dict(**counts, complete=all(h["status"] == "ready" for h in health), source_health=health,
                            scopes=self._scope_list(), catalog_revision=revision, as_of=_now())
        except SQLAlchemyError:
            raise StorageError("RAG catalog storage unavailable") from None

    def _item(self, conn, item_id):
        scope, kind, identifier = self.resolve(item_id)
        query = self._knowledge_query(scope) if kind == "knowledge_document" else self._checked_memory_query(conn, scope)
        relation = query.subquery()
        row = conn.execute(select(relation).where(relation.c.source_id == identifier)).mappings().first()
        if not row: raise KeyError("Catalog item not found")
        value = dict(row)
        value["excerpt"] = value["content"][:240]
        return scope, kind, identifier, self._project(value)

    def revision_for(self, conn, item_id):
        """Read after the caller locks the source row, on its transaction connection."""
        return self._item(conn, item_id)[3]["revision"]

    @staticmethod
    def _row(conn, table, scope, *conditions):
        row = conn.execute(select(table).where(table.c.workspace_id == scope["id"], *conditions)).mappings().first()
        return dict(row) if row else None

    def detail(self, item_id, version=None):
        try:
            with self.engine.connect() as conn:
                revision = self._revision(conn)
                scope, kind, identifier, item = self._item(conn, item_id)
                selected = version if version is not None else item["serving_version"] or item["latest_version"]
                if not isinstance(selected, str) or not 1 <= len(selected) <= 128: raise ValueError("Invalid source version")
                sections, source_file, provenance = [], None, {"source_kind": kind, "source_id": identifier}
                selected_title, selected_source = item["title"], item["source"]
                if kind == "knowledge_document":
                    if not re.fullmatch(r"[1-9][0-9]{0,8}", selected): raise ValueError("Invalid document version")
                    row = self._row(conn, versions, scope, versions.c.document_id == identifier, versions.c.version == int(selected))
                    if not row: raise KeyError("Source version not found")
                    content = row["content"]
                    selected_title, selected_source = row["title"], row["source"]
                    origin = self._row(conn, sources, scope, sources.c.document_id == identifier, sources.c.version == int(selected))
                    if origin:
                        for part in _decoded(origin["sections_json"], list):
                            if not isinstance(part, dict) or not isinstance(part.get("id"), str) or not isinstance(part.get("text"), str):
                                raise StorageError("RAG source metadata is malformed")
                            sections.append(dict(id=part["id"], label=part.get("locator", "原文段落"), text=part["text"], locator=part.get("locator", "")))
                        file_id = origin["file_id"]
                        file = scope["root"] / "originals" / file_id
                        available = bool(re.fullmatch(r"[a-f0-9]{32}", file_id) and not file.is_symlink()
                            and file.is_file() and file.parent.resolve() == scope["root"] / "originals")
                        source_file = {k: origin[k] for k in ("filename", "format", "sha256", "parser_version")}
                        source_file.update(reviewed=bool(origin["reviewed"]), warnings=_decoded(origin["warnings_json"], list), download_available=available)
                    if not sections: sections = [dict(id="content", label="原文", text=content, locator="")]
                    provenance.update(source=row["source"], original_file_available=source_file["download_available"] if source_file else None)
                else:
                    current = conn.execute(select(*[getattr(memories.c, key) for key in MEMORY_FIELDS], memories.c.evidence_json).where(
                        memories.c.workspace_id == scope["id"], memories.c.id == int(identifier))).mappings().first()
                    if current is None: raise KeyError("Catalog item not found")
                    if selected == item["latest_version"]:
                        payload = {k: current[k] for k in MEMORY_FIELDS}
                        payload["evidence"] = _decoded(current["evidence_json"] or "{}", (dict, list))
                    else:
                        snapshot = self._row(conn, memory_versions, scope, memory_versions.c.memory_id == int(identifier), memory_versions.c.case_version == selected)
                        if not snapshot: raise KeyError("Source version not found")
                        raw = _decoded(snapshot["payload_json"], dict)
                        payload = {k: raw.get(k) for k in MEMORY_FIELDS}
                        payload["evidence"] = raw.get("evidence", {})
                    selected_title = " · ".join(str(payload[k]) for k in ("operator", "memory_type") if payload.get(k)) or "历史经验记录"
                    selected_source = payload.get("source_run") or ""
                    sections = [dict(id=k, label=label, text=str(payload[k]), locator=k) for k, label in SECTION_FIELDS if payload.get(k)]
                    evidence = _evidence(payload.get("evidence", {}))
                    if evidence: sections.append(dict(id="evidence", label="原始证据记录", text=json.dumps(evidence, ensure_ascii=False, indent=2), locator="evidence"))
                    content = "\n\n".join(s["label"] + "\n" + s["text"] for s in sections)
                    provenance.update(source_run=_safe_reference(payload.get("source_run")), confidence_grade=payload.get("confidence_grade"), evidence_status="原始记录中的结果，不代表管理员已重新验证")
                if revision != self._revision(conn): raise Conflict("资料在读取时发生变化，请刷新")
                return {**item, "selected_version": selected, "selected_title": selected_title,
                        "selected_source": _safe_reference(selected_source), "content": content, "sections": sections,
                        "source_file": source_file, "provenance": provenance}
        except SQLAlchemyError:
            raise StorageError("RAG source unavailable") from None

    def _page(self, item_id, version, cursor, limit, domain, builder):
        if type(limit) is not int or not 1 <= limit <= 50: raise ValueError("Invalid page size")
        try:
            with self.engine.connect() as conn:
                revision = self._revision(conn)
                scope, kind, identifier, item = self._item(conn, item_id)
                offset = 0
                if cursor:
                    value = self._unsign(domain, cursor)
                    if not isinstance(value, dict): raise ValueError("Invalid catalog cursor")
                    if value.get("revision") != revision or value.get("binding") != [item_id, version, limit]:
                        raise Conflict("资料或权限已变化，请重新读取")
                    offset = value.get("offset")
                    if type(offset) is not int or not 0 <= offset <= 100000: raise ValueError("Invalid page")
                query, convert = builder(scope, kind, identifier, item, conn)
                found = list(conn.execute(query.offset(offset).limit(limit + 1)).mappings())
                if revision != self._revision(conn): raise Conflict("资料在读取时发生变化，请刷新")
                return dict(items=[convert(dict(r)) for r in found[:limit]], next_cursor=self._sign(domain,
                    dict(revision=revision, binding=[item_id, version, limit], offset=offset + limit)) if len(found) > limit else None,
                    version=version, catalog_revision=revision)
        except SQLAlchemyError:
            raise StorageError("RAG source unavailable") from None

    def versions(self, item_id, *, cursor=None, limit=25):
        def builder(scope, kind, identifier, item, conn):
            if kind == "knowledge_document":
                query = select(versions.c.version, versions.c.state, versions.c.chunk_count, versions.c.created_at).where(
                    versions.c.workspace_id == scope["id"], versions.c.document_id == identifier).order_by(versions.c.version.desc())
            else:
                history = select(memory_versions.c.case_version.label("version"), literal("recorded").label("state"),
                    literal(0).label("chunk_count"), memory_versions.c.captured_at.label("created_at")).where(
                    memory_versions.c.workspace_id == scope["id"], memory_versions.c.memory_id == int(identifier))
                # Pre-versioning imports may still have a readable current source.
                current = select(func.coalesce(memories.c.case_version, "legacy").label("version"),
                    literal("recorded").label("state"), literal(0).label("chunk_count"), memories.c.updated_at.label("created_at")).where(
                    memories.c.workspace_id == scope["id"], memories.c.id == int(identifier),
                    ~exists(select(memory_versions.c.memory_id).where(memory_versions.c.workspace_id == scope["id"],
                        memory_versions.c.memory_id == int(identifier), memory_versions.c.case_version == memories.c.case_version)))
                combined = union_all(history, current).subquery()
                query = select(combined).order_by(combined.c.created_at.desc(), combined.c.version)
            def convert(row):
                row["version"] = str(row["version"])
                return {**row, "is_latest": row["version"] == item["latest_version"], "is_serving": row["version"] == item["serving_version"]}
            return query, convert
        return self._page(item_id, None, cursor, limit, "versions", builder)

    def chunks(self, item_id, version, *, cursor=None, limit=25):
        # Verify that this exact source version exists before displaying any fragment.
        self.detail(item_id, version)
        def builder(scope, kind, identifier, item, conn):
            if kind == "knowledge_document":
                joined = knowledge_chunks.outerjoin(locations, and_(locations.c.workspace_id == knowledge_chunks.c.workspace_id,
                    locations.c.document_id == knowledge_chunks.c.document_id, locations.c.version == knowledge_chunks.c.version,
                    locations.c.chunk_id == knowledge_chunks.c.id))
                query = select(knowledge_chunks.c.id, knowledge_chunks.c.text, knowledge_chunks.c.position,
                    func.coalesce(locations.c.locator, "").label("locator")).select_from(joined).where(
                    knowledge_chunks.c.workspace_id == scope["id"], knowledge_chunks.c.document_id == identifier,
                    knowledge_chunks.c.version == int(version)).order_by(knowledge_chunks.c.position, knowledge_chunks.c.id)
            elif self.memory_mode == "qdrant" or version != item["latest_version"]:
                ic = memory_index_chunks
                query = select(ic.c.chunk_id.label("id"), ic.c.text, ic.c.provenance_json,
                    ic.c.generation, ic.c.revision).where(ic.c.workspace_id == scope["id"],
                    ic.c.memory_id == int(identifier), ic.c.case_version == version)
                if version == item["serving_version"] and self.memory_mode == "qdrant":
                    joined = ic.join(memory_indexes, and_(memory_indexes.c.workspace_id == ic.c.workspace_id,
                        memory_indexes.c.generation == ic.c.generation, memory_indexes.c.status == "active"))
                    joined = joined.join(memory_index_entries, and_(memory_index_entries.c.workspace_id == ic.c.workspace_id,
                        memory_index_entries.c.generation == ic.c.generation, memory_index_entries.c.memory_id == ic.c.memory_id,
                        memory_index_entries.c.revision == ic.c.revision, memory_index_entries.c.case_version == ic.c.case_version))
                    query = query.select_from(joined).where(memory_index_entries.c.status == "ready", memory_index_entries.c.desired == "present")
                query = query.order_by(ic.c.generation, ic.c.revision.desc(), ic.c.chunk_id)
                def convert(row):
                    raw = _decoded(row.pop("provenance_json"), dict)
                    row.update(position=raw.get("position", 0), locator=str(raw.get("source_field", "历史索引片段")))
                    return row
                return query, convert
            else:
                query = select(func.coalesce(memory_chunks.c.chunk_id, memory_chunks.c.kind + literal(":") + cast(memory_chunks.c.position, String)).label("id"),
                    memory_chunks.c.text, memory_chunks.c.position, memory_chunks.c.source_field.label("locator")).where(
                    memory_chunks.c.workspace_id == scope["id"], memory_chunks.c.memory_id == int(identifier),
                    memory_chunks.c.case_version == version).order_by(memory_chunks.c.kind, memory_chunks.c.position)
            return query, lambda row: row
        return self._page(item_id, version, cursor, limit, "chunks", builder)

    def _orphan_query(self, scope):
        """Count mappings per missing source, never expose vector points as documents."""
        if scope["kind"] == "memory_case":
            mappings = [(memory_index_entries, memory_index_entries.c.memory_id, "index_entries"),
                        (memory_index_chunks, memory_index_chunks.c.memory_id, "index_chunks")]
            parent = memories
        else:
            mappings = [(versions, versions.c.document_id, "index_entries"),
                        (knowledge_chunks, knowledge_chunks.c.document_id, "index_chunks")]
            parent = documents
        queries = []
        for table, identifier, count_name in mappings:
            joined = table.outerjoin(parent, and_(table.c.workspace_id == parent.c.workspace_id, identifier == parent.c.id))
            queries.append(select(literal(scope["kind"]).label("source_kind"), table.c.workspace_id.label("scope_id"),
                cast(identifier, String).label("source_id"),
                (func.count() if count_name == "index_entries" else literal(0)).label("index_entries"),
                (func.count() if count_name == "index_chunks" else literal(0)).label("index_chunks"))
                .select_from(joined).where(table.c.workspace_id == scope["id"], parent.c.id.is_(None))
                .group_by(table.c.workspace_id, identifier))
        return union_all(*queries)

    def _index_reconciliation(self, conn, scope_id, limit, cursor, revision):
        queries, health = [], []
        for scope in self.scopes.values():
            if scope_id != "all" and scope["id"] != scope_id: continue
            query = self._orphan_query(scope)
            try:
                conn.execute(query.limit(0))
            except SQLAlchemyError:
                health.append(dict(scope_id=scope["id"], source_kind=scope["kind"], status="unavailable"))
                continue
            queries.extend(query.selects)
            health.append(dict(scope_id=scope["id"], source_kind=scope["kind"], status="ready"))
        binding = [scope_id, limit, _digest(health)]
        offset = 0
        if cursor:
            value = self._unsign("index-reconciliation", cursor)
            if not isinstance(value, dict): raise ValueError("Invalid reconciliation cursor")
            if value.get("revision") != revision or value.get("binding") != binding:
                raise Conflict("资料或权限已变化，请重新对账")
            offset = value.get("offset")
            if type(offset) is not int or not 0 <= offset <= 100000: raise ValueError("Invalid page")
        rows, total, next_cursor = [], 0, None
        if queries:
            relation = union_all(*queries).subquery()
            c = relation.c
            grouped = select(c.source_kind, c.scope_id, c.source_id,
                func.sum(c.index_entries).label("index_entries"), func.sum(c.index_chunks).label("index_chunks"))\
                .group_by(c.source_kind, c.scope_id, c.source_id).subquery()
            total = conn.execute(select(func.count()).select_from(grouped)).scalar_one()
            found = list(conn.execute(select(grouped).order_by(grouped.c.source_kind, grouped.c.scope_id,
                grouped.c.source_id).offset(offset).limit(limit + 1)).mappings())
            for row in found[:limit]:
                scope = self.scopes[(row["source_kind"], row["scope_id"])]
                rows.append(dict(source_kind=row["source_kind"], source_id=row["source_id"],
                    scope={key: scope[key] for key in ("id", "label", "kind")}, reason="source_missing",
                    index_entries=int(row["index_entries"]), index_chunks=int(row["index_chunks"])))
            if len(found) > limit:
                next_cursor = self._sign("index-reconciliation", dict(revision=revision, binding=binding, offset=offset + limit))
        return dict(index_anomalies=rows, index_anomalies_total=total, index_anomalies_next_cursor=next_cursor,
                    index_scan_complete=all(h["status"] == "ready" for h in health), index_source_health=health)

    def reconciliation(self, *, cursor=None, limit=25, scope_id="all", index_cursor=None):
        result = self.catalog(status="attention", cursor=cursor, limit=limit, scope_id=scope_id)
        try:
            with self.engine.connect() as conn:
                revision = self._revision(conn)
                if revision != result["catalog_revision"]: raise Conflict("资料在读取时发生变化，请重新对账")
                result.update(self._index_reconciliation(conn, scope_id, limit, index_cursor, revision))
                if revision != self._revision(conn): raise Conflict("资料在读取时发生变化，请重新对账")
        except SQLAlchemyError:
            raise StorageError("RAG reconciliation unavailable") from None
        result.update(complete=result["complete"] and result["index_scan_complete"],
            coverage="authorized-source-and-database-index-metadata", vector_scan_performed=False,
            limitation="已核对授权资料和数据库中缺少源记录的索引映射，异常按来源资料去重；未扫描向量集合，不能证明不存在孤立向量，也未恢复缺失原文。")
        return result
