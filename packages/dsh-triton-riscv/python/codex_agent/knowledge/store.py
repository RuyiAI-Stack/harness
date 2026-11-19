"""Drafts and publication transactions; immutable source versions stay in MySQL."""
import hashlib
import json
from pathlib import Path
import secrets

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select, func, or_

from codex_agent.memory_chunking import split_chunk_text
from codex_agent.platform.store import timestamp
from codex_agent.storage.database import WorkspaceDatabase, engine
from codex_agent.tasks.store import Conflict, JobStore, digest, encode, now_ms
from codex_agent.tasks.schema import jobs as job_table
from .schema import documents, versions, chunks, requests, audit
from .file_schema import sources, locations
from .files import source_hash, source_row, has_preview
from .parsing import MAX_CHARS, MAX_SECTION, MAX_SECTIONS
from .lifecycle import controls, control, require_active


class SectionEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    id: str = Field(pattern=r"^s[1-9][0-9]*$", max_length=10)
    text: str = Field(max_length=MAX_SECTION)


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(default="", max_length=1000)
    content: str = Field(default="", max_length=60000)
    sections: list[SectionEdit] | None = Field(default=None, min_length=1, max_length=MAX_SECTIONS)
    request_id: str = Field(min_length=1, max_length=100)
    document_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    expected_version: int | None = Field(default=None, ge=1)
    catalog_revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_content(self):
        text = self.content if self.sections is None else "".join(s.text for s in self.sections)
        if (not self.title.strip() or not text.strip() or "\0" in text
                or (self.sections is not None and (self.content or not self.document_id or len(text) > MAX_CHARS))
                or len(self.content.encode("utf-8")) > 262144
                or bool(self.document_id) != bool(self.expected_version)):
            raise ValueError("nonempty text and matching version are required")
        return self


class KnowledgeStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.scope = hashlib.sha256(str(self.root).encode()).hexdigest()
        self.engine = engine()

    def rows(self, conn, table, *conditions):
        return [dict(row) for row in conn.execute(select(table).where(
            table.c.workspace_id == self.scope, *conditions)).mappings()]

    def row(self, conn, table, document_id, version=None, *, allow_deleted=False):
        if table is documents and not allow_deleted:
            require_active(self, conn, document_id)
        key = table.c.id if table is documents else table.c.document_id
        conditions = [key == document_id]
        if version is not None:
            conditions.append(table.c.version == version)
        found = self.rows(conn, table, *conditions)
        if not found:
            raise KeyError("Unknown knowledge document")
        return found[0]

    def writer(self):
        # Only gateway/index workers write; account processes may read published shared data.
        from codex_agent.accounts.guard import owner_id
        if owner_id():
            raise PermissionError("Account processes cannot administer shared knowledge")
        db = WorkspaceDatabase(self.root)
        db.register()
        return db

    def log(self, db, conn, actor, action, document_id=None, version=None):
        db.insert(conn, audit, dict(id=secrets.token_hex(16), actor_id=actor, action=action,
                  document_id=document_id, version=version, created_at=timestamp()))

    def summary(self, conn, doc, actor=None):
        v = self.row(conn, versions, doc["id"], doc["latest_version"])
        state, error = v["state"], v["error"]
        if v["job_id"] and state in {"pending", "parsing"}:
            from codex_agent.tasks.schema import jobs
            job = next(iter(self.rows(conn, jobs, jobs.c.id == v["job_id"])), None)
            if job and (job["cancel_requested"] or job["status"] in {"failed", "dead", "cancelled", "needs-reconciliation"}):
                state, error = ("parse_failed", "Parsing was interrupted; re-upload the file.") if state == "parsing" else (
                    "failed", "Indexing failed or interrupted; retry explicitly.")
        lifecycle = control(self, conn, doc["id"])
        origin = source_row(self, conn, doc["id"], v["version"])
        result = dict(id=doc["id"], title=v["title"], source=v["source"],
            latest_version=doc["latest_version"], published_version=doc["published_version"],
            state="deleted" if lifecycle["deleted_at"] else state, chunk_count=v["chunk_count"],
            error=error, updated_at=doc["updated_at"], deleted_at=lifecycle["deleted_at"],
            catalog_revision=lifecycle["revision"], format=origin["format"] if origin else "text",
            filename=origin["filename"] if origin else None)
        if actor is not None:
            from .publication import sync_for_document
            result["sync"] = sync_for_document(self, conn, doc, actor)
        return result

    def list(self, *, offset=0, limit=50):
        return self.catalog(offset=offset, limit=limit)["documents"]

    def catalog(self, *, offset=0, limit=50, query="", status="all", view="active", actor=None):
        if (not 0 <= offset <= 100000 or not 1 <= limit <= 50 or len(query) > 200
                or "\0" in query or status not in {"all", "published", "unpublished"}
                or view not in {"active", "trash"}):
            raise ValueError("Invalid catalog filter")
        source = documents.join(versions, (versions.c.workspace_id == documents.c.workspace_id) &
            (versions.c.document_id == documents.c.id) & (versions.c.version == documents.c.latest_version))
        source = source.outerjoin(controls, (controls.c.workspace_id == documents.c.workspace_id) &
            (controls.c.document_id == documents.c.id))
        conditions = [documents.c.workspace_id == self.scope,
            controls.c.deleted_at.is_not(None) if view == "trash" else controls.c.deleted_at.is_(None)]
        if query.strip():
            conditions.append(or_(func.lower(versions.c.title).contains(query.strip().lower(), autoescape=True),
                                  func.lower(versions.c.source).contains(query.strip().lower(), autoescape=True)))
        if status != "all":
            conditions.append(documents.c.published_version.is_not(None) if status == "published"
                              else documents.c.published_version.is_(None))
        with self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(source).where(*conditions)).scalar_one()
            found = conn.execute(select(documents).select_from(source).where(*conditions)
                .order_by(documents.c.updated_at.desc(), documents.c.id).offset(offset).limit(limit)).mappings()
            return dict(documents=[self.summary(conn, dict(doc), actor) for doc in found], total=total,
                        offset=offset, limit=limit, query=query, status=status, view=view)

    def detail(self, document_id, actor=None):
        with self.engine.connect() as conn:
            doc = self.row(conn, documents, document_id, allow_deleted=True)
            v = self.row(conn, versions, document_id, doc["latest_version"])
            parts = sorted(self.rows(conn, chunks, chunks.c.document_id == document_id,
                           chunks.c.version == v["version"]), key=lambda c: c["position"])
            origin = source_row(self, conn, document_id, v["version"])
            locators = {c["chunk_id"]: c["locator"] for c in self.rows(conn, locations,
                locations.c.document_id == document_id, locations.c.version == v["version"])}
            extra = {} if origin is None else dict(source_file={
                **{k: origin[k] for k in ("filename", "format", "sha256", "parser_version")},
                "reviewed": bool(origin["reviewed"]), "warnings": json.loads(origin["warnings_json"]),
                "preview_available": has_preview(self, origin)},
                sections=json.loads(origin["sections_json"]))
            return {**self.summary(conn, doc, actor), "content": v["content"],
                **extra, "chunks": [{**{k: c[k] for k in ("id", "text", "position")},
                    "locator": locators.get(c["id"], "")} for c in parts],
                "versions": [{k: v[k] for k in ("version", "state", "chunk_count", "error")}
                    for v in sorted(self.rows(conn, versions, versions.c.document_id == document_id),
                                    key=lambda v: -v["version"])]}

    def save(self, value: DocumentInput, actor):
        db = self.writer()
        with db.transaction() as conn:
            db.lock(conn, "knowledge-write")
            return self.save_transaction(value, actor, db, conn)

    def save_transaction(self, value, actor, db, conn, *, operation="save"):
        """Caller holds knowledge-write; used by atomic confirm + enqueue."""
        payload = value.model_dump(exclude={"request_id"})
        fingerprint, request_key = digest(payload), digest([actor, operation, value.request_id])
        parts = split_chunk_text(value.content) if value.sections is None else []
        if len(parts) > 500:
            raise ValueError("Document exceeds 500 preview chunks; split it into smaller documents")
        prior = self.rows(conn, requests, requests.c.id == request_key)
        if prior:
            if prior[0]["fingerprint"] != fingerprint:
                raise Conflict("Request ID reused with different content")
            saved = self.row(conn, documents, prior[0]["document_id"])
            require_active(self, conn, saved["id"], value.catalog_revision)
            if saved["latest_version"] != prior[0]["version"]:
                raise Conflict("Document changed after this request; reload before editing")
            return self.summary(conn, saved)
        identifier = value.document_id or secrets.token_hex(16)
        origin, section_values, content = None, None, value.content
        if value.document_id:
            doc = self.row(conn, documents, identifier)
            require_active(self, conn, identifier, value.catalog_revision)
            if doc["latest_version"] != value.expected_version:
                raise Conflict("Document changed; reload before saving a new version")
            origin = source_row(self, conn, identifier, value.expected_version)
            if origin:
                previous = self.row(conn, versions, identifier, value.expected_version)
                if previous["state"] in {"parsing", "parse_failed", "ocr_required", "ocr_empty"} or value.sections is None:
                    raise Conflict("Review parsed sections before saving this file")
                originals = json.loads(origin["sections_json"])
                if [s.id for s in value.sections] != [s["id"] for s in originals]:
                    raise Conflict("Section identities changed; reload the document")
                section_values = [{**old, "text": new.text,
                    "edited": old["edited"] or old["text"] != new.text}
                    for old, new in zip(originals, value.sections)]
                content = "\n\n".join(s["text"] for s in section_values)
            elif value.sections is not None:
                raise Conflict("This document has no file sections")
            from .publication import invalidate
            invalidate(self, db, conn, identifier, "superseded")
            version = doc["latest_version"] + 1
            db.update(conn, documents, dict(latest_version=version, updated_at=timestamp()),
                      documents.c.id == identifier)
        else:
            version = 1
            db.insert(conn, documents, dict(id=identifier, latest_version=1, published_version=None,
                created_at=timestamp(), updated_at=timestamp()))
        db.insert(conn, versions, dict(document_id=identifier, version=version, title=value.title,
            source=value.source, content=content, source_hash=source_hash(value.title, value.source, content, section_values),
            state="draft", definition_json="{}", fingerprint=None, job_id=None, chunk_count=len(parts),
            error=None, created_at=timestamp()))
        if origin:
            db.insert(conn, sources, {**{k: origin[k] for k in ("file_id", "filename", "format", "sha256", "parser_version", "warnings_json")},
                "document_id": identifier, "version": version, "reviewed": 1, "sections_json": encode(section_values)})
        for position, part in enumerate(parts):
            db.insert(conn, chunks, dict(document_id=identifier, version=version,
                id=digest([identifier, version, position, part]), position=position, text=part))
        db.insert(conn, requests, dict(id=request_key, fingerprint=fingerprint,
                                     document_id=identifier, version=version))
        self.log(db, conn, actor, "document-saved", identifier, version)
        return self.summary(conn, self.row(conn, documents, identifier))


    def enqueue(self, identifier, version, request_id, actor, catalog_revision=0):
        from .vectors import index_definition
        definition = index_definition()
        db = self.writer()
        with db.transaction() as conn:
            db.lock(conn, "knowledge-write")
            return self.enqueue_transaction(identifier, version, request_id, actor,
                catalog_revision, definition, db, conn)

    def enqueue_transaction(self, identifier, version, request_id, actor, catalog_revision,
                            definition, db, conn, *, publication_id=None):
        """Persist index job/outbox in caller's transaction, without network work."""
        fingerprint = digest(definition)
        jobs = JobStore(self.root, database=db)
        doc = self.row(conn, documents, identifier)
        require_active(self, conn, identifier, catalog_revision)
        v = self.row(conn, versions, identifier, version)
        origin = source_row(self, conn, identifier, version)
        if origin and not origin["reviewed"]:
            raise Conflict("Save the reviewed parsed sections before indexing")
        payload = dict(domain="knowledge", document_id=identifier, version=version,
                       source_hash=v["source_hash"], fingerprint=fingerprint,
                       catalog_revision=require_active(self, conn, identifier))
        if publication_id is not None:
            payload["publication_id"] = publication_id
        key = "knowledge:" + digest([actor, identifier, version, request_id])
        previous = jobs.find_key(conn, key)
        if previous:
            if json.loads(previous["payload_json"]) != payload:
                raise Conflict("Index request changed; use a new request ID")
            return self.summary(conn, doc)
        if doc["latest_version"] != version or doc["published_version"] == version:
            raise Conflict("Index only the latest unpublished version")
        if v["state"] in {"ready", "stale"}:
            # A timed-out cleanup request can still arrive at the vector
            # server later. Never reuse its exact document/version target.
            raise Conflict("Ready or cleaned versions are immutable; save a new version to reindex")
        if v["job_id"]:
            old = jobs._row(conn, job_table, v["job_id"])
            if old["status"] not in {"succeeded", "failed", "dead", "cancelled"}:
                raise Conflict("Index task is already active or requires reconciliation")
        job = jobs.submit("index", payload, key, connection=conn)
        db.update(conn, versions, dict(state="pending", definition_json=encode(definition),
            fingerprint=fingerprint, job_id=job["id"], error=None),
            versions.c.document_id == identifier, versions.c.version == version)
        self.log(db, conn, actor, "index-requested", identifier, version)
        return self.summary(conn, doc)


    def set_deleted(self, identifier, version, revision, actor, *, deleted):
        if type(version) is not int or version < 1 or type(revision) is not int or revision < 0:
            raise ValueError("Expected document version and lifecycle revision are required")
        db = self.writer()
        with db.transaction() as conn:
            db.lock(conn, "knowledge-write")
            doc = self.row(conn, documents, identifier, allow_deleted=True)
            current = control(self, conn, identifier)
            if doc["latest_version"] != version:
                raise Conflict("Document changed; reload before deleting or restoring")
            # Retrying an already committed transition is safe; old requests after
            # a restore/delete cycle must not act on the new lifecycle.
            if bool(current["deleted_at"]) == deleted and current["revision"] == revision + 1:
                return self.summary(conn, doc)
            if current["revision"] != revision or bool(current["deleted_at"]) == deleted:
                raise Conflict("Document lifecycle changed; reload before continuing")
            values = dict(document_id=identifier, revision=revision + 1,
                          deleted_at=timestamp() if deleted else None, actor_id=actor)
            if current["revision"] == 0:
                db.insert(conn, controls, values)
            else:
                db.update(conn, controls, values, controls.c.document_id == identifier)
            # Restore is intentionally unpublished. Nothing can silently resume
            # serving an old source or re-authorize an interrupted index worker.
            db.update(conn, documents, dict(published_version=None, updated_at=timestamp()), documents.c.id == identifier)
            from .publication import invalidate
            invalidate(self, db, conn, identifier, "withdrawn" if deleted else "superseded")
            if deleted:
                ids = [v["job_id"] for v in self.rows(conn, versions, versions.c.document_id == identifier) if v["job_id"]]
                if ids:
                    from sqlalchemy import case
                    db.update(conn, job_table, dict(cancel_requested=1, updated_ms=now_ms(conn),
                        status=case((job_table.c.status == "queued", "cancelled"), else_=job_table.c.status)),
                        job_table.c.id.in_(ids), job_table.c.status.in_(["queued", "running"]))
                from .cleanup import enqueue_cleanup
                enqueue_cleanup(self, db, conn, identifier, revision + 1, actor)
            self.log(db, conn, actor, "document-trashed" if deleted else "document-restored", identifier, version)
            return self.summary(conn, self.row(conn, documents, identifier, allow_deleted=True))

    def publish(self, identifier, version, actor, catalog_revision=0):
        db = self.writer()
        with db.transaction() as conn:
            db.lock(conn, "knowledge-write")
            doc = self.row(conn, documents, identifier)
            require_active(self, conn, identifier, catalog_revision)
            if version is not None:
                v = self.row(conn, versions, identifier, version)
                if v["state"] != "ready" or v["chunk_count"] < 1:
                    raise Conflict("Only completely indexed versions can be published")
                from .vectors import index_definition
                if v["fingerprint"] != digest(index_definition()):
                    raise Conflict("Embedding configuration changed; create and index a new version")
            from .publication import invalidate
            invalidate(self, db, conn, identifier, "withdrawn")
            db.update(conn, documents, dict(published_version=version, updated_at=timestamp()),
                      documents.c.id == identifier)
            self.log(db, conn, actor, "published" if version else "unpublished", identifier, version)
            return self.summary(conn, self.row(conn, documents, identifier))

    def configure_publishers(self, admin_user_ids, *, enabled=True):
        from .publication_policy import configure_publishers
        return configure_publishers(self, admin_user_ids, enabled=enabled)

    def confirm(self, value, actor, *, publisher_ids=None):
        from .publication import confirm
        return confirm(self, value, actor, publisher_ids=publisher_ids)

    def operation(self, request_id, actor):
        from .publication import operation
        return operation(self, request_id, actor)

    def published_document(self, identifier, version):
        with self.engine.connect() as conn:
            doc = self.row(conn, documents, identifier)
            if doc["published_version"] != version:
                raise KeyError("Document version is not published")
            v = self.row(conn, versions, identifier, version)
            return {k: v[k] for k in ("title", "source", "content", "version")}
