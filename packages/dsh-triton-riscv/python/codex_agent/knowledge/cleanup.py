"""Version-scoped vector cleanup; original text and immutable versions are retained.

Deletion is already enforced by SQL publication/lifecycle checks. Cleanup is a
separate, idempotent index job, one version per attempt. A durable stale marker
is committed before network effects so restoring a document cannot republish a
version whose vector deletion had an uncertain result.
"""
import json

from sqlalchemy import select

from codex_agent.platform.store import timestamp
from codex_agent.storage.schema import meta
from codex_agent.tasks.schema import jobs as job_table
from codex_agent.tasks.store import Conflict, JobStore, digest, encode
from .lifecycle import DocumentUnavailable, control
from .schema import documents, versions
from .store import KnowledgeStore
from .vectors import vector_index


def _key(identifier):
    return "knowledge-cleanup:" + identifier


def _state(store, conn, identifier):
    rows = store.rows(conn, meta, meta.c.key == _key(identifier))
    return json.loads(rows[0]["value"]) if rows else {}


def _save(store, db, conn, identifier, value):
    value = {**value, "updated_at": timestamp()}
    if store.rows(conn, meta, meta.c.key == _key(identifier)):
        db.update(conn, meta, {"value": encode(value)}, meta.c.key == _key(identifier))
    else:
        db.insert(conn, meta, {"key": _key(identifier), "value": encode(value)})
    return value


def cleanup_status(store, conn, identifier):
    """Read a safe receipt, interpreting queue failures without exposing payloads."""
    value = _state(store, conn, identifier)
    if not value:
        return {"state": "not_requested", "verified_at": None}
    result = {key: value.get(key) for key in
              ("state", "revision", "job_id", "updated_at", "verified_at")}
    if value.get("job_id") and value.get("state") not in {"complete", "cancelled"}:
        job = next(iter(store.rows(conn, job_table, job_table.c.id == value["job_id"])), None)
        if not job:
            result["state"] = "unknown"
        elif job["status"] in {"dead", "failed", "needs-reconciliation"}:
            result["state"] = "failed"
        elif job["status"] == "cancelled":
            result["state"] = "cancelled"
        elif job["status"] == "running":
            result["state"] = "running"
    return result


def enqueue_cleanup(store, db, conn, identifier, revision, actor, request_id=None):
    """Caller holds knowledge-write; the job and receipt commit with soft delete."""
    current = control(store, conn, identifier)
    if not current["deleted_at"] or current["revision"] != revision:
        raise Conflict("Cleanup requires the current deleted document revision")
    existing = _state(store, conn, identifier)
    if existing.get("revision") == revision:
        if existing.get("state") == "complete":
            return cleanup_status(store, conn, identifier)
        prior = next(iter(store.rows(conn, job_table, job_table.c.id == existing.get("job_id"))), None)
        if prior and prior["status"] in {"queued", "running"}:
            return cleanup_status(store, conn, identifier)
    payload = dict(domain="knowledge-cleanup", document_id=identifier,
                   catalog_revision=revision, after_version=0, actor_id=actor)
    jobs = JobStore(store.root, database=db)
    job = jobs.submit("index", payload,
        "knowledge-cleanup:" + digest([identifier, revision, request_id or "initial"]), connection=conn)
    value = dict(state="queued", revision=revision, job_id=job["id"], verified_at=None)
    return _save(store, db, conn, identifier, value)


def _require_deleted(store, conn, identifier, revision):
    doc = store.row(conn, documents, identifier, allow_deleted=True)
    current = control(store, conn, identifier)
    if (not current["deleted_at"] or current["revision"] != revision
            or doc["published_version"] is not None):
        raise DocumentUnavailable("Cleanup lifecycle was superseded")
    return doc


def _next_version(store, conn, identifier, after):
    row = conn.execute(select(versions).where(
        versions.c.workspace_id == store.scope, versions.c.document_id == identifier,
        versions.c.version > after, versions.c.fingerprint.is_not(None))
        .order_by(versions.c.version).limit(1)).mappings().first()
    return dict(row) if row else None


def _require_quiescent(store, conn, identifier):
    """Do not race an earlier parse/index worker still known to be running."""
    pending = conn.execute(select(job_table.c.id).join(versions,
        (versions.c.workspace_id == job_table.c.workspace_id) &
        (versions.c.job_id == job_table.c.id)).where(
        versions.c.workspace_id == store.scope, versions.c.document_id == identifier,
        job_table.c.status.in_(["queued", "running", "needs-reconciliation"])).limit(1)).first()
    if pending:
        raise Conflict("An earlier document worker must stop before cleanup")


def _delete_and_verify(index, identifier, version):
    """The same exact scope filter is used for deletion and count readback."""
    index.transport.timeout = min(index.transport.timeout, 5)
    index.delete_version(identifier, version)
    query = {"filter": {"must": [{"key": key, "match": {"value": value}} for key, value in (
        ("workspace_id", index.scope.workspace_id), ("document_id", identifier),
        ("version", version), ("generation", index.scope.generation),
        ("index_fingerprint", index.scope.index_fingerprint))]}, "exact": True}
    result = index.transport.request("POST", "/points/count", query, missing_ok=True)
    # A missing collection is already empty, never create it for cleanup.
    if result is not None and (not isinstance(result, dict) or type(result.get("count")) is not int
                               or result["count"] != 0):
        raise Conflict("Vector cleanup has not been confirmed empty")


def execute_cleanup(jobs, job, owner, *, index_factory=vector_index):
    store, db = KnowledgeStore(jobs.db.root), jobs.db
    with db.transaction() as conn:
        db.lock(conn, "knowledge-write")
        current = jobs.require_owner(conn, job["id"], owner)
        payload = json.loads(current["payload_json"])
        if (current["kind"] != "index" or payload.get("domain") != "knowledge-cleanup"
                or type(payload.get("catalog_revision")) is not int
                or type(payload.get("after_version")) is not int or payload["after_version"] < 0):
            raise ValueError("Invalid cleanup task")
        identifier, revision = payload["document_id"], payload["catalog_revision"]
        _require_deleted(store, conn, identifier, revision)
        _require_quiescent(store, conn, identifier)
        source = _next_version(store, conn, identifier, payload["after_version"])
        if source is None:
            _save(store, db, conn, identifier, dict(state="complete", revision=revision,
                job_id=job["id"], verified_at=timestamp()))
            jobs.finish(job["id"], owner, "succeeded", {"cleanup_state": "complete"}, connection=conn)
            return
        definition = json.loads(source["definition_json"])
        if not definition or digest(definition) != source["fingerprint"]:
            raise ValueError("Stored index definition does not match source version")
        # Must commit BEFORE deleting vectors. A timeout, crash, or later restore
        # cannot leave this version ready for immediate publication.
        db.update(conn, versions, {"state": "stale", "error": "Index cleanup requested; save a new version and reindex before publication"},
            versions.c.document_id == identifier, versions.c.version == source["version"])
        _save(store, db, conn, identifier, dict(state="running", revision=revision,
            job_id=job["id"], verified_at=None))

    index = index_factory(store.scope, definition)
    jobs.begin_effects(job["id"], owner)
    with db.transaction() as conn:
        # Serialize restore/reindex admission against this bounded remote effect.
        # There is at most one version and two five-second requests per job.
        db.lock(conn, "knowledge-write")
        jobs.require_owner(conn, job["id"], owner)
        _require_deleted(store, conn, identifier, revision)
        _require_quiescent(store, conn, identifier)
        _delete_and_verify(index, identifier, source["version"])
        jobs.require_owner(conn, job["id"], owner)
        _require_deleted(store, conn, identifier, revision)
        next_source = _next_version(store, conn, identifier, source["version"])
        if next_source:
            next_payload = {**payload, "after_version": source["version"]}
            successor = jobs.submit("index", next_payload,
                "knowledge-cleanup:" + digest([identifier, revision, source["version"], job["id"]]),
                connection=conn)
            value = dict(state="queued", revision=revision, job_id=successor["id"], verified_at=None)
        else:
            value = dict(state="complete", revision=revision, job_id=job["id"], verified_at=timestamp())
        _save(store, db, conn, identifier, value)
        jobs.finish(job["id"], owner, "succeeded", dict(cleanup_state=value["state"],
            version=source["version"], verified_at=value["verified_at"]), connection=conn)
