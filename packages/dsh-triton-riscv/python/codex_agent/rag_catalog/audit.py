"""Resumable, read-only historical memory reconciliation.

Only explicit server-local manifests are accepted. This module never creates a
source engine schema, index, collection, embedding, job, or account grant.
Checkpoints contain identities/digests/counters, never document text or secrets.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from sqlalchemy import and_, func, inspect, or_, select
from sqlalchemy.exc import SQLAlchemyError

from codex_agent.memory_lineage import CASE_FIELDS, case_identity, digest
from codex_agent.storage.schema import (memories, memory_versions, chunks,
    memory_indexes as indexes, memory_index_entries as entries,
    memory_index_chunks as indexed_chunks)
from .sources import history_engine, historical_scopes

PAYLOAD_FIELDS = ["workspace_id", "generation", "memory_id", "revision",
                  "case_version", "chunk_id", "index_fingerprint"]
MAX_PAGE_SIZE = 100
SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_FINDINGS = 10000


class AuditBlocked(RuntimeError):
    """Safe, stable reason code only: never propagate driver response bodies."""


class AuditDeadline(RuntimeError):
    pass


def _check(deadline):
    if time.monotonic() >= deadline:
        raise AuditDeadline()


def load_manifest(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 128 * 1024:
        raise AuditBlocked("invalid_manifest_file")
    return validate_manifest(json.loads(path.read_text()))


def validate_manifest(value):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise AuditBlocked("invalid_manifest_schema")
    if set(value) - {"schema_version", "sources", "vector_services"}:
        raise AuditBlocked("unsupported_manifest_field")
    services = value.get("vector_services", [])
    if not isinstance(services, list) or len(services) > 16:
        raise AuditBlocked("invalid_vector_service_grants")
    for service in services:
        if not isinstance(service, dict) or set(service) != {"url", "api_key_env"}:
            raise AuditBlocked("invalid_vector_service_grant")
        url = service["url"]
        if not isinstance(url, str) or url != url.strip() or any(ord(c) < 33 for c in url):
            raise AuditBlocked("invalid_vector_service_url")
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise AuditBlocked("invalid_vector_service_url")
        if not isinstance(service["api_key_env"], str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", service["api_key_env"]):
            raise AuditBlocked("invalid_vector_key_reference")
    sources = value.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 32:
        raise AuditBlocked("invalid_source_count")
    aliases = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) - {"alias", "url_env", "owner_user_id", "scopes"}:
            raise AuditBlocked("invalid_source_manifest")
        alias = source.get("alias", "")
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias) or alias in aliases:
            raise AuditBlocked("invalid_source_alias")
        aliases.add(alias)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", source.get("url_env", "")):
            raise AuditBlocked("connection_requires_environment_reference")
        if source.get("owner_user_id") is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source["owner_user_id"]):
            raise AuditBlocked("invalid_owner_grant")
        scopes = source.get("scopes")
        if not isinstance(scopes, list) or not 1 <= len(scopes) <= 128:
            raise AuditBlocked("invalid_scope_count")
        seen = set()
        for scope in scopes:
            if not isinstance(scope, dict) or set(scope) != {"id", "root", "kind"}:
                raise AuditBlocked("invalid_scope_manifest")
            root = Path(scope["root"])
            if (scope["kind"] != "memory_case" or not root.is_absolute()
                    or str(root.resolve()) != scope["root"]
                    or hashlib.sha256(scope["root"].encode()).hexdigest() != scope["id"]
                    or scope["id"] in seen):
                raise AuditBlocked("invalid_scope_identity")
            seen.add(scope["id"])
    return copy.deepcopy(value)


def _after(columns, cursor):
    if cursor is None:
        return []
    return [or_(*[and_(*[columns[j] == cursor[j] for j in range(i)], col > cursor[i])
                   for i, col in enumerate(columns)])]


def _json(raw):
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return None


def _case_version(row):
    result = {k: row.get(k) for k in CASE_FIELDS}
    for field in ("tl_ops", "environment", "evidence"):
        raw = row.get(field + "_json")
        value = _json(raw)
        if value is None:
            return None
        result[field] = value if raw else ([] if field == "tl_ops" else {})
    return digest(result)


def _text_ok(row):
    provenance = _json(row.get("provenance_json"))
    return (isinstance(provenance, dict)
        and provenance.get("text_sha256") == hashlib.sha256((row.get("text") or "").encode()).hexdigest()
        and provenance.get("case_id") == case_identity(row["workspace_id"], row["memory_id"]))


class SQLReadSource:
    """No Database/MemoryStore construction: those may initialize writable state."""
    def __init__(self, source, *, engine=None):
        self.source = source
        raw = os.environ.get(source["url_env"], "")
        if engine is None and not raw:
            raise AuditBlocked("source_environment_missing")
        self.engine = engine or history_engine(raw, os.getpid())
        # Bind the database identity without storing the URL or credentials.
        self.connection_fingerprint = hashlib.sha256((raw or str(self.engine.url)).encode()).hexdigest()

    def _rows(self, statement):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(statement).mappings()]

    def grants(self):
        cfg = SimpleNamespace(workspace_ids=[s["id"] for s in self.source["scopes"]],
            owner_user_id=self.source.get("owner_user_id"), label=self.source["alias"])
        actual = historical_scopes(cfg, self.engine)
        expected = {(s["id"], s["root"]) for s in self.source["scopes"]}
        if {(s["id"], str(s["root"])) for s in actual} != expected:
            raise AuditBlocked("scope_grant_changed")
        return digest(sorted(expected))

    def binding(self, deadline):
        grant = self.grants()
        required = [memories, memory_versions, chunks, indexes, entries, indexed_chunks]
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            for table in required:
                if not inspector.has_table(table.name):
                    raise AuditBlocked("source_schema_not_supported")
                if set(c.name for c in table.columns) - {c["name"] for c in inspector.get_columns(table.name)}:
                    raise AuditBlocked("source_schema_not_supported")
        fingerprints, definitions = {}, []
        # Bounded keyset reads. Hash source text in memory, never write it to disk.
        # This validation is outside the work-page budget but inside the deadline.
        # It also detects edits made without correctly updating case_version.
        for scope in self.source["scopes"]:
            for table in required:
                columns = [c for c in table.columns if c.name not in {
                    "embedding_json", "embedding_provider", "embedding_model", "embedding_dim",
                    "last_used_at", "useful_count"}]
                keys = [c for c in table.primary_key if c.name != "workspace_id"]
                cursor, h, count = None, hashlib.sha256(), 0
                while True:
                    _check(deadline)
                    rows = self._rows(select(*columns).where(table.c.workspace_id == scope["id"],
                        *_after(keys, cursor)).order_by(*keys).limit(MAX_PAGE_SIZE))
                    for row in rows:
                        h.update(digest(row).encode())
                        count += 1
                        if table is indexes:
                            definition = _json(row["config_json"])
                            if not isinstance(definition, dict) or digest(definition) != row["fingerprint"]:
                                raise AuditBlocked("index_definition_fingerprint_mismatch")
                            definitions.append({**row, "definition": definition})
                    if len(rows) < MAX_PAGE_SIZE:
                        break
                    cursor = [rows[-1][c.name] for c in keys]
                fingerprints[scope["id"] + ":" + table.name] = dict(digest=h.hexdigest(), count=count)
        return {"grant": grant, "connection": self.connection_fingerprint,
                "source_versions": fingerprints}, definitions

    def page(self, scope, phase, cursor, limit):
        table, names = {
            "sources": (memories, ["id"]),
            "entries": (entries, ["generation", "memory_id"]),
            "chunks": (indexed_chunks, ["generation", "memory_id", "revision", "chunk_id"]),
        }[phase]
        keys = [table.c[name] for name in names]
        columns = [c for c in table.columns if c.name not in {
            "embedding_json", "embedding_provider", "embedding_model", "embedding_dim"}]
        rows = self._rows(select(*columns).where(table.c.workspace_id == scope,
            *_after(keys, cursor)).order_by(*keys).limit(limit))
        next_cursor = [rows[-1][name] for name in names] if len(rows) == limit else None
        return rows, next_cursor

    def parent(self, scope, identifier):
        columns = [c for c in memories.columns if not c.name.startswith("embedding")]
        rows = self._rows(select(*columns).where(memories.c.workspace_id == scope, memories.c.id == identifier))
        return rows[0] if rows else None

    def snapshot(self, scope, identifier, version):
        rows = self._rows(select(memory_versions.c.payload_json).where(
            memory_versions.c.workspace_id == scope, memory_versions.c.memory_id == identifier,
            memory_versions.c.case_version == version))
        if not rows:
            return "missing"
        payload = _json(rows[0]["payload_json"])
        return "valid" if isinstance(payload, dict) and digest(payload) == version else "hash_mismatch"

    def entry(self, scope, generation, identifier):
        rows = self._rows(select(entries).where(entries.c.workspace_id == scope,
            entries.c.generation == generation, entries.c.memory_id == identifier))
        return rows[0] if rows else None

    def chunk_count(self, scope, entry):
        return self._rows(select(func.count().label("n")).select_from(indexed_chunks).where(
            indexed_chunks.c.workspace_id == scope, indexed_chunks.c.generation == entry["generation"],
            indexed_chunks.c.memory_id == entry["memory_id"], indexed_chunks.c.revision == entry["revision"],
            indexed_chunks.c.case_version == entry["case_version"]))[0]["n"]

    def vector_chunk(self, scope, generation, point_id):
        return self._rows(select(indexed_chunks).where(indexed_chunks.c.workspace_id == scope,
            indexed_chunks.c.generation == generation, indexed_chunks.c.point_id == point_id).limit(2))


def _authorize_vector(index, allowed_services):
    # SQL content and its self-declared fingerprint are not network/secret grants.
    # Authorize the exact endpoint + key-env pair BEFORE transport construction,
    # since the shared constructor reads the named environment variable.
    definition = index.get("definition", {})
    url, key_env = definition.get("qdrantUrl"), definition.get("apiKeyEnv")
    if not isinstance(url, str) or not any(url.rstrip("/") == service["url"].rstrip("/")
            and key_env == service["api_key_env"] for service in allowed_services):
        raise AuditBlocked("vector_endpoint_or_key_not_granted")


class ReadOnlyVector:
    def __init__(self, index, factory=None, *, allowed_services=()):
        _authorize_vector(index, allowed_services)
        if factory is None:
            from codex_agent.indexing.qdrant import Qdrant
            factory = Qdrant
        self.index = index
        self.transport = factory(index["definition"], index["collection"])

    def _request(self, method, path, value, deadline):
        _check(deadline)
        # Existing transport, no ensure/upsert/delete/embed. Per-call timeout cannot
        # exceed this invocation's remaining wall-clock budget.
        self.transport.timeout = max(.001, min(float(self.transport.timeout), deadline - time.monotonic()))
        result = self.transport.request(method, path, value)
        _check(deadline)
        return result

    def points(self, ids, deadline):
        result = self._request("POST", "/points", {"ids": ids,
            "with_vector": False, "with_payload": PAYLOAD_FIELDS}, deadline)
        if not isinstance(result, list) or len(result) > len(ids):
            raise AuditBlocked("invalid_vector_point_response")
        return result

    def page(self, scope, cursor, limit, deadline):
        result = self._request("POST", "/points/scroll", {
            "limit": limit, "offset": cursor, "with_vector": False,
            "with_payload": PAYLOAD_FIELDS, "filter": {"must": [
                {"key": "workspace_id", "match": {"value": scope}},
                {"key": "generation", "match": {"value": self.index["generation"]}}]}}, deadline)
        if not isinstance(result, dict) or not isinstance(result.get("points"), list) or len(result["points"]) > limit:
            raise AuditBlocked("invalid_vector_scroll_response")
        next_cursor = result.get("next_page_offset")
        if next_cursor is not None and (not result["points"] or next_cursor == cursor):
            raise AuditBlocked("vector_cursor_did_not_advance")
        return result["points"], next_cursor


def _write(path, state):
    encoded = json.dumps(state, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode()
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise AuditBlocked("checkpoint_size_budget_exhausted")
    fd, temporary = tempfile.mkstemp(prefix=".audit-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            os.fchmod(out.fileno(), 0o600)
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)



def _hex(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _point_id(value):
    return ((type(value) is int and 0 <= value < 2**64) or
        (isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value) is not None))


def _tasks(manifest, definitions):
    result = []
    for source in manifest["sources"]:
        for scope in source["scopes"]:
            base = dict(alias=source["alias"], scope=scope["id"], cursor=None)
            result.extend([{**base, "phase": phase} for phase in ("sources", "entries", "chunks")])
            for (alias, sid, generation), index in sorted(definitions.items()):
                if alias == source["alias"] and sid == scope["id"]:
                    result.append({**base, "phase": "vector_points", "generation": generation,
                        "collection": index["collection"], "index_fingerprint": index["fingerprint"]})
    return result


def _validate_checkpoint(state, expected_tasks):
    """Local checkpoints are resumable evidence, never new scope authority."""
    if (state.get("schema_version") != SCHEMA_VERSION or state.get("read_only") is not True
            or not isinstance(state.get("tasks"), list)
            or len(state["tasks"]) != len(expected_tasks)
            or type(state.get("task_index")) is not int
            or not 0 <= state["task_index"] <= len(expected_tasks)
            or not isinstance(state.get("findings"), dict) or len(state["findings"]) > MAX_FINDINGS):
        raise AuditBlocked("invalid_checkpoint")
    allowed = {(t["alias"], t["scope"]) for t in expected_tasks}
    for task, expected in zip(state["tasks"], expected_tasks):
        if not isinstance(task, dict) or {k:v for k,v in task.items() if k != "cursor"} != {k:v for k,v in expected.items() if k != "cursor"}:
            raise AuditBlocked("checkpoint_task_not_granted")
        cursor, phase = task.get("cursor"), task["phase"]
        if cursor is None:
            continue
        if phase == "vector_points":
            valid = _point_id(cursor)
        elif not isinstance(cursor, list):
            valid = False
        elif phase == "sources":
            valid = len(cursor) == 1 and type(cursor[0]) is int and cursor[0] >= 0
        elif phase == "entries":
            valid = len(cursor) == 2 and isinstance(cursor[0], str) and len(cursor[0]) <= 64 and type(cursor[1]) is int and cursor[1] > 0
        else:
            valid = (len(cursor) == 4 and isinstance(cursor[0], str) and len(cursor[0]) <= 64
                and type(cursor[1]) is int and cursor[1] > 0 and type(cursor[2]) is int
                and cursor[2] > 0 and _hex(cursor[3]))
        if not valid:
            raise AuditBlocked("invalid_checkpoint_cursor")
    for key, finding in state["findings"].items():
        if (not isinstance(finding, dict) or set(finding) != {"source_alias", "scope_id", "source_id", "version", "code", "occurrences", "examples"}
                or (finding["source_alias"], finding["scope_id"]) not in allowed
                or not isinstance(finding["source_id"], str) or not re.fullmatch(r"[0-9]{1,20}|unknown", finding["source_id"])
                or (finding["version"] is not None and not _hex(finding["version"]))
                or not isinstance(finding["code"], str) or not re.fullmatch(r"[a-z_]{1,100}", finding["code"])
                or type(finding["occurrences"]) is not int or finding["occurrences"] < 1
                or not isinstance(finding["examples"], list) or len(finding["examples"]) > 5
                or not all(_point_id(x) or (isinstance(x, str) and x.isdigit() and len(x) <= 20) for x in finding["examples"])):
            raise AuditBlocked("invalid_checkpoint_finding")
        if key != digest([finding["source_alias"], finding["scope_id"], finding["source_id"], finding["version"] or "", finding["code"]]):
            raise AuditBlocked("invalid_checkpoint_finding_key")


def _finding(state, alias, scope, identifier, version, code, *, point_id=None):
    version = version if _hex(version) else None
    identifier = identifier if type(identifier) is int and 0 <= identifier < 2**63 else "unknown"
    identity = [alias, scope, str(identifier), version or "", code]
    key = digest(identity)
    findings = state["findings"]
    if key not in findings and len(findings) >= MAX_FINDINGS:
        raise AuditBlocked("finding_budget_exhausted_review_before_new_audit")
    row = findings.setdefault(key, dict(source_alias=alias, scope_id=scope,
        source_id=str(identifier), version=version, code=code, occurrences=0, examples=[]))
    row["occurrences"] += 1
    if _point_id(point_id) and str(point_id) not in row["examples"] and len(row["examples"]) < 5:
        row["examples"].append(str(point_id))


def _check_source(reader, scope, row, emit):
    identifier, version = row["id"], row.get("case_version")
    if not (row.get("searchable_text") or "").strip():
        emit(identifier, version, "source_text_empty")
    if not version:
        emit(identifier, version, "source_version_missing")
    elif _case_version(row) != version:
        emit(identifier, version, "source_version_hash_mismatch")
    if version:
        status = reader.snapshot(scope, identifier, version)
        if status != "valid":
            emit(identifier, version, "current_snapshot_" + status)


def _check_entry(reader, scope, row, emit):
    identifier, version = row["memory_id"], row.get("case_version")
    parent = reader.parent(scope, identifier)
    if parent is None:
        emit(identifier, version, "entry_source_missing")
    elif version != parent.get("case_version"):
        emit(identifier, version, "entry_version_not_current")
    if version:
        snapshot = reader.snapshot(scope, identifier, version)
        if snapshot != "valid":
            emit(identifier, version, "entry_snapshot_" + snapshot)
    if row["status"] == "ready" and row["desired"] == "present":
        if not parent or not parent["active"]:
            emit(identifier, version, "ready_entry_source_inactive_or_missing")
        if reader.chunk_count(scope, row) != row["point_count"]:
            emit(identifier, version, "entry_chunk_count_mismatch")


def _check_chunk(reader, scope, row, emit):
    identifier, version = row["memory_id"], row["case_version"]
    parent = reader.parent(scope, identifier)
    if parent is None:
        emit(identifier, version, "chunk_source_missing", point_id=row["point_id"])
    status = reader.snapshot(scope, identifier, version)
    if status != "valid":
        emit(identifier, version, "chunk_snapshot_" + status, point_id=row["point_id"])
    if not _text_ok(row):
        emit(identifier, version, "chunk_hash_mismatch", point_id=row["point_id"])
    entry = reader.entry(scope, row["generation"], identifier)
    current = entry and all(entry[k] == row[k] for k in ("revision", "case_version"))
    return bool(current and entry["desired"] == "present" and entry["status"] == "ready"
                and parent and parent["active"] and parent.get("case_version") == version)


def _check_point(reader, scope, index, point, emit):
    payload = point.get("payload")
    if not isinstance(payload, dict):
        emit("unknown", None, "point_identity_invalid", point_id=point.get("id"))
        return
    identifier, version = payload.get("memory_id"), payload.get("case_version")
    if (type(identifier) is not int or identifier <= 0 or type(payload.get("revision")) is not int
            or not _hex(version) or not _hex(payload.get("chunk_id"))
            or payload["revision"] <= 0 or not _point_id(point.get("id"))):
        emit("unknown", None, "point_identity_invalid", point_id=point.get("id"))
        return
    if any(payload.get(k) != value for k, value in {
            "workspace_id": scope, "generation": index["generation"], "index_fingerprint": index["fingerprint"]}.items()):
        emit(identifier, version, "point_index_identity_mismatch", point_id=point.get("id"))
    parent = reader.parent(scope, identifier)
    if parent is None:
        emit(identifier, version, "point_source_missing", point_id=point.get("id"))
    elif parent.get("case_version") != version:
        emit(identifier, version, "point_version_not_current", point_id=point.get("id"))
    # A parent existing does not prove that this historical version exists.
    snapshot = reader.snapshot(scope, identifier, version)
    if snapshot != "valid":
        emit(identifier, version, "point_snapshot_" + snapshot, point_id=point.get("id"))
    rows = reader.vector_chunk(scope, index["generation"], point.get("id"))
    if len(rows) != 1:
        emit(identifier, version, "point_chunk_missing_or_ambiguous", point_id=point.get("id"))
    else:
        row = rows[0]
        if any(row[k] != payload.get(k) for k in ("memory_id", "revision", "case_version", "chunk_id")):
            emit(identifier, version, "point_chunk_identity_mismatch", point_id=point.get("id"))
        if not _text_ok(row):
            emit(identifier, version, "point_chunk_hash_mismatch", point_id=point.get("id"))
    entry = reader.entry(scope, index["generation"], identifier)
    if (not entry or entry["desired"] != "present" or entry["status"] != "ready"
            or any(entry[k] != payload.get(k) for k in ("revision", "case_version"))
            or not parent or not parent["active"]):
        emit(identifier, version, "point_not_current_ready_entry", point_id=point.get("id"))


def run_audit(manifest, audit_dir, *, resume=False, max_pages=20, deadline_seconds=60,
              page_size=100, source_factory=SQLReadSource, vector_factory=None):
    """Audit explicit grants, persist after every completed page, and stop safely.

    max_pages counts work pages; fingerprint validation also performs <=100-row
    keyset reads and shares the deadline. SQL transport has 10s socket timeouts;
    an in-flight SQL request can therefore overrun the deadline by at most its
    configured transport timeout. No retry loop is implicit. Use an unchanged
    manifest and --resume to retry a blocked/partial checkpoint.
    """
    manifest = validate_manifest(manifest)
    if (type(max_pages) is not int or not 1 <= max_pages <= 100000
            or type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE
            or not isinstance(deadline_seconds, (int, float)) or not 0 < deadline_seconds <= 3600):
        raise AuditBlocked("invalid_audit_budget")
    directory = Path(audit_dir)
    if directory.is_symlink():
        raise AuditBlocked("audit_directory_symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    checkpoint = directory / "checkpoint.json"
    if checkpoint.is_symlink():
        raise AuditBlocked("checkpoint_symlink")
    if (directory / "audit.lock").is_symlink():
        raise AuditBlocked("audit_lock_symlink")
    with (directory / "audit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AuditBlocked("audit_already_running") from None
        if checkpoint.exists():
            if not resume:
                raise AuditBlocked("checkpoint_exists_use_resume_or_new_directory")
            if checkpoint.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise AuditBlocked("checkpoint_size_budget_exhausted")
            state = json.loads(checkpoint.read_text())
        elif resume:
            raise AuditBlocked("checkpoint_missing")
        else:
            state = dict(schema_version=SCHEMA_VERSION, read_only=True,
                manifest_fingerprint=digest(manifest), status="partial", reason="initializing",
                bindings=None, tasks=[], task_index=0, pages_completed=0,
                counts={"sources": 0, "entries": 0, "chunks": 0, "vector_points": 0}, findings={},
                consistency="bounded_online_audit_not_a_cross_store_atomic_snapshot")
        if state.get("manifest_fingerprint") != digest(manifest):
            state.update(status="stale", reason="manifest_changed_resume_rejected")
            _write(checkpoint, state)
            return state
        if state.get("status") == "stale":
            return state
        deadline = time.monotonic() + deadline_seconds
        readers, definitions, bindings = {}, {}, {}
        allowed_services = manifest.get("vector_services", [])
        def vectors(index):
            _authorize_vector(index, allowed_services)
            return (vector_factory(index) if vector_factory is not None else
                    ReadOnlyVector(index, allowed_services=allowed_services))
        try:
            for source in manifest["sources"]:
                _check(deadline)
                alias = source["alias"]
                reader = source_factory(source)
                readers[alias] = reader
                binding, index_rows = reader.binding(deadline)
                bindings[alias] = binding
                for row in index_rows:
                    _authorize_vector(row, allowed_services)
                    definitions[(alias, row["workspace_id"], row["generation"])] = row
            if state["bindings"] is not None and state["bindings"] != bindings:
                state.update(status="stale", reason="source_or_grants_or_index_changed")
                _write(checkpoint, state)
                return state
            expected_tasks = _tasks(manifest, definitions)
            if state["bindings"] is None:
                state["bindings"] = bindings
                state["tasks"] = expected_tasks
            _validate_checkpoint(state, expected_tasks)
            state.update(status="partial", reason="running")
            _write(checkpoint, state)
            pages = 0
            while state["task_index"] < len(state["tasks"]) and pages < max_pages:
                _check(deadline)
                # A failed page never advances its cursor or double-counts findings.
                candidate = copy.deepcopy(state)
                task = candidate["tasks"][candidate["task_index"]]
                alias, scope, phase = task["alias"], task["scope"], task["phase"]
                reader = readers[alias]
                if reader.grants() != bindings[alias]["grant"]:
                    raise AuditBlocked("scope_grant_changed")
                def emit(identifier, version, code, **kw):
                    _finding(candidate, alias, scope, identifier, version, code, **kw)
                if phase == "vector_points":
                    index = definitions[(alias, scope, task["generation"])]
                    rows, cursor = vectors(index).page(scope, task["cursor"], page_size, deadline)
                    for row in rows:
                        _check(deadline)
                        _check_point(reader, scope, index, row, emit)
                else:
                    rows, cursor = reader.page(scope, phase, task["cursor"], page_size)
                    pending = {}
                    for row in rows:
                        _check(deadline)
                        if phase == "sources":
                            _check_source(reader, scope, row, emit)
                        elif phase == "entries":
                            _check_entry(reader, scope, row, emit)
                        elif _check_chunk(reader, scope, row, emit):
                            pending.setdefault(row["generation"], []).append(row)
                    for generation, group in pending.items():
                        index = definitions.get((alias, scope, generation))
                        if index is None:
                            raise AuditBlocked("registered_index_missing")
                        points = vectors(index).points([r["point_id"] for r in group], deadline)
                        by_id = {str(p.get("id")): p for p in points}
                        for row in group:
                            point = by_id.get(row["point_id"])
                            if point is None:
                                emit(row["memory_id"], row["case_version"], "ready_chunk_vector_missing", point_id=row["point_id"])
                            else:
                                _check_point(reader, scope, index, point, emit)
                _check(deadline)
                if reader.grants() != bindings[alias]["grant"]:
                    raise AuditBlocked("scope_grant_changed")
                candidate["counts"][phase] += len(rows)
                candidate["pages_completed"] += 1
                task["cursor"] = cursor
                if cursor is None:
                    candidate["task_index"] += 1
                state = candidate
                pages += 1
                _write(checkpoint, state)
            if state["task_index"] == len(state["tasks"]):
                # Source edits/index worker progress during audit invalidate the run.
                latest = {alias: reader.binding(deadline)[0] for alias, reader in readers.items()}
                if latest != bindings:
                    state.update(status="stale", reason="source_changed_during_audit")
                else:
                    state.update(status="complete", reason="all_registered_scopes_and_generations_checked")
            else:
                state.update(status="partial", reason="page_budget_exhausted")
        except AuditDeadline:
            state.update(status="partial", reason="deadline_exhausted_resume_available")
        except AuditBlocked as error:
            state.update(status="blocked", reason=str(error))
        except (TimeoutError, ConnectionError):
            state.update(status="partial", reason="read_timeout_resume_available")
        except (PermissionError, OSError, SQLAlchemyError, ValueError, KeyError, TypeError):
            state.update(status="blocked", reason="source_unavailable_or_response_invalid")
        except Exception:
            # Vector transport exceptions intentionally hide response bodies/keys.
            state.update(status="blocked", reason="read_transport_failed_resume_available")
        _write(checkpoint, state)
        return state
