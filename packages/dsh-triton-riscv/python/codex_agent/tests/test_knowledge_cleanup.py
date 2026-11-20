"""Isolated SQL/fake-vector lifecycle tests; no production DB, model or Qdrant."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.pool import StaticPool

from codex_agent.knowledge import cleanup
from codex_agent.knowledge.lifecycle import controls, DocumentUnavailable
from codex_agent.knowledge.schema import documents, versions
from codex_agent.storage.schema import meta
from codex_agent.tasks.schema import jobs as job_table
from codex_agent.tasks.store import Conflict, digest, encode


@compiles(LONGTEXT, "sqlite")
def _text(_type, _compiler, **_kw):
    return "TEXT"


class DB:
    def __init__(self):
        self.root = Path("/tmp/isolated-knowledge-cleanup")
        self.workspace_id = "a" * 64
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        for table in (documents, versions, controls, meta, job_table):
            table.create(self.engine)
        self.depth = 0

    @contextmanager
    def transaction(self):
        with self.engine.begin() as conn:
            yield conn

    def lock(self, conn, name):
        assert name == "knowledge-write"

    def insert(self, conn, table, value):
        conn.execute(table.insert().values(workspace_id=self.workspace_id, **value))

    def update(self, conn, table, value, *conditions):
        return conn.execute(table.update().where(table.c.workspace_id == self.workspace_id,
                                                 *conditions).values(**value))


class Store:
    def __init__(self, db):
        self.db, self.root, self.scope, self.engine = db, db.root, db.workspace_id, db.engine

    def rows(self, conn, table, *conditions):
        return [dict(row) for row in conn.execute(select(table).where(
            table.c.workspace_id == self.scope, *conditions)).mappings()]

    def row(self, conn, table, identifier, version=None, *, allow_deleted=False):
        conditions = ([table.c.document_id == identifier, table.c.version == version]
                      if version is not None else [table.c.id == identifier])
        rows = self.rows(conn, table, *conditions)
        if not rows:
            raise KeyError(identifier)
        return rows[0]


class Jobs:
    def __init__(self, db):
        self.db, self.submitted, self.finished = db, [], []

    def require_owner(self, conn, identifier, owner):
        row = conn.execute(select(job_table).where(job_table.c.id == identifier)).mappings().one()
        assert row["owner"] == owner
        return dict(row)

    def begin_effects(self, identifier, owner):
        pass

    def find_key(self, conn, key):
        return None

    def finish(self, identifier, owner, status, result, *, connection):
        self.db.update(connection, job_table, {"status": status, "result_json": encode(result)},
                       job_table.c.id == identifier)
        self.finished.append((identifier, status, result))

    def submit(self, kind, payload, key, *, connection):
        identifier = "queued-" + str(len(self.submitted))
        value = make_job(identifier, payload, status="queued")
        self.db.insert(connection, job_table, value)
        self.submitted.append(value)
        return value


def make_job(identifier, payload, status="running"):
    return dict(id=identifier, kind="index", status=status, idempotency_key=identifier,
        fingerprint=digest(payload), payload_json=encode(payload), result_json="{}", run_id=None,
        execution_host="fixture", attempt=1, generation=1, owner="owner", lease_until=9999999999999,
        effects_started=0, cancel_requested=0, created_ms=1, updated_ms=1)


@pytest.fixture
def setup(monkeypatch):
    db, definition = DB(), {"generation": "g1", "timeoutSeconds": 30, "dimensions": 2}
    store, jobs = Store(db), Jobs(db)
    monkeypatch.setattr(cleanup, "KnowledgeStore", lambda _root: store)
    monkeypatch.setattr(cleanup, "JobStore", lambda _root, database: jobs)
    identifier = "b" * 32
    with db.transaction() as conn:
        db.insert(conn, documents, dict(id=identifier, latest_version=1, published_version=None,
                                       created_at="2026-01-01", updated_at="2026-01-01"))
        db.insert(conn, controls, dict(document_id=identifier, revision=1,
                                      deleted_at="2026-01-02", actor_id="admin"))
        db.insert(conn, versions, dict(document_id=identifier, version=1, title="Original title",
            source="fixture", content="Original text", source_hash="x"*64, state="ready",
            definition_json=encode(definition), fingerprint=digest(definition), job_id=None,
            chunk_count=2, error=None, created_at="2026-01-01"))
        job = make_job("cleanup", dict(domain="knowledge-cleanup", document_id=identifier,
                                      catalog_revision=1, after_version=0, actor_id="admin"))
        db.insert(conn, job_table, job)
    yield db, store, jobs, identifier, job, definition
    db.engine.dispose()


class Index:
    def __init__(self, scope, definition, *, count=0, error=None):
        self.scope = SimpleNamespace(workspace_id=scope, generation=definition["generation"],
                                     index_fingerprint=digest(definition))
        self.transport = SimpleNamespace(timeout=30, request=self.request)
        self.deletions, self.count_queries = [], []
        self.count, self.error = count, error

    def delete_version(self, identifier, version):
        self.deletions.append((identifier, version))
        if self.error:
            raise self.error

    def request(self, method, path, body, *, missing_ok):
        self.count_queries.append(body)
        return {"count": self.count}


def read_version(store, identifier):
    with store.engine.connect() as conn:
        return store.rows(conn, versions, versions.c.document_id == identifier)[0]


def test_cleanup_verifies_exact_scope_and_retains_original(setup):
    db, store, jobs, identifier, job, definition = setup
    index = Index(store.scope, definition)
    cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: index)
    assert index.deletions == [(identifier, 1)]
    assert index.transport.timeout == 5
    conditions = {entry["key"]: entry["match"]["value"] for entry in index.count_queries[0]["filter"]["must"]}
    assert conditions == dict(workspace_id=store.scope, document_id=identifier, version=1,
                              generation="g1", index_fingerprint=digest(definition))
    assert read_version(store, identifier)["content"] == "Original text"
    assert read_version(store, identifier)["state"] == "stale"
    with db.engine.connect() as conn:
        result = cleanup.cleanup_status(store, conn, identifier)
    assert result["state"] == "complete" and result["verified_at"]


def test_timeout_does_not_leave_a_publishable_version(setup):
    _, store, jobs, identifier, job, definition = setup
    index = Index(store.scope, definition, error=TimeoutError("synthetic timeout"))
    with pytest.raises(TimeoutError):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: index)
    assert read_version(store, identifier)["state"] == "stale"
    assert not jobs.finished


def test_nonempty_readback_is_not_success(setup):
    _, store, jobs, identifier, job, definition = setup
    index = Index(store.scope, definition, count=1)
    with pytest.raises(Conflict, match="confirmed empty"):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: index)
    assert not jobs.finished
    assert read_version(store, identifier)["state"] == "stale"


def test_restore_between_admission_and_remote_call_prevents_delete(setup):
    db, store, jobs, identifier, job, definition = setup
    index = Index(store.scope, definition)
    def restore_after_stale(*_):
        with db.transaction() as conn:
            db.update(conn, controls, {"revision": 2, "deleted_at": None}, controls.c.document_id == identifier)
        return index
    with pytest.raises(DocumentUnavailable):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=restore_after_stale)
    assert not index.deletions
    assert read_version(store, identifier)["state"] == "stale"


def test_obsolete_cleanup_does_not_touch_restored_source(setup):
    db, store, jobs, identifier, job, definition = setup
    with db.transaction() as conn:
        db.update(conn, controls, {"revision": 2, "deleted_at": None}, controls.c.document_id == identifier)
    factory_calls = []
    with pytest.raises(DocumentUnavailable):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: factory_calls.append(True))
    assert not factory_calls
    assert read_version(store, identifier)["state"] == "ready"


def test_known_inflight_index_worker_blocks_cleanup(setup):
    db, store, jobs, identifier, job, definition = setup
    with db.transaction() as conn:
        db.insert(conn, job_table, make_job("prior-index", {}))
        db.update(conn, versions, {"job_id": "prior-index"}, versions.c.document_id == identifier)
    with pytest.raises(Conflict, match="earlier document worker"):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: pytest.fail("network forbidden"))
    assert read_version(store, identifier)["state"] == "ready"


def test_retry_after_unknown_delete_repeats_exact_version_safely(setup):
    _, store, jobs, identifier, job, definition = setup
    failing = Index(store.scope, definition, error=TimeoutError())
    with pytest.raises(TimeoutError):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: failing)
    successful = Index(store.scope, definition)
    cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: successful)
    assert failing.deletions == successful.deletions == [(identifier, 1)]
    assert jobs.finished[-1][2]["cleanup_state"] == "complete"


def test_cleanup_attempt_is_bounded_to_one_version(setup):
    db, store, jobs, identifier, job, definition = setup
    source = read_version(store, identifier)
    with db.transaction() as conn:
        db.insert(conn, versions, {k: (2 if k == "version" else value) for k, value in source.items()
                                  if k != "workspace_id"})
        db.update(conn, documents, {"latest_version": 2}, documents.c.id == identifier)
    index = Index(store.scope, definition)
    cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: index)
    assert index.deletions == [(identifier, 1)]
    assert len(jobs.submitted) == 1
    assert json.loads(jobs.submitted[0]["payload_json"])["after_version"] == 1
    assert jobs.finished[-1][2]["cleanup_state"] == "queued"


def test_enqueue_is_idempotent_while_job_is_live(setup):
    db, store, jobs, identifier, job, _ = setup
    with db.transaction() as conn:
        first = cleanup.enqueue_cleanup(store, db, conn, identifier, 1, "admin")
        second = cleanup.enqueue_cleanup(store, db, conn, identifier, 1, "admin", "another-click")
    assert first["job_id"] == second["job_id"]
    assert len(jobs.submitted) == 1


def test_terminal_job_failure_is_visible_without_raw_errors(setup):
    db, store, jobs, identifier, _, _ = setup
    with db.transaction() as conn:
        receipt = cleanup.enqueue_cleanup(store, db, conn, identifier, 1, "admin")
        db.update(conn, job_table, {"status": "dead", "result_json": '{"secret":"do not expose"}'},
                  job_table.c.id == receipt["job_id"])
    with db.engine.connect() as conn:
        public = cleanup.cleanup_status(store, conn, identifier)
    assert public["state"] == "failed"
    assert "secret" not in encode(public)


def test_cleaned_version_cannot_be_reindexed_after_restore(setup, monkeypatch):
    from codex_agent.knowledge import store as store_module, vectors as vectors_module
    db, store, jobs, identifier, job, definition = setup
    index = Index(store.scope, definition, error=TimeoutError())
    with pytest.raises(TimeoutError):
        cleanup.execute_cleanup(jobs, job, "owner", index_factory=lambda *_: index)
    with db.transaction() as conn:
        db.update(conn, controls, {"revision": 2, "deleted_at": None}, controls.c.document_id == identifier)
    store.writer = lambda: db
    from types import MethodType
    store.enqueue_transaction = MethodType(store_module.KnowledgeStore.enqueue_transaction, store)
    monkeypatch.setattr(store_module, "source_row", lambda *_: None)
    monkeypatch.setattr(store_module, "JobStore", lambda _root, database: jobs)
    monkeypatch.setattr(vectors_module, "index_definition", lambda: definition)
    with pytest.raises(Conflict, match="save a new version"):
        store_module.KnowledgeStore.enqueue(store, identifier, 1, "reindex", "admin", 2)
    # A delayed delete still targets version 1; rebuilding must use a new version.
    assert not jobs.submitted
