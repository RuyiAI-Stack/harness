"""SQL-isolated lifecycle, scope, and capability tests; no MySQL or model calls."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select

from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts import guard
from codex_agent.accounts.schema import users, workspaces as account_workspaces
from codex_agent.rag_catalog import actions
from codex_agent.rag_catalog.api import ActionInput
from codex_agent.rag_catalog.scopes import authorized_scopes
from codex_agent.rag_catalog.store import CatalogStore
from codex_agent.storage.database import WorkspaceDatabase, StorageError
from codex_agent.storage.schema import meta, memories, memory_versions, workspaces, counters
from codex_agent.tasks.store import Conflict
from codex_agent.tests.test_rag_catalog_readonly import fixture, store as catalog_store, item, KEY


ACTOR = "a" * 32
OWNER = "b" * 32
THIRD = "c" * 32


def add_user(conn, identifier, *, status="active"):
    conn.execute(users.insert().values(id=identifier, username="user-" + identifier[:4],
        password_hash="not-a-real-password", status=status, auth_version=1, created_ms=1))


class IsolatedDB(WorkspaceDatabase):
    """Use actual scoped SQL writes/_changed, substitute only MySQL transaction locks."""
    def __init__(self, scope, engine):
        self.root, self.workspace_id, self.engine = scope["root"], scope["id"], engine
        self.committed_changes = []

    @contextmanager
    def transaction(self):
        changes = None
        with self.engine.begin() as conn:
            conn.info["cache_changes"] = set()
            try:
                yield conn
                changes = set(conn.info["cache_changes"])
                for wid, domain in changes:
                    name = "cache-version:" + domain
                    existing = conn.execute(select(counters.c.value).where(
                        counters.c.workspace_id == wid, counters.c.name == name)).scalar_one_or_none()
                    if existing is None:
                        conn.execute(counters.insert().values(workspace_id=wid, name=name, value=1))
                    else:
                        conn.execute(counters.update().where(counters.c.workspace_id == wid,
                            counters.c.name == name).values(value=existing + 1))
            finally:
                conn.info.pop("cache_changes", None)
        self.committed_changes.append(changes)

    def lock(self, conn, name):
        assert name in {"memory-write", "rag-operation"}


class IsolatedMemory:
    def __init__(self, db):
        self.db, self.index_requests = db, []
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def _queue_index(self, conn, memory_id):
        self.index_requests.append(memory_id)


@pytest.fixture
def action_fixture(fixture, monkeypatch):
    f = fixture
    users.create(f["db"])
    meta.create(f["db"])
    with f["db"].begin() as conn:
        add_user(conn, ACTOR)
    monkeypatch.delenv(guard.OWNER_ENV, raising=False)
    dbs = {str(scope["root"]): IsolatedDB(scope, f["db"]) for scope in (f["team"], f["denied"])}
    stores = {path: IsolatedMemory(db) for path, db in dbs.items()}
    monkeypatch.setattr(actions, "MemoryStore", lambda root: stores[str(root)])
    f.update(catalog=catalog_store(f), dbs=dbs, memory_stores=stores)
    return f


def value(row, *, action="correct", request_id="request-one", content="管理员说明，尚待审核", **kw):
    return ActionInput(action=action, expected_revision=row["revision"], request_id=request_id,
        content=content if action == "correct" else "", **kw)


def source_rows(f, table=memories):
    with f["db"].connect() as conn:
        return [dict(x) for x in conn.execute(select(table).order_by(table.c.workspace_id)).mappings()]


def test_correction_is_separate_pending_note_not_a_rewrite(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    before, snapshots = source_rows(f), source_rows(f, memory_versions)
    actions.execute(s, row["item_id"], value(row), ACTOR)
    assert source_rows(f) == before and source_rows(f, memory_versions) == snapshots
    notes = actions.corrections(s, row["item_id"])
    assert len(notes) == 1 and notes[0]["state"] == "pending_review"
    assert notes[0]["actor_id"] == ACTOR and notes[0]["source_version"] == row["latest_version"]
    assert not f["memory_stores"][str(f["team"]["root"])].index_requests


def test_exact_retry_creates_one_note_and_has_actor_bound_receipt(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    action = value(row)
    actions.execute(s, row["item_id"], action, ACTOR)
    actions.execute(s, row["item_id"], action, ACTOR)
    assert len(actions.corrections(s, row["item_id"])) == 1
    assert actions.operation(s, ACTOR, action.request_id, row["item_id"])["status"] == "committed"
    assert actions.operation(s, OWNER, action.request_id, row["item_id"]) == {"status": "unknown"}
    with pytest.raises(Conflict, match="different action"):
        actions.execute(s, row["item_id"], value(row, content="different content"), ACTOR)


def test_archive_excludes_only_target_memory_and_invalidates_retrieval_cache(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    original = source_rows(f)
    actions.execute(s, row["item_id"], value(row, action="archive", reason="经核实停用"), ACTOR)
    current = item(s, "memory_case", 1)
    assert current["retrieval_state"] == "excluded" and current["serving_version"] is None
    assert current["revision"] != row["revision"]
    other = [r for r in source_rows(f) if r["workspace_id"] == f["denied"]["id"]]
    assert other == [r for r in original if r["workspace_id"] == f["denied"]["id"]]
    db = f["dbs"][str(f["team"]["root"])]
    assert {(f["team"]["id"], "memory"), (f["team"]["id"], "rag-catalog")} <= db.committed_changes[-1]
    assert f["memory_stores"][str(f["team"]["root"])].index_requests == [1]


def test_archive_retry_stays_idempotent_even_after_source_revision_changes(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    request = value(row, action="archive")
    actions.execute(s, row["item_id"], request, ACTOR)
    actions.execute(s, row["item_id"], request, ACTOR)
    assert f["memory_stores"][str(f["team"]["root"])].index_requests == [1]


def test_stale_source_revision_cannot_mutate_newer_record(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    with f["db"].begin() as conn:
        conn.execute(memories.update().where(memories.c.workspace_id == f["team"]["id"],
            memories.c.id == 1).values(case_version="9" * 64))
    with pytest.raises(Conflict, match="重新读取"):
        actions.execute(s, row["item_id"], value(row), ACTOR)
    assert actions.corrections(s, row["item_id"]) == []


def test_disabled_actor_cannot_apply_previously_prepared_action(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    with f["db"].begin() as conn:
        conn.execute(users.update().where(users.c.id == ACTOR).values(status="disabled"))
    with pytest.raises(PermissionError, match="revoked"):
        actions.execute(s, row["item_id"], value(row), ACTOR)


def test_foreign_and_read_only_grants_cannot_be_mutated(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    limited = CatalogStore([{**f["team"], "writable": False}], cursor_key=KEY, database_engine=f["db"])
    with pytest.raises(PermissionError):
        actions.execute(limited, row["item_id"], value(row), ACTOR)
    narrow = CatalogStore([f["shared"]], cursor_key=KEY, database_engine=f["db"])
    with pytest.raises(KeyError):
        actions.execute(narrow, row["item_id"], value(row), ACTOR)
    with pytest.raises(KeyError):
        actions.operation(narrow, ACTOR, "request-one", row["item_id"])


def test_receipts_distinguish_identical_request_ids_in_different_scopes(action_fixture):
    f = action_fixture
    s = CatalogStore([f["team"], f["denied"]], cursor_key=KEY, database_engine=f["db"], memory_mode="legacy")
    targets = [r for r in s.catalog()["items"] if r["source_id"] == "1"]
    assert len(targets) == 2
    for target in targets:
        actions.execute(s, target["item_id"], value(target), ACTOR)
    for target in targets:
        receipt = actions.operation(s, ACTOR, "request-one", target["item_id"])
        assert receipt["status"] == "committed" and receipt["item_id"] == target["item_id"]
    same_scope_different_item = next(r for r in s.catalog()["items"] if r["source_id"] == "2")
    assert actions.operation(s, ACTOR, "request-one", same_scope_different_item["item_id"]) == {"status": "unknown"}
    with pytest.raises(Conflict, match="different action"):
        actions.execute(s, same_scope_different_item["item_id"], value(same_scope_different_item), ACTOR)


def test_archive_queue_failure_rolls_back_source_and_operation_receipt(action_fixture, monkeypatch):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    def queue_failed(*_):
        raise RuntimeError("synthetic local queue failure")
    monkeypatch.setattr(f["memory_stores"][str(f["team"]["root"])], "_queue_index", queue_failed)
    with pytest.raises(RuntimeError):
        actions.execute(s, row["item_id"], value(row, action="archive"), ACTOR)
    assert item(s, "memory_case", 1)["lifecycle_state"] == "active"
    assert actions.operation(s, ACTOR, "request-one", row["item_id"]) == {"status": "unknown"}


def test_correction_limit_rejects_without_changing_source_or_receipt(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    with f["db"].begin() as conn:
        for index in range(50):
            conn.execute(meta.insert().values(workspace_id=f["team"]["id"],
                key="rag-note:1:" + str(index), value=json.dumps({"content": "previous note"})))
    before = source_rows(f)
    with pytest.raises(Conflict, match="上限"):
        actions.execute(s, row["item_id"], value(row), ACTOR)
    assert source_rows(f) == before
    assert len(actions.corrections(s, row["item_id"])) == 50
    assert actions.operation(s, ACTOR, "request-one", row["item_id"]) == {"status": "unknown"}


def test_correcting_memory_invalidates_catalog_but_not_memory_answer_cache(action_fixture):
    f, s = action_fixture, action_fixture["catalog"]
    row = item(s, "memory_case", 1)
    actions.execute(s, row["item_id"], value(row), ACTOR)
    assert f["dbs"][str(f["team"]["root"])].committed_changes[-1] == {(f["team"]["id"], "rag-catalog")}


@pytest.fixture
def scope_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv(guard.OWNER_ENV, raising=False)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    for table in (users, account_workspaces, workspaces): table.create(engine)
    config = SimpleNamespace(data_dir=tmp_path / "data", knowledge_enabled=True, rag_workspace_ids=())
    def register(uid, *, status="active", linked=True, root=None, wid=None):
        root = (root or config.data_dir / "workspaces" / uid).resolve()
        wid = wid or hashlib.sha256(str(root).encode()).hexdigest()
        with engine.begin() as conn:
            add_user(conn, uid, status=status)
            conn.execute(workspaces.insert().values(workspace_id=wid, root=str(root)))
            if linked:
                conn.execute(account_workspaces.insert().values(workspace_id=wid, user_id=uid,
                    root=str(root), status="ready"))
        return dict(id=wid, root=root)
    own, other = register(ACTOR), register(OWNER)
    yield SimpleNamespace(engine=engine, config=config, own=own, other=other, register=register)
    engine.dispose()


def test_admin_catalog_contains_only_shared_documents(scope_fixture):
    f = scope_fixture
    scopes = authorized_scopes(f.config, f, {"id": ACTOR})
    assert {s["kind"] for s in scopes} == {"knowledge_document"}
    assert len(scopes) == 1 and scopes[0]["root"] == (f.config.data_dir / "knowledge").resolve()
    assert f.own["id"] not in {s["id"] for s in scopes} and f.other["id"] not in {s["id"] for s in scopes}


def test_legacy_workspace_grants_cannot_expose_account_memory(scope_fixture):
    f = scope_fixture
    f.config.rag_workspace_ids = (f.own["id"], f.other["id"])
    assert {s["kind"] for s in authorized_scopes(f.config, f, {"id": ACTOR})} == {"knowledge_document"}
    with f.engine.begin() as conn:
        conn.execute(users.update().where(users.c.id == OWNER).values(status="disabled"))
    assert f.other["id"] not in {s["id"] for s in authorized_scopes(f.config, f, {"id": ACTOR})}


def test_bad_root_binding_is_not_an_authorized_scope(scope_fixture):
    f = scope_fixture
    invalid = f.register(THIRD, root=f.config.data_dir / "not-expected-account-root")
    f.config.rag_workspace_ids = (invalid["id"],)
    assert invalid["id"] not in {s["id"] for s in authorized_scopes(f.config, f, {"id": ACTOR})}


def test_legacy_memory_requires_a_separate_readonly_source(scope_fixture):
    f = scope_fixture
    legacy = f.register(THIRD, linked=False, root=f.config.data_dir / "legacy")
    assert legacy["id"] not in {s["id"] for s in authorized_scopes(f.config, f, {"id": ACTOR})}
    f.config.rag_workspace_ids = (legacy["id"],)
    assert legacy["id"] not in {s["id"] for s in authorized_scopes(f.config, f, {"id": ACTOR})}


def test_disabled_shared_library_does_not_fall_back_to_private_memory(scope_fixture):
    f = scope_fixture
    f.config.knowledge_enabled = False
    assert authorized_scopes(f.config, f, {"id": ACTOR}) == []


def test_admin_workspace_capability_is_exact_and_resets_after_exception(scope_fixture):
    f = scope_fixture
    for uid, entry in ((ACTOR, f.own), (OWNER, f.other)):
        entry["root"].mkdir(parents=True)
        (entry["root"] / guard.MARKER).write_text(uid)
    with f.engine.connect() as conn:
        with pytest.raises(PermissionError): guard.check_workspace(f.other["root"], conn)
        with pytest.raises(RuntimeError, match="finish request"):
            with guard.admin_workspace(f.other["root"], ACTOR):
                guard.check_workspace(f.other["root"], conn)
                with pytest.raises(PermissionError): guard.check_workspace(f.own["root"], conn)
                raise RuntimeError("finish request")
        with pytest.raises(PermissionError): guard.check_workspace(f.other["root"], conn)


def test_account_child_process_cannot_grant_itself_admin_workspace(scope_fixture, monkeypatch):
    monkeypatch.setenv(guard.OWNER_ENV, OWNER)
    with pytest.raises(PermissionError):
        with guard.admin_workspace(scope_fixture.other["root"], ACTOR):
            pytest.fail("must not enter")


def test_admin_scope_rechecks_disabled_target_before_read(scope_fixture):
    f = scope_fixture
    f.other["root"].mkdir(parents=True)
    (f.other["root"] / guard.MARKER).write_text(OWNER)
    with f.engine.begin() as conn:
        conn.execute(users.update().where(users.c.id == OWNER).values(status="disabled"))
        with guard.admin_workspace(f.other["root"], ACTOR), pytest.raises(PermissionError):
            guard.check_workspace(f.other["root"], conn)


@pytest.mark.parametrize("grants", [("not-a-workspace",), ("a" * 64,) * 101])
def test_account_config_rejects_malformed_or_unbounded_grants(tmp_path, grants):
    with pytest.raises(ValueError):
        AccountConfig(origin="https://example.test", data_dir=tmp_path / "data", template=tmp_path / "template",
                      rag_workspace_ids=grants)


def test_cache_domain_changes_require_transaction():
    db = object.__new__(WorkspaceDatabase)
    db.workspace_id = "a" * 64
    conn = SimpleNamespace(info={})
    with pytest.raises(StorageError): db._changed(conn, memories)
    conn.info["cache_changes"] = set()
    db._changed(conn, meta)
    assert conn.info["cache_changes"] == {(db.workspace_id, "rag-catalog")}
