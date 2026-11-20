"""Real SQL scope queries + gateway boundaries; no workbench, models or live data."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.schema import audit, users, workspaces
from codex_agent.accounts.store import AccountError
from codex_agent.accounts import usage
from codex_agent.storage.schema import runs, sessions, workspaces as domain_workspaces


ADMIN, FIRST, SECOND, NEW = (letter * 32 for letter in "abcd")
ORIGIN = "http://localhost:8766"


@compiles(LONGTEXT, "sqlite")
def sqlite_longtext(_type, _compiler, **_kwargs):
    return "TEXT"


class Accounts:
    def __init__(self, engine):
        self.engine, self.limits = engine, []
        self.tokens = {uid: 1 for uid in (ADMIN, FIRST, SECOND, NEW)}

    def authenticate(self, token, *, touch=True):
        if token not in self.tokens:
            raise AccountError(401, "Sign in required")
        with self.engine.connect() as conn:
            row = conn.execute(select(users).where(users.c.id == token)).mappings().one()
        if row["status"] != "active" or row["auth_version"] != self.tokens[token]:
            raise AccountError(401, "Session revoked")
        return {"id": row["id"], "username": row["username"], "csrf_token": "csrf"}

    def limit(self, *args):
        self.limits.append(args)

    def logout(self, token, _uid):
        self.tokens.pop(token, None)

    def log(self, *_args):
        pass


class NoWorkbenches:
    def get(self, _uid):
        raise AssertionError("Usage or reserved gateway routes must not provision a private service")


def insert_run(conn, uid, i, *, failed=False):
    conn.execute(runs.insert().values(workspace_id=uid, id=f"run-{i:02}", session_id="session",
        intent=f"SECRET-intent-{uid}", operator="private-operator", status="failed" if failed else "completed",
        phase="completed", request_json='{"secret":"private-request"}', result_json='{"secret":"private-result"}',
        created_at=f"2026-11-14T00:{i:02}:00Z", updated_at=f"2026-11-14T01:{i:02}:00Z", next_sequence=0))


@pytest.fixture
def env(tmp_path):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    for table in (users, workspaces, audit, domain_workspaces, sessions, runs):
        table.create(engine)
    with engine.begin() as conn:
        for n, uid in enumerate((ADMIN, FIRST, SECOND, NEW)):
            conn.execute(users.insert().values(id=uid, username=f"person-{n}", password_hash="SECRET-password-hash",
                status="active", auth_version=1, created_ms=n))
        for uid, count in ((FIRST, 12), (SECOND, 3)):
            conn.execute(workspaces.insert().values(workspace_id=uid, user_id=uid, root=f"/private/{uid}", status="ready"))
            conn.execute(domain_workspaces.insert().values(workspace_id=uid, root=f"/private/{uid}"))
            conn.execute(sessions.insert().values(workspace_id=uid, id="session", title="SECRET-session",
                status="active", created_at="2026-11-13T00:00:00Z", updated_at="2026-11-13T00:00:00Z", pinned=0))
            for i in range(count):
                insert_run(conn, uid, i, failed=(i < 2))
    cfg = AccountConfig(origin=ORIGIN, data_dir=tmp_path / "data", template=tmp_path / "template",
        allow_local_http=True, admin_user_ids=(ADMIN,), knowledge_enabled=False)
    accounts = Accounts(engine)
    client = TestClient(create_gateway(cfg, accounts=accounts, workbenches=NoWorkbenches()), base_url=ORIGIN)
    client.cookies.set(cfg.cookie_name, FIRST)
    yield SimpleNamespace(engine=engine, config=cfg, accounts=accounts, client=client)
    client.close()
    engine.dispose()


def headers(uid=FIRST):
    return {"Origin": ORIGIN, "X-CSRF-Token": "csrf", "X-Triton-Account": uid}


def test_personal_usage_is_scoped_and_recent_runs_are_bounded(env):
    response = env.client.get("/api/account/usage", headers=headers())
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["account"] == {"id": FIRST, "username": "person-1", "status": "active"}
    assert result["totals"] == {"sessions": 1, "runs": 12, "failed_runs": 2}
    assert result["last_used_at"] == "2026-11-14T01:11:00Z"
    assert [row["id"] for row in result["recent_runs"]] == [f"run-{i:02}" for i in range(11, 1, -1)]
    assert all(set(row) == {"id", "status", "created_at", "updated_at"} for row in result["recent_runs"])
    assert set(result) == {"account", "totals", "last_used_at", "recent_runs", "as_of"}
    assert result["as_of"] and "SECRET" not in response.text and "/private" not in response.text
    assert SECOND not in response.text
    assert env.accounts.limits[-1] == ("account-usage", FIRST, 60, 60)


def test_second_user_has_independent_totals_even_with_identical_run_ids(env):
    env.client.cookies.set(env.config.cookie_name, SECOND)
    result = env.client.get("/api/account/usage", headers=headers(SECOND)).json()
    assert result["account"]["id"] == SECOND
    assert result["totals"] == {"sessions": 1, "runs": 3, "failed_runs": 2}
    assert len(result["recent_runs"]) == 3


def test_new_account_has_zero_usage_without_creating_workspace(env):
    env.client.cookies.set(env.config.cookie_name, NEW)
    result = env.client.get("/api/account/usage", headers=headers(NEW)).json()
    assert result["totals"] == {"sessions": 0, "runs": 0, "failed_runs": 0}
    assert result["last_used_at"] is None and result["recent_runs"] == []
    with env.engine.connect() as conn:
        assert conn.execute(select(workspaces).where(workspaces.c.user_id == NEW)).first() is None
    assert not env.config.data_dir.exists()


@pytest.mark.parametrize("query", [f"user_id={SECOND}", f"account_id={SECOND}", f"workspace_id={SECOND}",
    "scope=all", "offset=0", "user_id=x&user_id=y"])
def test_caller_cannot_select_another_user_or_workspace(env, query):
    response = env.client.get("/api/account/usage?" + query, headers=headers())
    assert response.status_code == 422 and SECOND not in response.text


def test_forged_identity_header_rejected_before_read(env):
    assert env.client.get("/api/account/usage", headers=headers(SECOND)).status_code == 401
    assert not env.accounts.limits


def test_admin_overview_includes_all_accounts_but_personal_endpoint_stays_personal(env):
    env.client.cookies.set(env.config.cookie_name, ADMIN)
    result = env.client.get("/api/admin/overview", headers=headers(ADMIN)).json()
    assert result["totals"] == {"users": 4, "sessions": 2, "runs": 15, "failed_runs": 4}
    assert {row["id"] for row in result["users"]} == {ADMIN, FIRST, SECOND, NEW}
    personal = env.client.get("/api/account/usage", headers=headers(ADMIN)).json()
    assert personal["account"]["id"] == ADMIN and personal["totals"]["runs"] == 0


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/admin/overview"), ("GET", "/api/admin/knowledge"),
    ("GET", "/api/admin/knowledge/abc"), ("GET", "/api/admin/knowledge/abc/original"),
    ("GET", "/api/admin/knowledge/operations/request"),
    ("POST", "/api/admin/knowledge"), ("POST", "/api/admin/knowledge/confirm"),
    ("POST", "/api/admin/knowledge/abc/index"), ("POST", "/api/admin/knowledge/abc/publish"),
    ("POST", "/api/admin/knowledge/abc/unpublish"), ("POST", "/api/admin/knowledge/abc/delete"),
    ("POST", "/api/admin/knowledge/abc/restore"),
    ("GET", "/api/admin/rag/sources"), ("GET", "/api/admin/rag/catalog"),
    ("GET", "/api/admin/rag/catalog/summary"), ("GET", "/api/admin/rag/catalog/reconciliation"),
    ("GET", "/api/admin/rag/catalog/item"), ("GET", "/api/admin/rag/catalog/item/versions"),
    ("GET", "/api/admin/rag/catalog/item/versions/1"), ("GET", "/api/admin/rag/catalog/item/chunks?version=1"),
    ("GET", "/api/admin/rag/operations/request?item_id=item"),
    ("POST", "/api/admin/rag/catalog/item/actions"), ("POST", "/api/admin/rag/catalog/item/share?source_id=daily-history"),
    ("GET", "/api/admin/unknown"), ("DELETE", "/api/admin/knowledge/abc"),
])
def test_ordinary_user_cannot_read_or_mutate_admin_resources(env, method, path):
    kwargs = {"json": {}} if method != "GET" else {}
    response = env.client.request(method, path, headers=headers(), **kwargs)
    assert response.status_code == 403


def test_ordinary_user_cannot_upload_knowledge_file(env):
    response = env.client.post("/api/admin/knowledge/files?filename=test.pdf&request_id=blocked-upload", content=b"file",
        headers={**headers(), "Content-Type": "application/octet-stream"})
    assert response.status_code == 403


@pytest.mark.parametrize("method,path", [("POST", "/api/account/usage"),
    ("GET", "/api/account/other/usage"), ("GET", "/api/account/usage/other")])
def test_reserved_account_namespace_cannot_fall_through_to_private_workbench(env, method, path):
    kwargs = {"json": {}} if method != "GET" else {}
    assert env.client.request(method, path, headers=headers(), **kwargs).status_code == 404


def test_admin_unknown_route_does_not_fall_through(env):
    env.client.cookies.set(env.config.cookie_name, ADMIN)
    assert env.client.get("/api/admin/unknown", headers=headers(ADMIN)).status_code == 404


def test_logged_out_and_disabled_accounts_have_no_usage_access(env):
    assert env.client.post("/api/auth/logout", headers=headers(), json={}).status_code == 200
    assert env.client.get("/api/account/usage").status_code == 401
    env.client.cookies.set(env.config.cookie_name, SECOND)
    with env.engine.begin() as conn:
        conn.execute(users.update().where(users.c.id == SECOND).values(status="disabled"))
    assert env.client.get("/api/account/usage", headers=headers(SECOND)).status_code == 401


def test_revoked_session_is_rejected(env):
    with env.engine.begin() as conn:
        conn.execute(users.update().where(users.c.id == FIRST).values(auth_version=2))
    assert env.client.get("/api/account/usage", headers=headers()).status_code == 401


def test_revocation_during_usage_read_discards_result(env, monkeypatch):
    original = usage.personal_usage
    def revoke(accounts, user_id):
        result = original(accounts, user_id)
        accounts.tokens.pop(user_id)
        return result
    monkeypatch.setattr(usage, "personal_usage", revoke)
    response = env.client.get("/api/account/usage", headers=headers())
    assert response.status_code == 401 and "recent_runs" not in response.text


@pytest.mark.parametrize("error", [TimeoutError("private driver secret"), SQLAlchemyError("database password")])
def test_errors_fail_closed_without_leaking_database_details(env, monkeypatch, error):
    def fail(*_):
        raise error
    monkeypatch.setattr(usage, "personal_usage", fail)
    response = env.client.get("/api/account/usage", headers=headers())
    assert response.status_code == 503 and "private" not in response.text and "password" not in response.text


def test_rate_limit_precedes_data_query(env, monkeypatch):
    def limit(*_):
        raise AccountError(429, "Too many requests")
    def unexpected(*_):
        raise AssertionError("Rate limited request must not query usage")
    monkeypatch.setattr(env.accounts, "limit", limit)
    monkeypatch.setattr(usage, "personal_usage", unexpected)
    assert env.client.get("/api/account/usage", headers=headers()).status_code == 429


def test_workspace_reconciliation_does_not_show_false_zeroes(env):
    with env.engine.begin() as conn:
        conn.execute(workspaces.update().where(workspaces.c.user_id == FIRST).values(status="pending"))
    assert env.client.get("/api/account/usage", headers=headers()).status_code == 409


def test_usage_refresh_observes_new_run_without_global_cache(env):
    first = env.client.get("/api/account/usage", headers=headers()).json()
    with env.engine.begin() as conn:
        insert_run(conn, FIRST, 12)
    second = env.client.get("/api/account/usage", headers=headers()).json()
    assert first["totals"]["runs"] == 12 and second["totals"]["runs"] == 13
    assert second["recent_runs"][0]["id"] == "run-12"
