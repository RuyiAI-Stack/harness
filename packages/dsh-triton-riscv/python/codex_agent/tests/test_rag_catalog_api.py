"""Gateway boundary tests with fake accounts/catalog; no private service or model."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.store import AccountError
from codex_agent.rag_catalog import api
from codex_agent.tasks.store import Conflict


ADMIN = "a" * 32
OTHER = "b" * 32
ORIGIN = "http://localhost:8766"


class Accounts:
    engine = object()

    def __init__(self):
        self.limits = []

    def authenticate(self, token, *, touch=True):
        if token not in {"admin-session", "other-session"}:
            raise AccountError(401, "Login required")
        return {"id": ADMIN if token == "admin-session" else OTHER, "csrf_token": "confirmed-csrf"}

    def limit(self, *args):
        self.limits.append(args)


class Catalog:
    def __init__(self, calls):
        self.calls = calls
        self.error = None

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.error:
            raise self.error
        return {"route": name, "source_kind": "memory_case", "selected_version": "v1"}

    def catalog(self, **kwargs): return self._call("catalog", **kwargs)
    def summary(self, **kwargs): return self._call("summary", **kwargs)
    def reconciliation(self, **kwargs): return self._call("reconciliation", **kwargs)
    def detail(self, item_id, **kwargs): return self._call("detail", item_id, **kwargs)
    def versions(self, item_id, **kwargs): return self._call("versions", item_id, **kwargs)
    def chunks(self, item_id, version, **kwargs): return self._call("chunks", item_id, version, **kwargs)


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    cfg = AccountConfig(origin=ORIGIN, data_dir=tmp_path / "data", template=tmp_path / "template",
        allow_local_http=True, admin_user_ids=(ADMIN,), knowledge_enabled=False)
    accounts, calls, mutations = Accounts(), [], []
    catalog = Catalog(calls)
    monkeypatch.setattr(api, "runtime_config", lambda: SimpleNamespace(
        indexing=SimpleNamespace(enabled=False), memory=SimpleNamespace(retrievalMode="legacy")))
    monkeypatch.setattr(api, "authorized_scopes", lambda *args: [])
    monkeypatch.setattr(api, "CatalogStore", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(api.actions, "corrections", lambda *_: [])
    monkeypatch.setattr(api.actions, "execute", lambda *args: mutations.append(args))
    monkeypatch.setattr(api.actions, "operation", lambda _catalog, actor, rid, item_id:
                        {"actor": actor, "request_id": rid, "item_id": item_id})
    client = TestClient(create_gateway(cfg, accounts=accounts, workbenches=object()), base_url=ORIGIN)
    client.cookies.set(cfg.cookie_name, "admin-session")
    yield SimpleNamespace(config=cfg, accounts=accounts, client=client, calls=calls,
                          mutations=mutations, catalog=catalog)
    client.close()


def headers(**changes):
    return {"origin": ORIGIN, "x-csrf-token": "confirmed-csrf", **changes}


def action(**changes):
    return {"action": "correct", "expected_revision": "rev", "request_id": "request-one",
            "content": "This is an annotation only", **changes}


def test_catalog_requires_authenticated_admin_before_any_read(gateway):
    f = gateway
    f.client.cookies.clear()
    assert f.client.get("/api/admin/rag/catalog").status_code == 401
    f.client.cookies.set(f.config.cookie_name, "other-session")
    assert f.client.get("/api/admin/rag/catalog").status_code == 403
    assert f.calls == [] and f.mutations == []


@pytest.mark.parametrize("extra", [{"origin": "https://attacker.example"},
    {"x-csrf-token": "wrong"}, {"x-triton-account": OTHER}, {"sec-fetch-site": "cross-site"}])
def test_mutation_rejects_cross_origin_csrf_and_stale_account(gateway, extra):
    r = gateway.client.post("/api/admin/rag/catalog/item/actions", json=action(), headers=headers(**extra))
    assert r.status_code in {401, 403}
    assert not gateway.mutations


def test_successful_action_is_bound_to_authenticated_actor_and_read_back(gateway):
    r = gateway.client.post("/api/admin/rag/catalog/item/actions", json=action(), headers=headers())
    assert r.status_code == 200 and r.json()["route"] == "detail"
    assert gateway.mutations[0][1] == "item" and gateway.mutations[0][3] == ADMIN
    assert r.headers["cache-control"] == "no-store"
    assert gateway.accounts.limits[-1] == ("rag-admin", ADMIN, 120, 60)


@pytest.mark.parametrize("value", [action(root="/private/another-user"), action(actor=OTHER),
    action(action="delete"), action(expected_revision=1), action(content=" "),
    action(request_id="bad\0key"), action(action="archive", content="not allowed")])
def test_invalid_actions_and_browser_supplied_scope_are_rejected(gateway, value):
    r = gateway.client.post("/api/admin/rag/catalog/item/actions", json=value, headers=headers())
    assert r.status_code == 422 and not gateway.mutations


@pytest.mark.parametrize("path,route", [("/api/admin/rag/catalog/summary", "summary"),
    ("/api/admin/rag/catalog/reconciliation", "reconciliation"),
    ("/api/admin/rag/catalog/item/versions", "versions"),
    ("/api/admin/rag/catalog/item/versions/v1", "detail"),
    ("/api/admin/rag/catalog/item/chunks?version=v1", "chunks")])
def test_specific_routes_do_not_fall_into_item_or_proxy(gateway, path, route):
    r = gateway.client.get(path)
    assert r.status_code == 200 and r.json()["route"] == route


@pytest.mark.parametrize("error,status,forbidden", [(KeyError("secret-root"), 404, "secret-root"),
    (PermissionError("private-path"), 403, "private-path"),
    (ValueError("password=private"), 422, "password"),
    (Conflict("请重新读取"), 409, None), (TimeoutError("driver secret"), 503, "driver secret")])
def test_read_failure_is_bounded_and_safe(gateway, error, status, forbidden):
    gateway.catalog.error = error
    r = gateway.client.get("/api/admin/rag/catalog")
    assert r.status_code == status
    if forbidden:
        assert forbidden not in r.text


def test_mutation_unknown_result_tells_user_to_read_receipt(gateway, monkeypatch):
    def interrupted(*_):
        raise TimeoutError()
    monkeypatch.setattr(api.actions, "execute", interrupted)
    r = gateway.client.post("/api/admin/rag/catalog/item/actions", json=action(), headers=headers())
    assert r.status_code == 503 and "操作回执" in r.json()["detail"]
    receipt = gateway.client.get("/api/admin/rag/operations/request-one?item_id=item")
    assert receipt.status_code == 200 and receipt.json()["actor"] == ADMIN
    assert receipt.json()["item_id"] == "item"
    assert gateway.client.get("/api/admin/rag/operations/request-one").status_code == 422


def test_body_limit_applies_before_mutation(gateway):
    r = gateway.client.post("/api/admin/rag/catalog/item/actions",
        content='{"content":"' + "x" * 100001 + '"}',
        headers={**headers(), "content-type": "application/json"})
    assert r.status_code == 413 and not gateway.mutations


def test_feature_flag_disables_routes_for_admin(gateway):
    # The gateway reads its immutable config object; build one disabled instance.
    cfg = gateway.config.model_copy(update={"rag_catalog_enabled": False})
    c = TestClient(create_gateway(cfg, accounts=gateway.accounts, workbenches=object()), base_url=ORIGIN)
    c.cookies.set(cfg.cookie_name, "admin-session")
    try:
        assert c.get("/api/admin/rag/catalog").status_code == 404
    finally:
        c.close()
    assert not gateway.calls
