"""Assert real gateway/MySQL/Redis/Qdrant behavior on disposable services only.

Required environment: TRITON_MYSQL_URL, TRITON_AUTH_REDIS_URL,
TRITON_TEST_QDRANT_URL. Schema must already be upgraded. Embeddings are fixed
synthetic vectors; no model or broker is contacted. Jobs use the real SQL
outbox/lease protocol, manually delivered by this fixture.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import select

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.guard import MARKER, admin_workspace
from codex_agent.accounts.schema import workspaces as account_workspaces
from codex_agent.accounts.store import Accounts
from codex_agent.indexing.retrieval import retrieve as indexed_retrieve
from codex_agent.indexing.store import IndexStore
from codex_agent.indexing.worker import execute_index
from codex_agent.knowledge.cleanup import cleanup_status, execute_cleanup
from codex_agent.knowledge.processes import KnowledgeWorkers
from codex_agent.knowledge.schema import versions
from codex_agent.knowledge.source_links import source_links
from codex_agent.knowledge.store import KnowledgeStore
from codex_agent.knowledge.vectors import execute_knowledge_index, index_definition, search, vector_index
from codex_agent.memory import MemoryQuery, MemoryRecord, MemoryStore
from codex_agent.rag_catalog import actions
from codex_agent.rag_catalog.api import ActionInput
from codex_agent.rag_catalog.store import CatalogStore
from codex_agent.storage.schema import memories, counters
from codex_agent.tasks.store import JobStore


helpers = runpy.run_path(str(Path(__file__).with_name("knowledge-fixture.py")))
Provider, NoChildren = helpers["Provider"], helpers["NoChildren"]


class MemoryProvider(Provider):
    # The memory worker also verifies the declared provider identity.
    name = "ollama"


def claim(jobs, expected_domain):
    event = jobs.claim_publication()
    assert event is not None, "Expected a durable index outbox event"
    jobs.publication_result(event)
    job = jobs.claim(jobs.envelope(event), "index")
    assert job and json.loads(job["payload_json"]).get("domain") == expected_domain
    return job


def count_version(index, identifier, version):
    conditions = [("workspace_id", index.scope.workspace_id), ("document_id", identifier),
        ("version", version), ("generation", index.scope.generation),
        ("index_fingerprint", index.scope.index_fingerprint)]
    result = index.transport.request("POST", "/points/count", {"filter": {"must": [
        {"key": key, "match": {"value": value}} for key, value in conditions]}, "exact": True})
    assert isinstance(result.get("count"), int)
    return result["count"]


def run():
    result = {"embedding": "deterministic synthetic vectors; no external model"}
    for name in ("TRITON_MYSQL_URL", "TRITON_AUTH_REDIS_URL", "TRITON_TEST_QDRANT_URL"):
        assert os.environ.get(name), "Missing disposable-service configuration: " + name
    with tempfile.TemporaryDirectory(prefix="rag-catalog-integration-") as directory:
        base = Path(directory)
        template = base / "template"
        template.mkdir()
        cfg = dict(schemaVersion=1, repoRoot=str(template), stateDir=str(base / "state"),
            memory=dict(retrievalMode="qdrant"), indexing=dict(enabled=True, generation="rag-fixture-v1",
            modelRevision="rag-admin-fixture-v1", dimensions=4,
            embedding=dict(provider="ollama", model="fixture", baseUrl="http://127.0.0.1:1", tokenBudget=180),
            qdrantUrl=os.environ["TRITON_TEST_QDRANT_URL"]))
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
        config = AccountConfig(origin="http://127.0.0.1:8766", data_dir=base / "accounts",
            template=template, allow_local_http=True, knowledge_enabled=True)
        accounts = Accounts(config)
        password = secrets.token_urlsafe(20)
        admin = accounts.register("ra_" + secrets.token_hex(4), password, accounts.invite())
        regular = accounts.register("ru_" + secrets.token_hex(4), password, accounts.invite())
        config = config.model_copy(update={"admin_user_ids": (admin["id"],)})
        own_root = config.data_dir / "workspaces" / admin["id"]
        denied_root = config.data_dir / "workspaces" / regular["id"]
        # Seed two genuine account-bound scopes, including their disk marker.
        for user, path in ((admin, own_root), (regular, denied_root)):
            path.mkdir(parents=True)
            (path / MARKER).write_text(user["id"])
            with accounts.engine.begin() as conn:
                conn.execute(account_workspaces.insert().values(
                    workspace_id=hashlib.sha256(str(path.resolve()).encode()).hexdigest(),
                    root=str(path.resolve()), user_id=user["id"], status="ready"))
            with admin_workspace(path, admin["id"]), MemoryStore(path) as memory:
                mid, created = memory.add(MemoryRecord(memory_type="failure-diagnosis", operator="Atlas",
                    semantics="Atlas input shape", pytorch_reference="reference(x)", summary="Atlas check shape",
                    outcome="failed", confidence_grade="C", source_run="fixture-run", failure_stage="compile",
                    evidence={"summary": "Synthetic local case"}))
                assert created
                if user["id"] == admin["id"]:
                    memory_id = mid
                    memory_jobs = JobStore(path)
                    task = claim(memory_jobs, None)
                    execute_index(memory_jobs, task, task["owner"], provider_factory=lambda _: MemoryProvider())
                    IndexStore(path).activate("rag-fixture-v1")
        query = MemoryQuery(operator="Atlas", semantics="Atlas input shape", pytorch_reference="reference(x)")
        with admin_workspace(own_root, admin["id"]), MemoryStore(own_root) as memory:
            assert indexed_retrieve(memory, query, provider_factory=lambda _: MemoryProvider())
        root = config.data_dir / "knowledge"
        store, jobs = KnowledgeStore(root), JobStore(root)
        gateway = create_gateway(config, accounts=accounts, workbenches=NoChildren())
        def auth(user):
            token, data = accounts.login(user["username"], password)
            return {"Origin": config.origin, "Cookie": config.cookie_name + "=" + token,
                    "X-CSRF-Token": data["csrf_token"], "Content-Type": "application/json"}
        ah, uh = auth(admin), auth(regular)
        with TestClient(gateway, base_url=config.origin) as client:
            def get(path, **params):
                response = client.get(path, headers=ah, params=params)
                assert response.status_code in {200, 201}, (path, response.status_code, response.text)
                return response.json()
            def post(path, body):
                response = client.post(path, headers=ah, json=body)
                expected = 201 if path == "/api/admin/knowledge" else 202 if path.endswith("/index") else 200
                assert response.status_code == expected, (path, response.status_code, response.text)
                return response.json()
            kp, cp = "/api/admin/knowledge", "/api/admin/rag/catalog"
            assert client.get(cp).status_code == 401
            assert client.get(cp, headers=uh).status_code == 403
            first = post(kp, dict(title="Atlas retention", source="Synthetic fixture",
                content="Atlas snapshots are retained for 37 days.", request_id="doc-v1"))
            identifier = first["id"]
            post(kp + f"/{identifier}/index", dict(version=1, request_id="index-v1"))
            task = claim(jobs, "knowledge")
            execute_knowledge_index(jobs, task, task["owner"], provider_factory=lambda _: Provider())
            post(kp + f"/{identifier}/publish", dict(version=1))
            post(kp, dict(document_id=identifier, expected_version=1, title="Atlas retention",
                source="Synthetic fixture", content="Atlas snapshots now last 90 days.", request_id="doc-v2"))
            catalog = get(cp)
            assert catalog["complete"] is True and catalog["total"] == 1
            assert len(catalog["scopes"]) == 1 and catalog["scopes"][0]["kind"] == "knowledge_document"
            doc = next(r for r in catalog["items"] if r["source_kind"] == "knowledge_document")
            assert get(cp, source_kind="memory_case")["items"] == []
            for path in (own_root, denied_root):
                rejected = client.get(cp, headers=ah, params={"scope_id": hashlib.sha256(str(path.resolve()).encode()).hexdigest()})
                assert rejected.status_code == 422, (rejected.status_code, rejected.text)
            # Exercise the internal memory lifecycle separately; it is not an admin-catalog scope.
            own_catalog = CatalogStore([dict(id=hashlib.sha256(str(own_root.resolve()).encode()).hexdigest(),
                root=own_root, label="Private fixture memory", kind="memory_case", writable=True)],
                cursor_key=secrets.token_bytes(32), database_engine=accounts.engine, memory_mode="qdrant")
            case = own_catalog.catalog()["items"][0]
            assert case["scope"]["id"] == hashlib.sha256(str(own_root.resolve()).encode()).hexdigest()
            assert case["source_id"] == str(memory_id)
            assert case["retrieval_state"] == "eligible"
            assert doc["latest_version"] == "2" and doc["serving_version"] == "1"
            detail = get(cp + "/" + doc["item_id"])
            assert detail["selected_version"] == "1" and "37 days" in detail["content"]
            assert "90 days" in get(cp + "/" + doc["item_id"], version="2")["content"]
            old_chunks = get(cp + "/" + doc["item_id"] + "/chunks", version="1")
            assert any("37 days" in c["text"] for c in old_chunks["items"])
            assert len(get(cp + "/" + doc["item_id"] + "/versions")["items"]) == 2
            live = get(cp + "/" + doc["item_id"] + "/vectors", version="1")
            assert live["vector_scan_performed"] and live["checks"][0]["state"] == "present"
            assert live["checks"][0]["actual_count"] > 0
            assert get(cp + "/" + doc["item_id"] + "/vectors", version="2")["checks"] == []
            assert client.get(cp + "/" + doc["item_id"] + "/vectors", headers=uh, params={"version": "1"}).status_code == 403
            assert client.get(cp + "/" + case["item_id"] + "/vectors", headers=ah, params={"version": case["latest_version"]}).status_code == 404
            assert get(cp, origin="uploaded")["total"] == 1
            assert get(cp, origin="history:sample")["total"] == 0
            with store.writer().transaction() as conn:
                conn.execute(source_links.insert().values(workspace_id=store.scope, origin_key="a" * 64,
                    source_alias="sample", source_scope="b" * 64, source_kind="memory_case", source_id="1",
                    source_version="v1", source_revision="r1", source_content_hash="c" * 64,
                    document_id=identifier, actor_id=admin["id"], created_at="2026-11-18T12:00:00Z"))
            assert get(cp, origin="uploaded")["total"] == 0
            assert get(cp, origin="history:sample")["total"] == 1
            assert get(cp + "/summary", origin="history:sample")["total"] == 1
            from codex_agent.rag_catalog import vectors as probes
            def offline(*_args, **_kw):
                raise probes.httpx.ConnectError("simulated outage")
            with patch.object(probes, "httpx", SimpleNamespace(Client=offline, HTTPError=probes.httpx.HTTPError)):
                offline_result = get(cp + "/" + doc["item_id"] + "/vectors", version="1")
            assert offline_result["checks"][0]["state"] == "unavailable"
            assert offline_result["checks"][0]["actual_count"] is None
            result["vectors"] = dict(actual_count_verified=True, draft_not_claimed_indexed=True,
                private_and_member_denied=True, outage_not_missing=True, origin_filter_matches_summary=True)
            result["catalog"] = {"authorized_sources": 1, "private_scope_excluded": True,
                                 "admin_private_scope_excluded": True,
                                 "serving_v1_draft_v2": True, "historical_chunks_readable": True}

            # Correction stays separate, versioned, pending review, and idempotent.
            with admin_workspace(own_root, admin["id"]), MemoryStore(own_root) as memory:
                before = memory.db.rows(memories, memories.c.id == memory_id)[0]
                memory_cache_before = memory.db.revision("memory")
            correction = dict(action="correct", expected_revision=case["revision"], request_id="case-note",
                              content="Synthetic correction pending review")
            for _ in range(2):
                actions.execute(own_catalog, case["item_id"], ActionInput(**correction), admin["id"])
            corrected = own_catalog.detail(case["item_id"])
            corrected["corrections"] = actions.corrections(own_catalog, case["item_id"])
            assert len(corrected["corrections"]) == 1 and corrected["corrections"][0]["state"] == "pending_review"
            with admin_workspace(own_root, admin["id"]), MemoryStore(own_root) as memory:
                assert memory.db.rows(memories, memories.c.id == memory_id)[0] == before
                assert memory.db.revision("memory") == memory_cache_before
            receipt = actions.operation(own_catalog, admin["id"], "case-note", case["item_id"])
            assert receipt["status"] == "committed" and receipt["item_id"] == case["item_id"]
            actions.execute(own_catalog, case["item_id"], ActionInput(action="archive",
                expected_revision=corrected["revision"], request_id="case-archive", reason="Synthetic exclusion"), admin["id"])
            archived = own_catalog.detail(case["item_id"])
            assert archived["retrieval_state"] == "excluded" and archived["serving_version"] is None
            # Pending removal vectors still exist, but authoritative SQL already rejects them.
            with admin_workspace(own_root, admin["id"]), MemoryStore(own_root) as memory:
                assert indexed_retrieve(memory, query, provider_factory=lambda _: MemoryProvider()) == []
                assert memory.db.revision("memory") > memory_cache_before
            result["memory"] = {"correction_preserves_original": True, "one_note_after_retry": True,
                "receipt_committed": True, "archive_excludes_before_worker": True, "cache_invalidated": True}

            # Index draft v2 without publication; both stored vector versions must be cleaned.
            post(kp + f"/{identifier}/index", dict(version=2, request_id="index-v2"))
            task = claim(jobs, "knowledge")
            execute_knowledge_index(jobs, task, task["owner"], provider_factory=lambda _: Provider())
            index = vector_index(store.scope, index_definition())
            assert count_version(index, identifier, 1) > 0 and count_version(index, identifier, 2) > 0
            deleted = post(kp + f"/{identifier}/delete", dict(version=2, catalog_revision=0))
            assert deleted["published_version"] is None and deleted["catalog_revision"] == 1
            assert search(root, "Atlas snapshot", provider_factory=lambda _: Provider())["items"] == []
            for _ in range(3):
                task = claim(jobs, "knowledge-cleanup")
                execute_cleanup(jobs, task, task["owner"])
                assert jobs.get(task["id"])["status"] == "succeeded"
                with store.engine.connect() as conn:
                    status = cleanup_status(store, conn, identifier)
                if status["state"] == "complete":
                    break
            assert status["state"] == "complete" and status["verified_at"]
            assert count_version(index, identifier, 1) == count_version(index, identifier, 2) == 0
            absent = get(cp + "/" + doc["item_id"] + "/vectors", version="1")
            assert absent["checks"][0]["actual_count"] == 0 and absent["checks"][0]["state"] == "missing"
            result["vectors"]["deleted_vectors_read_back"] = True
            with store.engine.connect() as conn:
                retained = store.rows(conn, versions, versions.c.document_id == identifier)
            assert len(retained) == 2 and all(v["state"] == "stale" for v in retained)
            assert any("37 days" in v["content"] for v in retained)
            restored = post(kp + f"/{identifier}/restore", dict(version=2, catalog_revision=1))
            assert restored["published_version"] is None and restored["state"] == "stale"
            unsafe = client.post(kp + f"/{identifier}/index", headers=ah,
                json=dict(version=2, catalog_revision=2, request_id="unsafe-reuse"))
            assert unsafe.status_code == 409
            post(kp, dict(document_id=identifier, expected_version=2, catalog_revision=2,
                title="Atlas retention", source="Synthetic fixture", content="Atlas snapshots now last 90 days.",
                request_id="doc-v3"))
            post(kp + f"/{identifier}/index", dict(version=3, catalog_revision=2, request_id="index-v3"))
            task = claim(jobs, "knowledge")
            execute_knowledge_index(jobs, task, task["owner"], provider_factory=lambda _: Provider())
            post(kp + f"/{identifier}/publish", dict(version=3, catalog_revision=2))
            recovered = search(root, "Atlas snapshot", provider_factory=lambda _: Provider())
            assert recovered["items"] and all(row["version"] == 3 for row in recovered["items"])
            assert "90 days" in recovered["context"] and count_version(index, identifier, 3) > 0
            result["cleanup"] = {"verified_old_point_count": 0, "versions_processed": 2,
                "originals_retained": True, "stale_version_reuse_rejected": True,
                "restored_new_v3_retrievable": True}
            assert client.get(cp, headers=uh).status_code == 403
        result["passed"] = True
        return result


if __name__ == "__main__":
    # No subprocess workers or broker: the fixture owns each bounded SQL lease.
    with patch.object(KnowledgeWorkers, "start", lambda self: None), patch.object(KnowledgeWorkers, "close", lambda self: None):
        print(json.dumps(run(), ensure_ascii=False))
