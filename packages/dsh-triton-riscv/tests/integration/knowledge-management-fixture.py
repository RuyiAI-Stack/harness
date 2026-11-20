"""Management route probes on disposable services; assertions live in Vitest."""
import json
import os
from pathlib import Path
import runpy
import secrets
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.store import Accounts
from codex_agent.knowledge.store import KnowledgeStore, DocumentInput
from codex_agent.knowledge.schema import versions, audit
from codex_agent.knowledge.vectors import execute_knowledge_index, search, vector_index
from codex_agent.knowledge.files import upload
from codex_agent.knowledge.processes import KnowledgeWorkers
from codex_agent.tasks.store import JobStore
from codex_agent.workers.handlers import execute_job

helpers = runpy.run_path(str(Path(__file__).with_name("knowledge-fixture.py")))
Provider, NoChildren = helpers["Provider"], helpers["NoChildren"]
pdf_bytes = runpy.run_path(str(Path(__file__).with_name("knowledge-files-fixture.py")))["pdf_bytes"]


def run():
    result = {}
    with tempfile.TemporaryDirectory(prefix="knowledge-management-") as directory:
        base = Path(directory)
        template = base / "template"
        template.mkdir()
        config = AccountConfig(origin="http://127.0.0.1:8766", data_dir=base / "accounts",
                               template=template, allow_local_http=True, knowledge_enabled=True)
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(dict(schemaVersion=1, repoRoot=str(template),
            stateDir=str(base / "state"), indexing=dict(enabled=True, modelRevision="management-v1", dimensions=4,
            embedding=dict(provider="ollama", model="fixture", baseUrl="http://127.0.0.1:1", tokenBudget=180),
            qdrantUrl=os.environ["TRITON_TEST_QDRANT_URL"])))
        accounts = Accounts(config)
        password = secrets.token_urlsafe(20)
        admin = accounts.register("ma_" + secrets.token_hex(4), password, accounts.invite())
        regular = accounts.register("mu_" + secrets.token_hex(4), password, accounts.invite())
        config = config.model_copy(update={"admin_user_ids": (admin["id"],)})
        root = config.data_dir / "knowledge"
        store, jobs = KnowledgeStore(root), JobStore(root)
        gateway = create_gateway(config, accounts=accounts, workbenches=NoChildren())
        def headers(user):
            token, data = accounts.login(user["username"], password)
            return {"Origin": config.origin, "Cookie": config.cookie_name + "=" + token,
                    "X-CSRF-Token": data["csrf_token"], "Content-Type": "application/json"}
        ah, uh = headers(admin), headers(regular)
        def claim():
            # A previous soft delete can enqueue cleanup ahead of the next
            # document index. Exercise that real handler instead of interpreting
            # the cleanup payload as an index, or silently discarding the task.
            for _ in range(20):
                event = jobs.claim_publication()
                assert event is not None, "Expected the next document index task"
                jobs.publication_result(event)
                job = jobs.claim(jobs.envelope(event), "index")
                if job is None:  # A cancelled task may retain an outbox delivery.
                    continue
                domain = json.loads(job["payload_json"]).get("domain")
                if domain == "knowledge":
                    return job
                assert domain == "knowledge-cleanup", f"Unexpected queued domain: {domain}"
                execute_job(root, job["id"], job["owner"])
                state = jobs.get(job["id"])["status"]
                assert state in {"succeeded", "cancelled"}, f"Cleanup did not settle: {state}"
            raise AssertionError("Cleanup drain exceeded its bounded task budget")
        with TestClient(gateway, base_url=config.origin) as client:
            def post(path, data, h=ah):
                return client.post("/api/admin/knowledge" + path, headers=h, json=data)
            def catalog(**params):
                return client.get("/api/admin/knowledge", params=params, headers=ah).json()
            def save(title, content="Atlas retention is 37 days.", **extra):
                return post("", dict(title=title, content=content, source="manual Atlas", request_id=secrets.token_hex(12), **extra)).json()
            doc = save("Atlas Guide 100%")
            identifier = doc["id"]
            second = save("中文手册")
            post(f"/{identifier}/index", dict(version=1, request_id="index"))
            job = claim()
            execute_knowledge_index(jobs, job, job["owner"], provider_factory=lambda _d: Provider())
            post(f"/{identifier}/publish", dict(version=1))
            result["catalog"] = dict(total=catalog()["total"], chinese=catalog(q="中文")["total"],
                source=catalog(q="manual atlas")["total"], literal=catalog(q="%")["total"],
                published=catalog(status="published")["total"], unpublished=catalog(status="unpublished")["total"],
                next_page=catalog(offset=50)["documents"])
            save("Atlas Guide 100%", content="Atlas retention is now 90 days.",
                           document_id=identifier, expected_version=1)
            result["stale_delete"] = post(f"/{identifier}/delete", dict(version=1, catalog_revision=0)).status_code
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["before_delete"] = search(root, "Atlas retention")["context"]
                result["regular_delete"] = post(f"/{identifier}/delete", dict(version=2, catalog_revision=0), uh).status_code
                result["regular_list"] = client.get("/api/admin/knowledge", headers=uh).status_code
                result["missing_csrf"] = post(f"/{identifier}/delete", dict(version=2), {k:v for k,v in ah.items() if k != "X-CSRF-Token"}).status_code
                deleted = post(f"/{identifier}/delete", dict(version=2, catalog_revision=0)).json()
                result["delete"] = {k:deleted[k] for k in ("state", "published_version", "catalog_revision")}
                result["repeat_delete"] = post(f"/{identifier}/delete", dict(version=2, catalog_revision=0)).status_code
                result["after_delete"] = search(root, "Atlas retention")["items"]
            result["active_count"] = catalog()["total"]
            result["trash_count"] = catalog(view="trash")["total"]
            result["deleted_publish"] = post(f"/{identifier}/publish", dict(version=1)).status_code
            result["deleted_save"] = post("", dict(title="bad", content="bad", request_id="deleted",
                document_id=identifier, expected_version=2)).status_code
            result["deleted_public_read"] = client.get(f"/api/knowledge/documents/{identifier}?version=1", headers=uh).status_code
            detail = client.get(f"/api/admin/knowledge/{identifier}", headers=ah).json()
            result["preserved"] = len(detail["versions"]) == 2 and "90 days" in detail["content"]
            restored = post(f"/{identifier}/restore", dict(version=2, catalog_revision=1)).json()
            result["restore"] = {k:restored[k] for k in ("state", "published_version", "catalog_revision")}
            result["repeat_restore"] = post(f"/{identifier}/restore", dict(version=2, catalog_revision=1)).status_code
            result["old_delete"] = post(f"/{identifier}/delete", dict(version=2, catalog_revision=0)).status_code
            result["old_publish"] = post(f"/{identifier}/publish", dict(version=1, catalog_revision=0)).status_code
            result["republish"] = post(f"/{identifier}/publish", dict(version=1, catalog_revision=2)).status_code
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["restored_search"] = "37 days" in search(root, "Atlas retention")["context"]
            # Stale vector hits remain in Qdrant but must not hydrate deleted SQL evidence.
            post(f"/{second['id']}/index", dict(version=1, request_id="second"))
            job = claim()
            execute_knowledge_index(jobs, job, job["owner"], provider_factory=lambda _d: Provider())
            post(f"/{second['id']}/publish", dict(version=1))
            real_index = None
            def deleting_index(scope, definition):
                nonlocal real_index
                real_index = vector_index(scope, definition)
                original = real_index.search
                def stale_hits(*args, **kwargs):
                    hits = original(*args, **kwargs)
                    store.set_deleted(identifier, 2, 2, admin["id"], deleted=True)
                    real_index.search = original
                    return hits
                real_index.search = stale_hits
                return real_index
            outcome = search(root, "Atlas retention", provider_factory=lambda _d: Provider(), index_factory=deleting_index)
            result["stale_vectors_filtered"] = all(item["document_id"] != identifier for item in outcome["items"])
            result["other_document_survives"] = any(item["document_id"] == second["id"] for item in outcome["items"])
            # Cancel a queued job without affecting other documents.
            queued = save("queued document")
            store.enqueue(queued["id"], 1, "queued", admin["id"])
            with store.engine.connect() as conn:
                queued_job = store.row(conn, versions, queued["id"], 1)["job_id"]
            store.set_deleted(queued["id"], 1, 0, admin["id"], deleted=True)
            result["queued_cancelled"] = jobs.get(queued_job)["status"]
            # Running embedding/parse can finish computation, but not commit after deletion+restore.
            for mode in ("index", "parse"):
                isolated_root = base / mode
                isolated = KnowledgeStore(isolated_root)
                js = JobStore(isolated_root)
                if mode == "index":
                    d = isolated.save(DocumentInput(title="Atlas race", content="Atlas evidence", request_id="race"), "fixture")
                    isolated.enqueue(d["id"], 1, "race", "fixture")
                else:
                    d = upload(isolated, pdf_bytes(), "race.pdf", "Atlas race", "fixture", "race", "fixture")
                event = js.claim_publication()
                js.publication_result(event)
                running = js.claim(js.envelope(event), "index")
                def cycle():
                    isolated.set_deleted(d["id"], 1, 0, "fixture", deleted=True)
                    isolated.set_deleted(d["id"], 1, 1, "fixture", deleted=False)
                class RacingProvider(Provider):
                    def embed(self, texts):
                        cycle()
                        return super().embed(texts)
                def racing_parse(*_args):
                    cycle()
                    return dict(state="review", sections=[dict(id="s1", text="late content", locator="Page 1", edited=False)], warnings=[])
                with patch("codex_agent.knowledge.vectors.cached_provider", return_value=RacingProvider()), \
                     patch("codex_agent.knowledge.files.isolated_parse", side_effect=racing_parse):
                    execute_job(isolated_root, running["id"], running["owner"])
                result[mode + "_race"] = dict(job=js.get(running["id"])["status"],
                    state=isolated.detail(d["id"])["state"], content=isolated.detail(d["id"])["content"])
            with store.engine.connect() as conn:
                actions = [row["action"] for row in store.rows(conn, audit)]
                result["audit"] = "document-trashed" in actions and "document-restored" in actions
        return result


if __name__ == "__main__":
    with patch.object(KnowledgeWorkers, "start", lambda self: None), patch.object(KnowledgeWorkers, "close", lambda self: None):
        print(json.dumps(run(), ensure_ascii=False))
