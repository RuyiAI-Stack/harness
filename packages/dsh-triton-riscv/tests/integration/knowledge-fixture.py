"""Production-route probes against disposable services; Vitest owns assertions."""
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import select

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.store import Accounts
from codex_agent.accounts.workspaces import Workbenches
from codex_agent.harness import HarnessAgent, HarnessSettings
from codex_agent.harness.runtime import HarnessRunResult
from codex_agent.knowledge.store import DocumentInput, KnowledgeStore
from codex_agent.knowledge.vectors import (ROOT_ENV, augment_chat, execute_knowledge_index,
                                          index_definition, search, vector_index)
from codex_agent.knowledge.schema import documents, versions, audit
from codex_agent.knowledge.processes import KnowledgeWorkers
from codex_agent.memory_chunking import ChunkPolicy, split_chunk_text
from codex_agent.platform.executor import HarnessRunExecutor
from codex_agent.platform.service import PlatformService
from codex_agent.platform.store import PlatformStore
from codex_agent.tasks.store import JobStore, digest, encode
from codex_agent.messaging.broker import Broker
from codex_agent.messaging.relay import publish_once


class NoChildren:
    def start(self): pass
    def close(self): pass
    async def monitor(self):
        while True:
            await asyncio.sleep(60)
    def get(self, _uid):
        raise RuntimeError("Probe must not access a private child service")


class Provider:
    name = "fixture"
    model = "fixture"
    max_input_tokens = 180
    def count_tokens(self, text): return len(text)
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] if "Atlas" in t or "snapshot" in t
                else [0.0, 1.0, 0.0, 0.0] for t in texts]


class CapturingBackend:
    prompt = ""
    def run(self, prompt, session_id=None, on_event=None):
        self.prompt = prompt
        return HarnessRunResult(session_id="fixture", finish_reason="completed", final_response="Fixture model response [K1]")
    def close(self): pass


def run():
    result = {}
    with tempfile.TemporaryDirectory(prefix="knowledge-probe-") as directory:
        base = Path(directory)
        template = base / "template"
        template.mkdir()
        config = AccountConfig(origin="http://127.0.0.1:8766", data_dir=base / "accounts",
                               template=template, allow_local_http=True, knowledge_enabled=True)
        definition = {"enabled": True, "modelRevision": "probe-v1", "dimensions": 4,
            "embedding": {"provider": "ollama", "model": "fixture", "baseUrl": "http://127.0.0.1:1",
                          "tokenBudget": 180}, "qdrantUrl": os.environ["TRITON_TEST_QDRANT_URL"]}
        cfg = dict(schemaVersion=1, repoRoot=str(template), stateDir=str(base / "unused"), indexing=definition)
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
        accounts = Accounts(config)
        password = secrets.token_urlsafe(20)
        admin = accounts.register("ka_" + secrets.token_hex(4), password, accounts.invite())
        regular = accounts.register("ku_" + secrets.token_hex(4), password, accounts.invite())
        config = config.model_copy(update={"admin_user_ids": (admin["id"],)})
        gateway = create_gateway(config, accounts=accounts, workbenches=NoChildren())
        root = config.data_dir / "knowledge"
        root.mkdir(parents=True)
        store = KnowledgeStore(root)
        provider_factory = lambda _d: Provider()
        headers = {"Origin": config.origin, "Content-Type": "application/json"}
        with TestClient(gateway, base_url=config.origin) as client:
            result["anonymous"] = client.get("/api/admin/overview").status_code
            token_u, u = accounts.login(regular["username"], password)
            user_headers = {**headers, "Cookie": config.cookie_name + "=" + token_u, "X-CSRF-Token": u["csrf_token"]}
            result["regular_admin"] = client.get("/api/admin/overview", headers={**user_headers, "X-Role": "admin"}).status_code
            result["regular_write"] = client.post("/api/admin/knowledge", headers=user_headers, json={}).status_code
            result["regular_flag"] = client.get("/api/auth/me", headers=user_headers).json()["user"]["is_admin"]
            token_a, a = accounts.login(admin["username"], password)
            admin_headers = {**headers, "Cookie": config.cookie_name + "=" + token_a, "X-CSRF-Token": a["csrf_token"]}
            result["admin_flag"] = client.get("/api/auth/me", headers=admin_headers).json()["user"]["is_admin"]
            result["missing_csrf"] = client.post("/api/admin/knowledge", headers={**headers, "Cookie": admin_headers["Cookie"]}, json={}).status_code
            result["bad_origin"] = client.post("/api/admin/knowledge", headers={**admin_headers, "Origin": "https://other.invalid"}, json={}).status_code
            payload = dict(title="Atlas snapshot retention", source="fixture manual", content="Atlas snapshots are retained for 37 days. The recovery code is SILVER-271.", request_id="first")
            saved = client.post("/api/admin/knowledge", headers=admin_headers, json=payload)
            result["save_status"] = saved.status_code
            first = saved.json()
            identifier = first["id"]
            result["draft"] = first
            result["repeat_save_same_id"] = client.post("/api/admin/knowledge", headers=admin_headers, json=payload).json()["id"] == identifier
            result["request_conflict"] = client.post("/api/admin/knowledge", headers=admin_headers, json={**payload, "content": "changed"}).status_code
            result["early_publish"] = client.post(f"/api/admin/knowledge/{identifier}/publish", headers=admin_headers, json={"version": 1}).status_code
            result["draft_hidden"] = client.get(f"/api/knowledge/documents/{identifier}?version=1", headers=user_headers).status_code
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["draft_search"] = client.post("/api/knowledge/search", headers=user_headers, json={"query": "Atlas snapshot"}).json()["status"]
            indexed = client.post(f"/api/admin/knowledge/{identifier}/index", headers=admin_headers, json={"version": 1, "request_id": "index-1"})
            result["index_accepted"] = indexed.status_code
            client.post(f"/api/admin/knowledge/{identifier}/index", headers=admin_headers, json={"version": 1, "request_id": "index-1"})
            jobs = JobStore(root)
            # Exercise real publisher confirms + broker delivery before claiming the job.
            async def delivery():
                async with Broker(jobs) as broker:
                    await publish_once(jobs, broker)
                    message = await broker.queues["index"].get(timeout=10)
                    envelope = json.loads(message.body)
                    await message.ack()
                    return envelope
            envelope = asyncio.run(delivery())
            job = jobs.claim(envelope, "index")
            execute_knowledge_index(jobs, job, job["owner"], provider_factory=provider_factory)
            result["indexed"] = store.detail(identifier)["state"]
            result["duplicate_claim"] = jobs.claim(envelope, "index") is None
            result["one_outbox"] = jobs.claim_publication() is None
            result["published"] = client.post(f"/api/admin/knowledge/{identifier}/publish", headers=admin_headers, json={"version": 1}).json()["published_version"]
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                hit = client.post("/api/knowledge/search", headers=user_headers, json={"query": "Atlas snapshot retention"}).json()
                result["retrieval"] = {"status": hit["status"], "text": hit["context"], "count": len(hit["items"])}
                with patch("codex_agent.knowledge.vectors.cached_provider", side_effect=RuntimeError("offline")):
                    result["unavailable_status"] = client.post("/api/knowledge/search", headers=user_headers,
                        json={"query": "Atlas snapshot"}).status_code
                    os.environ[ROOT_ENV] = str(root)
                    fallback, state = augment_chat("original task", "Atlas snapshot")
                    result["unavailable_chat"] = state["status"] == "unavailable" and fallback.startswith("original task")
                    os.environ.pop(ROOT_ENV, None)
                # Run the real conversation executor and capture the model boundary.
                chat_root = base / "chat"
                chat_root.mkdir()
                os.environ[ROOT_ENV] = str(root)
                platform = PlatformStore(chat_root)
                backend = CapturingBackend()
                settings = replace(HarnessSettings.from_env(chat_root), managed_context=False)
                agent = HarnessAgent(settings, backend=backend)
                service = PlatformService(chat_root, platform, agent=agent)
                session = service.create_session()
                run_id = service.handle_message(session["id"], "Atlas snapshot retention?")["run"]["id"]
                executor = HarnessRunExecutor(chat_root, platform, agent, workers=0)
                executor._execute(run_id)
                result["model_boundary"] = {"has_evidence": "SILVER-271" in backend.prompt,
                    "untrusted_marked": "untrusted reference data" in backend.prompt,
                    "completed": platform.get_run(run_id)["status"] == "completed"}
                executor.close()
                os.environ.pop(ROOT_ENV, None)
            update = {**payload, "content": "Atlas snapshots now last 90 days.", "request_id": "second",
                      "document_id": identifier, "expected_version": 1}
            result["update_version"] = client.post("/api/admin/knowledge", headers=admin_headers, json=update).json()["latest_version"]
            result["stale_replay"] = client.post("/api/admin/knowledge", headers=admin_headers, json=payload).status_code
            result["stale_update"] = client.post("/api/admin/knowledge", headers=admin_headers, json={**update, "request_id": "stale"}).status_code
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["old_publication_preserved"] = "37 days" in search(root, "Atlas snapshot")["context"]
            client.post(f"/api/admin/knowledge/{identifier}/index", headers=admin_headers, json={"version": 2, "request_id": "index-2"})
            event = jobs.claim_publication()
            job2 = jobs.claim(jobs.envelope(event), "index")
            class Broken(Provider):
                def embed(self, _texts): raise RuntimeError("fixture failure, never publish")
            try:
                execute_knowledge_index(jobs, job2, job2["owner"], provider_factory=lambda _d: Broken())
            except RuntimeError:
                jobs.finish(job2["id"], job2["owner"], "failed", {"reason": "fixture"})
            result["failed_draft"] = store.detail(identifier)["state"]
            result["failed_publish"] = client.post(f"/api/admin/knowledge/{identifier}/publish", headers=admin_headers, json={"version": 2}).status_code
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["failure_preserves_old"] = "37 days" in search(root, "Atlas snapshot")["context"]
            result["unpublish"] = client.post(f"/api/admin/knowledge/{identifier}/unpublish", headers=admin_headers, json={}).status_code
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["unpublished_search"] = search(root, "Atlas snapshot")["status"]
            result["unpublished_document"] = client.get(f"/api/knowledge/documents/{identifier}?version=1", headers=user_headers).status_code
            result["overview"] = client.get("/api/admin/overview", headers=admin_headers).json()
            result["overview_no_secrets"] = all(key not in json.dumps(result["overview"]) for key in (password, "password_hash", "csrf_token"))
            long = "中文资料无空格" * 150
            parts = split_chunk_text(long, token_count=len, policy=ChunkPolicy(max_tokens=180, overlap_tokens=24))
            result["unicode_chunks"] = len(parts) > 1 and all(len(p) <= 180 for p in parts)
            result["short_chunk"] = split_chunk_text("A complete short paragraph.") == ["A complete short paragraph."]
            with store.engine.connect() as conn:
                result["audited"] = len(store.rows(conn, audit)) >= 5
                result["versions_retained"] = len(store.rows(conn, versions, versions.c.document_id == identifier)) == 2
            # Another trusted catalog is still a different vector/SQL scope.
            other = KnowledgeStore(base / "other")
            other.writer()
            result["catalog_isolation"] = search(base / "other", "Atlas snapshot", provider_factory=provider_factory)["status"] == "empty"
            result["oversize"] = client.post("/api/admin/knowledge", headers=admin_headers,
                json={**payload, "request_id": "oversize", "content": "x" * 60001}).status_code
            accounts.set_status(admin["username"], False)
            result["disabled_admin"] = client.get("/api/admin/overview", headers=admin_headers).status_code
        # The real worker handler must reschedule a transient failure, then reuse
        # the same immutable source on the next lease without duplicating chunks.
        retry_store = KnowledgeStore(base / "retry")
        retry_doc = retry_store.save(DocumentInput(**{**payload, "request_id": "retry"}), "fixture")
        retry_store.enqueue(retry_doc["id"], 1, "retry-index", "fixture")
        retry_jobs = JobStore(base / "retry")
        event = retry_jobs.claim_publication()
        retry_jobs.publication_result(event)
        attempt = retry_jobs.claim(retry_jobs.envelope(event), "index")
        from codex_agent.workers.handlers import execute_job
        with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Broken()):
            execute_job(retry_store.root, attempt["id"], attempt["owner"])
        queued = retry_jobs.get(attempt["id"])["status"] == "queued"
        event = None
        for _ in range(60):
            event = retry_jobs.claim_publication()
            if event:
                break
            time.sleep(.1)
        if not event:
            raise RuntimeError("Index retry was not scheduled")
        retry_jobs.publication_result(event)
        attempt2 = retry_jobs.claim(retry_jobs.envelope(event), "index")
        with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
            execute_job(retry_store.root, attempt2["id"], attempt2["owner"])
        result["index_retry"] = queued and attempt2["id"] == attempt["id"] and retry_store.detail(retry_doc["id"])["state"] == "ready"
        retry_store.publish(retry_doc["id"], 1, "fixture")
        changed = {**cfg, "indexing": {**definition, "generation": "v2"}}
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(changed)
        try:
            search(retry_store.root, "Atlas snapshot", provider_factory=provider_factory)
        except ValueError:
            result["configuration_mismatch"] = True
        finally:
            os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
        # Local real embeddings: a functional integration test, not a quality benchmark.
        endpoint = os.environ.get("TRITON_TEST_EMBEDDING_URL")
        if endpoint:
            definition.update(dimensions=384, modelRevision="local-all-minilm-v2",
                embedding={"provider": "ollama", "model": "all-minilm:v2", "baseUrl": endpoint, "tokenBudget": 180})
            os.environ["TRITON_RISCV_CONFIG"] = json.dumps({**cfg, "indexing": definition})
            real_config = config.model_copy(update={"data_dir": base / "real"})
            real_root = real_config.data_dir / "knowledge"
            real_root.mkdir(parents=True)
            real = KnowledgeStore(real_root)
            doc = real.save(DocumentInput(title="Atlas backup policy", source="synthetic acceptance document",
                content="Atlas database snapshots are retained for 37 days. Recovery requires the code SILVER-271.", request_id="real"), "fixture")
            real.enqueue(doc["id"], 1, "real-index", "fixture")
            workers = KnowledgeWorkers(real_config)
            try:
                workers.start()
                for _ in range(240):
                    status = real.detail(doc["id"])["state"]
                    if status in {"ready", "failed"}:
                        break
                    time.sleep(.25)
                if status != "ready":
                    raise RuntimeError(f"Background index did not complete: {status}")
            finally:
                workers.close()
            real.publish(doc["id"], 1, "fixture")
            found = search(real_root, "How long are Atlas database backups retained?")
            result["real_embedding"] = {"status": found["status"], "count": len(found["items"]),
                "has_evidence": "37 days" in found["context"], "model": "all-minilm:v2", "dimensions": 384}
        return result


if __name__ == "__main__":
    print(json.dumps(run()))
