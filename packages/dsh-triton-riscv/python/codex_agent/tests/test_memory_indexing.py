"""Index engineering fixtures, not real embeddings or retrieval efficacy claims."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from codex_agent.indexing.store import IndexStore
from codex_agent.indexing.worker import execute_index
from codex_agent.indexing.qdrant import Qdrant
from codex_agent.memory import MemoryStore, MemoryQuery
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.schema import (memories, memory_versions, memory_indexes as indexes,
    memory_index_entries as entries, memory_index_chunks as chunks)
from codex_agent.tasks.schema import jobs, outbox
from codex_agent.tasks.store import Conflict, LeaseLost
from codex_agent.tests.test_memory import record, FakeEmbeddingProvider
from codex_agent.workers.handlers import execute_job


class FixtureEmbedding(FakeEmbeddingProvider):
    # Deliberately artificial 4D vectors with exact character-count fixture tokenizer.
    name = "sentence-transformers"
    model = "synthetic-index-fixture-not-a-real-model"


class VectorFixture:
    def __init__(self):
        self.points = {}
        self.upserts = 0

    def ensure(self):
        pass

    def upsert(self, points):
        self.upserts += 1
        self.points.update({p["id"]: p for p in points})

    def delete_revision(self, workspace, memory_id, revision, *, older=False, include=False):
        for key, point in list(self.points.items()):
            p = point["payload"]
            match = p["revision"] <= revision if include else p["revision"] < revision if older else p["revision"] == revision
            if p["workspace_id"] == workspace and p["memory_id"] == memory_id and match:
                del self.points[key]


class Fixture:
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.cfg = dict(schemaVersion=1, repoRoot=str(self.root), stateDir=str(self.root / "agent-results"),
            queue={"enabled": False}, indexing={"enabled": True, "generation": "test-v1", "dimensions": 4,
                "modelRevision": "synthetic-fixture-1", "embedding": {"provider": FixtureEmbedding.name,
                    "model": FixtureEmbedding.model}, "qdrantUrl": os.getenv("TRITON_TEST_QDRANT_URL", "http://127.0.0.1:1")})
        env = patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg),
            "TRITON_AMQP_URL": os.getenv("TRITON_TEST_AMQP_URL", "amqp://guest:guest@127.0.0.1:1/")})
        env.start()
        self.addCleanup(env.stop)
        self.memory = MemoryStore(self.root)
        self.index = IndexStore(self.root)
        self.jobs = self.index.jobs
        self.vectors = VectorFixture()

    def configure(self):
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(self.cfg)

    def add(self, **kwargs):
        return self.memory.add(record(**kwargs))[0]

    def queued(self):
        return self.jobs.db.rows(jobs, jobs.c.kind == "index", order=(jobs.c.created_ms,))

    def entry(self, memory_id):
        return self.jobs.db.rows(entries, entries.c.memory_id == memory_id,
                                entries.c.generation == self.cfg["indexing"]["generation"])[0]

    def claim(self, job):
        event = self.jobs.db.rows(outbox, outbox.c.job_id == job["id"], order=(outbox.c.generation.desc(),))[0]
        return self.jobs.claim(self.jobs.envelope(event), "index")

    def run_job(self, job, *, provider=None, vectors=None):
        claimed = self.claim(job)
        execute_index(self.jobs, claimed, claimed["owner"],
                      provider_factory=lambda _: provider or FixtureEmbedding(),
                      vector_factory=lambda *_: vectors or self.vectors)
        return claimed


class IndexTests(Fixture, unittest.TestCase):
    def test_default_disabled_and_sync_task_can_trigger_index(self):
        self.assertFalse(runtime_config().queue.enabled)
        mid = self.add()
        self.assertEqual(self.entry(mid)["status"], "pending")
        self.assertEqual(len(self.queued()), 1)
        self.assertEqual(self.vectors.points, {})
        self.assertIsNone(self.memory.embedding_provider)
        self.cfg["indexing"]["enabled"] = False
        self.configure()
        self.add(source_run="another")
        self.assertEqual(len(self.queued()), 1)

    def test_case_job_and_outbox_rollback_together(self):
        original = self.jobs.__class__._enqueue
        def fail(*args):
            original(*args)
            raise RuntimeError("rollback fixture")
        with patch.object(self.jobs.__class__, "_enqueue", autospec=True, side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.add()
        for table in (memories, memory_versions, entries, jobs, outbox):
            self.assertEqual(self.jobs.db.rows(table), [])

    def test_unchanged_duplicate_and_access_do_not_enqueue(self):
        mid = self.add()
        self.add()
        self.memory.mark_useful(mid)
        self.assertEqual(len(self.queued()), 1)
        self.assertEqual(len(self.jobs.db.rows(outbox)), 1)

    def test_concurrent_duplicate_is_one_job(self):
        with ThreadPoolExecutor(4) as pool:
            ids = list(pool.map(lambda _: self.add(), range(8)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.queued()), 1)

    def test_vectors_link_exact_case_version_and_source(self):
        mid = self.add(evidence={"recommended_actions": ["try a different lowering"], "test_summary": "unverified fixture"})
        query = MemoryQuery(operator="tanh_and_mul", semantics="", pytorch_reference="")
        before = self.memory.retrieve(query)
        self.run_job(self.queued()[0])
        row = self.entry(mid)
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["point_count"], len(self.vectors.points))
        records = self.jobs.db.rows(chunks)
        self.assertEqual(len(records), len(self.vectors.points))
        self.assertTrue(all(p["payload"]["case_version"] == row["case_version"] for p in self.vectors.points.values()))
        provenance = [json.loads(r["provenance_json"]) for r in records]
        advice = next(p for p in provenance if p["source_field"].startswith("evidence.recommended_actions"))
        self.assertEqual(advice["source"]["state"], "not-executed")
        after = self.memory.retrieve(query)
        # Retrieval itself updates this access timestamp; scoring and evidence must not change.
        stable = lambda rows: [{k: v for k, v in r.items() if k != "last_used_at"} for r in rows]
        self.assertEqual(stable(before), stable(after))

    def test_old_job_cannot_replace_new_version(self):
        mid = self.add()
        first = self.queued()[0]
        self.add(evidence={"test_summary": "updated source fixture"})
        latest = self.entry(mid)
        self.assertEqual(latest["revision"], 2)
        self.run_job(self.queued()[-1])
        self.run_job(first)
        self.assertEqual(self.entry(mid)["status"], "ready")
        self.assertEqual({p["payload"]["revision"] for p in self.vectors.points.values()}, {2})
        self.assertEqual(self.jobs.get(first["id"])["result"]["index_status"], "superseded")

    def test_update_during_embedding_is_fenced(self):
        mid = self.add()
        provider = FixtureEmbedding()
        original = provider.embed
        def changed(texts):
            self.add(evidence={"test_summary": "new version during embedding"})
            return original(texts)
        provider.embed = changed
        self.run_job(self.queued()[0], provider=provider)
        self.assertEqual(self.entry(mid)["status"], "pending")
        self.assertEqual(self.vectors.points, {})

    def test_archive_and_reactivation_do_not_cross_delete(self):
        mid = self.add()
        self.run_job(self.queued()[0])
        self.memory.archive(mid, "fixture")
        deletion = self.queued()[-1]
        self.add()
        self.run_job(self.queued()[-1])
        self.run_job(deletion)
        self.assertEqual(self.entry(mid)["revision"], 3)
        self.assertEqual({p["payload"]["revision"] for p in self.vectors.points.values()}, {3})

    def test_archive_deletes_vectors_but_preserves_audit(self):
        mid = self.add()
        self.run_job(self.queued()[0])
        self.memory.archive(mid, "fixture")
        self.run_job(self.queued()[-1])
        self.assertEqual(self.vectors.points, {})
        self.assertEqual(self.entry(mid)["status"], "deleted")
        self.assertTrue(self.jobs.db.rows(chunks))
        count = len(self.queued())
        self.memory.archive(mid, "fixture again")
        self.assertEqual(len(self.queued()), count)

    def test_external_source_delete_is_reconciled_by_backfill(self):
        mid = self.add()
        self.run_job(self.queued()[0])
        with self.jobs.db.transaction() as conn:
            self.jobs.db.delete(conn, memories, memories.c.id == mid)
        self.assertEqual(self.index.backfill()["submitted"], 1)
        self.run_job(self.queued()[-1])
        self.assertEqual(self.vectors.points, {})

    def test_failed_worker_retries_bounded_without_losing_memory(self):
        mid = self.add()
        job = self.queued()[0]
        for _ in range(3):
            claimed = self.claim(job)
            with patch("codex_agent.indexing.worker.execute_index", side_effect=RuntimeError("secret not exposed")):
                execute_job(self.root, job["id"], claimed["owner"])
        final = self.jobs.get(job["id"])
        self.assertEqual(final["status"], "dead")
        self.assertEqual(final["attempt"], 3)
        self.assertNotIn("secret", json.dumps(final))
        self.assertIsNotNone(self.memory.get(mid))
        self.assertEqual(self.index.backfill(retry=True)["submitted"], 1)
        self.assertEqual(self.entry(mid)["revision"], 2)

    def test_effectful_lease_recovery_replays_index_but_fences_old_owner(self):
        self.add()
        job = self.queued()[0]
        old = self.claim(job)
        self.jobs.begin_effects(job["id"], old["owner"])
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, jobs, {"lease_until": 1}, jobs.c.id == job["id"])
        self.assertEqual(self.jobs.recover()[0]["status"], "queued")
        self.run_job(job)
        with self.assertRaises(LeaseLost):
            self.jobs.finish(job["id"], old["owner"], "succeeded", {})

    def test_partial_upsert_replay_is_idempotent(self):
        self.add()
        job = self.queued()[0]
        original = self.vectors.upsert
        def fail_after_write(points):
            original(points)
            raise RuntimeError("connection lost after server accepted batch")
        self.vectors.upsert = fail_after_write
        claimed = self.claim(job)
        with self.assertRaises(RuntimeError):
            execute_index(self.jobs, claimed, claimed["owner"], provider_factory=lambda _: FixtureEmbedding(),
                          vector_factory=lambda *_: self.vectors)
        self.jobs.retry_index(job["id"], claimed["owner"], {"fixture": "connection lost"})
        self.vectors.upsert = original
        self.run_job(job)
        self.assertEqual(len(self.vectors.points), len(self.jobs.db.rows(chunks)))

    def test_wrong_dimension_never_marks_ready(self):
        mid = self.add()
        provider = FixtureEmbedding()
        provider.embed = lambda texts: [[1.0, 2.0]] * len(texts)
        with self.assertRaises(ValueError):
            self.run_job(self.queued()[0], provider=provider)
        self.assertEqual(self.entry(mid)["status"], "pending")
        self.assertEqual(self.vectors.points, {})

    def test_vector_service_failure_never_marks_ready(self):
        mid = self.add()
        with patch.object(self.vectors, "ensure", side_effect=OSError("fixture service down")):
            with self.assertRaises(OSError):
                self.run_job(self.queued()[0])
        self.assertEqual(self.entry(mid)["status"], "pending")
        self.assertIsNotNone(self.memory.get(mid))

    def test_archive_during_embedding_removes_late_points(self):
        mid = self.add()
        provider = FixtureEmbedding()
        original = provider.embed
        def archive(texts):
            self.memory.archive(mid, "archived during embedding fixture")
            return original(texts)
        provider.embed = archive
        self.run_job(self.queued()[0], provider=provider)
        self.assertEqual(self.vectors.points, {})
        self.run_job(self.queued()[-1])
        self.assertEqual(self.entry(mid)["status"], "deleted")

    def test_wrong_persisted_job_binding_cannot_remove_existing_points(self):
        self.add()
        job = self.queued()[0]
        self.run_job(job)
        previous = dict(self.vectors.points)
        forged = self.jobs.submit("index", json.loads(job["payload_json"]), "unrelated-request")
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.run_job(forged)
        self.assertEqual(previous, self.vectors.points)

    def test_cleanup_removes_interrupted_stale_points_only(self):
        from codex_agent.indexing.cleanup import cleanup
        self.add()
        self.run_job(self.queued()[0])
        obsolete = dict(self.vectors.points)
        self.add(evidence={"test_summary": "updated fixture"})
        self.run_job(self.queued()[-1])
        fresh = dict(self.vectors.points)
        self.vectors.points.update(obsolete)
        cleanup(self.index, "test-v1", vector_factory=lambda *_: self.vectors)
        self.assertEqual(fresh, self.vectors.points)

    def test_index_backlog_does_not_consume_interactive_task_capacity(self):
        self.jobs.config = self.jobs.config.model_copy(update={"maxPending": 1})
        self.add()
        self.assertEqual(self.jobs.submit("validation", {}, "user-task")["status"], "queued")

    def test_immutable_snapshot_corruption_is_rejected(self):
        mid = self.add()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, memory_versions, {"payload_json": "{}"}, memory_versions.c.memory_id == mid)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_job(self.queued()[0])

    def test_second_workspace_has_no_first_workspace_index_or_jobs(self):
        self.add()
        other = IndexStore(self.root / "other")
        self.assertEqual(other.status()["indexes"], [])
        self.assertEqual(other.jobs.db.rows(jobs), [])

    def test_generation_definition_is_immutable_and_activation_checks_coverage(self):
        self.add()
        with self.assertRaises(Conflict):
            self.index.activate("test-v1")
        self.run_job(self.queued()[0])
        self.index.activate("test-v1")
        self.cfg["indexing"]["dimensions"] = 5
        self.configure()
        with self.assertRaises(Conflict):
            self.add(source_run="another")
        self.cfg["indexing"]["generation"] = "test-v2"
        self.configure()
        self.index.backfill()
        with self.assertRaises(Conflict):
            self.index.activate("test-v2")
        active = self.jobs.db.rows(indexes, indexes.c.status == "active")
        self.assertEqual([r["generation"] for r in active], ["test-v1"])

    def test_backfill_is_paginated_and_idempotent(self):
        self.cfg["indexing"]["enabled"] = False
        self.configure()
        for n in range(5):
            self.add(source_run=f"fixture-{n}")
        self.cfg["indexing"]["enabled"] = True
        self.configure()
        self.assertEqual(self.index.backfill(page_size=2)["submitted"], 5)
        self.assertEqual(self.index.backfill(page_size=2)["submitted"], 0)

    def test_retire_preserves_old_vectors_and_stops_new_jobs(self):
        self.add()
        self.run_job(self.queued()[0])
        self.index.activate("test-v1")
        with self.assertRaises(Conflict):
            self.index.retire("test-v1")
        self.cfg["indexing"]["generation"] = "test-v2"
        self.configure()
        self.index.backfill()
        self.run_job(self.queued()[-1])
        self.index.activate("test-v2")
        old_points = dict(self.vectors.points)
        self.index.retire("test-v1")
        self.assertEqual(self.vectors.points, old_points)
        before = len(self.queued())
        self.add(evidence={"test_summary": "new after retire"})
        self.assertEqual(len(self.queued()), before + 1)
        self.assertEqual(json.loads(self.queued()[-1]["payload_json"])["generation"], "test-v2")
        self.memory.archive(1, "also remove retired generation vectors")
        deleted = [json.loads(row["payload_json"])["generation"] for row in self.queued()[-2:]]
        self.assertEqual(set(deleted), {"test-v1", "test-v2"})

    def test_tokenizer_checksum_is_frozen_and_verified_by_worker(self):
        from codex_agent.indexing.worker import provider_for
        path = self.root / "fixture-tokenizer.json"
        path.write_text("not-a-real-tokenizer")
        self.cfg["indexing"]["embedding"]["tokenizerJson"] = str(path)
        self.configure()
        self.add()
        definition = json.loads(self.jobs.db.rows(indexes)[0]["config_json"])
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "tokenizer changed"):
            provider_for(definition)
        with patch("codex_agent.indexing.store.Path.read_bytes", side_effect=AssertionError("must not hash on each write")):
            self.add(source_run="a different source")

    def test_invalid_index_config_is_rejected_without_echoing_credentials(self):
        import copy
        for override in ({"dimensions": 0}, {"overlapTokens": 180}, {"generation": "../bad"},
                         {"qdrantUrl": "http://user:secret@host"}, {"apiKeyEnv": "TRITON_MYSQL_URL"},
                         {"embedding": {"provider": "ollama", "model": "fake", "baseUrl": "https://host?api_key=secret"}}):
            cfg = copy.deepcopy(self.cfg)
            cfg["indexing"].update(override)
            with self.assertRaises(ValueError) as caught:
                runtime_config({"TRITON_RISCV_CONFIG": json.dumps(cfg)})
            self.assertNotIn("secret", str(caught.exception))

    def test_cancelled_index_does_not_claim_ready(self):
        mid = self.add()
        job = self.queued()[0]
        self.jobs.cancel(job["id"])
        self.assertIsNone(self.claim(job))
        self.assertEqual(self.entry(mid)["status"], "pending")
        self.assertEqual(self.index.status()["indexes"][0]["states"][0]["job_status"], "cancelled")

    def test_index_secrets_are_not_in_operator_test_environment(self):
        from codex_agent.process_control import validation_environment
        self.cfg["indexing"]["apiKeyEnv"] = "VECTOR_SECRET"
        self.cfg["indexing"]["embedding"]["apiKeyEnv"] = "EMBED_SECRET"
        self.configure()
        with patch.dict(os.environ, VECTOR_SECRET="fixture", EMBED_SECRET="fixture"):
            self.assertNotIn("VECTOR_SECRET", validation_environment())
            self.assertNotIn("EMBED_SECRET", validation_environment())

    def test_index_jobs_are_not_blocked_by_operator_queue(self):
        self.jobs.submit("validation", {"fixture": True}, "blocking")
        self.add()
        self.assertIsNotNone(self.claim(self.queued()[0]))


@unittest.skipUnless(os.getenv("TRITON_TEST_QDRANT_URL"), "temporary Qdrant server required")
class RealQdrantTests(Fixture, unittest.TestCase):
    def vector_service(self):
        index = self.jobs.db.rows(indexes)[0]
        service = Qdrant(json.loads(index["config_json"]), index["collection"])
        self.addCleanup(lambda: service.request("DELETE", missing_ok=True))
        return service

    def test_actual_server_upsert_count_query_and_archive(self):
        mid = self.add()
        service = self.vector_service()
        self.run_job(self.queued()[0], vectors=service)
        count = service.request("POST", "/points/count", {"exact": True})["count"]
        self.assertEqual(count, self.entry(mid)["point_count"])
        found = service.request("POST", "/points/query", {"query": [1, 0, 0, 0], "limit": 3, "with_payload": True})
        self.assertTrue(found["points"])
        self.assertEqual(found["points"][0]["payload"]["memory_id"], mid)
        self.memory.archive(mid, "fixture")
        self.run_job(self.queued()[-1], vectors=service)
        self.assertEqual(service.request("POST", "/points/count", {"exact": True})["count"], 0)

    def test_actual_server_partial_write_retry_and_old_revision_cleanup(self):
        mid = self.add()
        service = self.vector_service()
        original = service.upsert
        def interrupted(points):
            original(points)
            raise RuntimeError("fixture lost response")
        service.upsert = interrupted
        job = self.queued()[0]
        claimed = self.claim(job)
        with self.assertRaises(RuntimeError):
            execute_index(self.jobs, claimed, claimed["owner"], provider_factory=lambda _: FixtureEmbedding(),
                          vector_factory=lambda *_: service)
        self.jobs.retry_index(job["id"], claimed["owner"], {"fixture": "lost response"})
        service.upsert = original
        self.run_job(job, vectors=service)
        self.add(evidence={"test_summary": "new snapshot"})
        self.run_job(self.queued()[-1], vectors=service)
        self.assertEqual(service.request("POST", "/points/count", {"exact": True})["count"], self.entry(mid)["point_count"])


@unittest.skipUnless(os.getenv("TRITON_TEST_AMQP_URL"), "temporary RabbitMQ required")
class IndexBrokerTests(Fixture, unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(os.getenv("TRITON_TEST_QDRANT_URL"), "temporary Qdrant server required")
    async def test_full_cli_pipeline_uses_real_mq_mysql_qdrant_and_fake_local_embedding(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                value = json.dumps({"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in request["input"]],
                                    "prompt_eval_count": max(map(len, request["input"]))}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(value)))
                self.end_headers()
                self.wfile.write(value)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        self.cfg["indexing"]["embedding"] = {"provider": "ollama", "model": "synthetic-local-http-fixture",
            "baseUrl": f"http://127.0.0.1:{server.server_port}", "tokenBudget": 180}
        self.configure()
        mid = self.add()
        index = self.jobs.db.rows(indexes)[0]
        service = Qdrant(json.loads(index["config_json"]), index["collection"])
        self.addCleanup(lambda: service.request("DELETE", missing_ok=True))
        for role in ("relay", "index"):
            done = await asyncio.to_thread(subprocess.run,
                [sys.executable, "-I", "-m", "codex_agent.workers", role, "--workspace", str(self.root), "--once"],
                capture_output=True, timeout=40)
            self.assertEqual(done.returncode, 0, done.stderr.decode())
        job = self.jobs.get(self.queued()[0]["id"])
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(self.entry(mid)["status"], "ready")
        self.assertGreater(self.entry(mid)["point_count"], 0)
        self.assertEqual(service.request("POST", "/points/count", {"exact": True})["count"], self.entry(mid)["point_count"])
        print(json.dumps({"fixture": "real-mysql-rabbitmq-qdrant-cli-with-fake-local-embedding",
            "job_status": job["status"], "points": self.entry(mid)["point_count"], "attempts": job["attempt"]}), flush=True)

    async def test_relay_delivers_index_when_agent_queue_is_disabled(self):
        from codex_agent.messaging.broker import Broker
        from codex_agent.messaging.relay import publish_once
        mid = self.add()
        async with Broker(self.jobs) as broker:
            self.assertTrue(await publish_once(self.jobs, broker))
            message = await broker.queues["index"].get(timeout=5)
            claim = self.jobs.claim(json.loads(message.body), "index")
            execute_index(self.jobs, claim, claim["owner"], provider_factory=lambda _: FixtureEmbedding(),
                          vector_factory=lambda *_: self.vectors)
            await message.ack()
        self.assertEqual(self.entry(mid)["status"], "ready")
        self.assertEqual(self.jobs.get(claim["id"])["status"], "succeeded")

    async def test_real_child_bad_tokenizer_retries_without_model_request(self):
        self.cfg["indexing"]["embedding"] = {"provider": "openai-compatible", "model": "synthetic-unconfigured",
            "baseUrl": "http://127.0.0.1:1/v1", "apiKeyEnv": "MISSING_INDEX_FIXTURE_KEY"}
        self.configure()
        self.add()
        for role in ("relay", "index"):
            done = await asyncio.to_thread(subprocess.run,
                [sys.executable, "-I", "-m", "codex_agent.workers", role, "--workspace", str(self.root), "--once"],
                capture_output=True, timeout=30)
            self.assertEqual(done.returncode, 0, done.stderr.decode())
        job = self.jobs.get(self.queued()[0]["id"])
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["attempt"], 1)
        self.assertEqual(job["result"]["error_type"], "ValueError")
