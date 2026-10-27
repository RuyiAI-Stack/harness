"""Read-path safety fixtures. These are not measurements of business RAG quality."""
import copy
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from codex_agent.diagnostic_memory import retrieve_memories
from codex_agent.indexing.qdrant import Qdrant, VectorServiceError
from codex_agent.indexing.retrieval import retrieve, RetrievalUnavailable
from codex_agent.memory import MemoryQuery, MemoryStore
from codex_agent.storage.schema import memories, memory_index_chunks as chunks, memory_indexes as indexes
from codex_agent.tests.test_memory_indexing import Fixture, FixtureEmbedding


class ReadFixture(Fixture):
    def setUp(self):
        super().setUp()
        self.query = MemoryQuery("tanh_and_mul", "exponential formulation", "")
        self.trace = {}
        self.calls = []
        self.hit_override = None
        self.vectors.query = self.vector_query

    def vector_query(self, vector, workspace, generation, fingerprint, *, limit, offset=0):
        self.calls.append(dict(workspace=workspace, generation=generation, fingerprint=fingerprint,
                               limit=limit, offset=offset))
        hits = self.hit_override if self.hit_override is not None else [
            {"id": p["id"], "payload": p["payload"], "score": 1.0}
            for p in self.vectors.points.values()]
        return hits[offset:offset + limit]

    def ready(self, **kwargs):
        mid = self.add(**kwargs)
        self.run_job(self.queued()[-1])
        self.index.activate("test-v1")
        return mid

    def read(self, **kwargs):
        return retrieve(self.memory, self.query, trace=self.trace,
            provider_factory=lambda _: FixtureEmbedding(), vector_factory=lambda *_: self.vectors, **kwargs)


class VectorRetrievalTests(ReadFixture, unittest.TestCase):
    def test_lookup_returns_mysql_evidence_without_scanning_all_cases(self):
        mid = self.ready()
        with patch.object(self.memory, "_stored", side_effect=AssertionError("full scan")), \
             patch.object(self.memory, "chunk_rows", side_effect=AssertionError("full chunk scan")):
            items = self.read()
        self.assertEqual([i["id"] for i in items], [mid])
        evidence = items[0]["matched_evidence"]
        self.assertTrue(evidence["text"])
        self.assertEqual(evidence["case_version"], self.entry(mid)["case_version"])
        self.assertGreater(self.trace["deduplicated"], 0)
        self.assertEqual(self.calls[0]["workspace"], self.memory.db.workspace_id)

    def test_default_legacy_does_not_call_vector_search(self):
        self.ready()
        with patch("codex_agent.indexing.retrieval.retrieve", side_effect=AssertionError("default changed")):
            self.assertTrue(self.memory.retrieve(self.query))

    def test_product_mode_uses_active_index_and_bypasses_stale_cache(self):
        self.ready()
        with patch("codex_agent.indexing.retrieval.provider_for", return_value=FixtureEmbedding()), \
             patch("codex_agent.indexing.retrieval.Qdrant", return_value=self.vectors), \
             patch.object(self.memory.cache, "get", side_effect=AssertionError("stale cache")):
            self.assertTrue(self.memory.retrieve(self.query, score_mode="qdrant"))

    def test_matching_text_and_unverified_state_reach_public_model_context(self):
        mid = self.ready(outcome="failed", summary="QUERY_MATCH_NOT_PROVEN_ROOT_CAUSE")
        summary = next(p for p in self.vectors.points.values() if p["payload"]["source_field"] == "summary")
        self.hit_override = [{"id": summary["id"], "payload": summary["payload"], "score": .9}]
        self.cfg["memory"] = {"retrievalMode": "qdrant"}
        self.configure()
        with patch("codex_agent.indexing.retrieval.provider_for", return_value=FixtureEmbedding()), \
             patch("codex_agent.indexing.retrieval.Qdrant", return_value=self.vectors):
            result = retrieve_memories(self.root, operator_name="tanh_and_mul")
        self.assertEqual(result["status"], "found", result)
        self.assertEqual(result["items"][0]["memory_id"], mid)
        self.assertIn("QUERY_MATCH_NOT_PROVEN_ROOT_CAUSE", result["context_excerpt"])
        self.assertIn("reported-not-proven", result["context_excerpt"])
        self.assertEqual(result["retrieval_trace"]["mode"], "qdrant")
        from codex_agent.memory_api import MemoryRetrievalToolResult
        self.assertEqual(MemoryRetrievalToolResult.model_validate(result).retrieval_trace["mode"], "qdrant")
        from codex_agent.memory import render_memory_context
        self.assertIn("QUERY_MATCH_NOT_PROVEN_ROOT_CAUSE",
                      render_memory_context(result["items"], context_format="compact"))

    def test_foreign_workspace_or_fingerprint_and_forged_points_rejected(self):
        self.ready()
        original = self.vector_query(None, None, None, None, limit=100)
        for field, value in (("workspace_id", "other"), ("generation", "other"),
                             ("index_fingerprint", "other"), ("case_version", "other"),
                             ("revision", 999), ("memory_id", 2**80), ("chunk_id", "missing")):
            with self.subTest(field=field):
                self.hit_override = copy.deepcopy(original)
                for hit in self.hit_override:
                    hit["payload"][field] = value
                self.assertEqual(self.read(), [])
                self.assertGreater(self.trace["rejected"], 0)
        self.hit_override = copy.deepcopy(original)
        for hit in self.hit_override:
            hit["id"] = "no-such-point"
        self.assertEqual(self.read(), [])

    def test_pending_update_and_archived_source_never_reuse_old_vectors(self):
        mid = self.ready()
        self.add(evidence={"test_summary": "updated evidence pending embedding"})
        self.assertEqual(self.read(), [])
        self.run_job(self.queued()[-1])
        self.assertEqual(self.read()[0]["matched_evidence"]["case_version"], self.entry(mid)["case_version"])
        self.memory.archive(mid, "fixture archive before delete worker")
        self.assertTrue(self.vectors.points)
        self.assertEqual(self.read(), [])

    def test_deleted_source_and_uncommitted_vector_points_rejected(self):
        mid = self.ready()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.delete(conn, memories, memories.c.id == mid)
        self.assertEqual(self.read(), [])

    def test_corrupted_sql_chunk_or_parent_not_used(self):
        mid = self.ready()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, chunks, {"text": "corrupt"}, chunks.c.memory_id == mid)
        self.assertEqual(self.read(), [])

    def test_unversioned_sql_parent_edit_is_rejected(self):
        mid = self.ready()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, memories, {"summary": "changed outside MemoryStore"}, memories.c.id == mid)
        self.assertEqual(self.read(), [])

    def test_invalid_limits_and_empty_queries_do_not_search(self):
        self.ready()
        for limit in (-1, 0, 51, True):
            with self.assertRaises(ValueError):
                self.read(limit=limit)
        self.query = MemoryQuery("", "", "")
        self.assertEqual(self.read(), [])
        self.assertEqual(self.calls, [])

    def test_public_errors_are_unavailable_not_false_empty(self):
        self.cfg["memory"] = {"retrievalMode": "qdrant"}
        self.configure()
        result = retrieve_memories(self.root, operator_name="tanh_and_mul")
        self.assertEqual(result["status"], "unavailable")

    def test_exclusions_refill_with_other_case_and_no_duplicate_slots(self):
        self.ready()
        mid = self.ready(source_run="independent-fixture", operator="another")
        items = self.read(exclude_source_runs=["agent-results/development/run-1"])
        self.assertEqual([i["id"] for i in items], [mid])
        self.assertGreater(self.trace["excluded"], 0)

    def test_staging_is_only_available_via_explicit_preview(self):
        self.add()
        self.run_job(self.queued()[-1])
        with self.assertRaises(RetrievalUnavailable):
            self.read()
        self.assertTrue(self.read(generation="test-v1", allow_staging=True))
        with self.assertRaises(ValueError):
            self.read(allow_staging=True)

    def test_bad_vectors_oversized_query_and_disabled_index_fail_before_search(self):
        self.ready()
        for vector in ([1, 2], [0, 0, 0, 0], [float("nan"), 0, 0, 0]):
            provider = FixtureEmbedding()
            provider.embed = lambda texts: [vector]
            with self.assertRaises(RetrievalUnavailable):
                retrieve(self.memory, self.query, provider_factory=lambda _: provider,
                         vector_factory=lambda *_: self.vectors)
        self.query = MemoryQuery("", "x" * 2049, "")
        with self.assertRaises(ValueError):
            self.read()
        self.cfg["indexing"]["enabled"] = False
        self.configure()
        with self.assertRaises(RetrievalUnavailable):
            self.read()
        self.assertEqual(self.calls, [])

    def test_provider_mismatch_and_service_failure_do_not_fallback(self):
        self.ready()
        provider = FixtureEmbedding()
        provider.model = "wrong-model"
        with self.assertRaises(RetrievalUnavailable):
            retrieve(self.memory, self.query, provider_factory=lambda _: provider)
        self.vectors.query = MagicMock(side_effect=VectorServiceError("fixture down"))
        with self.assertRaises(VectorServiceError):
            self.read()

    def test_generation_switch_during_search_requires_retry(self):
        self.ready()
        original = self.vectors.query
        def switch(*args, **kwargs):
            with self.jobs.db.transaction() as conn:
                self.jobs.db.update(conn, indexes, {"status": "retired"}, indexes.c.generation == "test-v1")
            return original(*args, **kwargs)
        self.vectors.query = switch
        with self.assertRaises(RetrievalUnavailable):
            self.read()

    def test_candidate_budget_is_bounded_and_reported(self):
        self.ready()
        hit = self.vector_query(None, None, None, None, limit=1)[0]
        self.hit_override = [hit] * 400
        self.assertEqual(len(self.read(limit=5)), 1)
        self.assertEqual(self.trace["scanned"], 200)
        self.assertTrue(self.trace["budget_exhausted"])

    def test_source_archived_between_pages_is_rechecked_before_return(self):
        mid = self.ready()
        hit = self.vector_query(None, None, None, None, limit=1)[0]
        self.hit_override = [hit] * 40
        original = self.vectors.query
        def archive_between_pages(*args, **kwargs):
            if kwargs["offset"]:
                self.memory.archive(mid, "changed after first page")
            return original(*args, **kwargs)
        self.vectors.query = archive_between_pages
        self.assertEqual(self.read(limit=5), [])

    def test_malformed_hits_fail_closed(self):
        self.ready()
        self.hit_override = [None, {}, {"score": float("inf"), "payload": {}}, {"payload": None}]
        self.assertEqual(self.read(), [])


class QueryAdapterTests(unittest.TestCase):
    def test_rest_request_never_fetches_vectors_and_scopes_all_filters(self):
        service = Qdrant({"qdrantUrl": "http://localhost:6333", "timeoutSeconds": 10,
                          "dimensions": 4, "apiKeyEnv": "UNSET_FIXTURE_KEY"}, "fixture")
        with patch.object(service, "request", return_value={"points": []}) as request:
            self.assertEqual(service.query([1, 0, 0, 0], "ws", "gen", "fp", limit=10), [])
        body = request.call_args.args[2]
        self.assertFalse(body["with_vector"])
        self.assertEqual({f["key"] for f in body["filter"]["must"]},
                         {"workspace_id", "generation", "index_fingerprint"})

    def test_revision_is_passed_to_actual_sentence_transformer_loader(self):
        from codex_agent.embeddings import build_embedding_provider, _sentence_encoder
        _sentence_encoder.cache_clear()
        fake = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": fake}):
            build_embedding_provider("sentence-transformers", model="BAAI/bge-m3", revision="pinned-revision")
            fake.SentenceTransformer.assert_called_once_with("BAAI/bge-m3", revision="pinned-revision",
                trust_remote_code=False, model_kwargs={"weights_only": True})
        _sentence_encoder.cache_clear()


@unittest.skipUnless(os.getenv("TRITON_TEST_QDRANT_URL"), "temporary Qdrant required")
class RealVectorReadTests(ReadFixture, unittest.TestCase):
    def test_real_mysql_qdrant_query_hydration_and_stale_rejection(self):
        mid = self.add()
        row = self.jobs.db.rows(indexes)[0]
        service = Qdrant(json.loads(row["config_json"]), row["collection"])
        self.addCleanup(lambda: service.request("DELETE", missing_ok=True))
        self.run_job(self.queued()[-1], vectors=service)
        self.index.activate("test-v1")
        items = retrieve(self.memory, self.query, provider_factory=lambda _: FixtureEmbedding())
        self.assertEqual([i["id"] for i in items], [mid])
        self.assertIn("exponential formulation", items[0]["matched_evidence"]["text"])
        self.add(evidence={"test_summary": "now pending"})
        self.assertEqual(retrieve(self.memory, self.query, provider_factory=lambda _: FixtureEmbedding()), [])
