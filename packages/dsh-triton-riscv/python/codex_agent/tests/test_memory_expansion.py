"""Source-bound expansion, budget and actual tool/bridge context-path tests."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from codex_agent.diagnostic_memory import _public_item, retrieve_memories
from codex_agent.memory import render_memory_context
from codex_agent.memory_api import MemoryRetrievalToolResult
from codex_agent.memory_expansion import expand_cases, resolve_stage
from codex_agent.storage.schema import memories, memory_index_chunks as chunks, memory_indexes as indexes
from codex_agent.tests.test_vector_retrieval import ReadFixture, FixtureEmbedding


class ExpansionTests(ReadFixture, unittest.TestCase):
    def seed(self, **kwargs):
        defaults = dict(outcome="failed", summary="Hypothesis, not verified.",
            evidence={"error_excerpt": ["OBSERVED_ERROR " + "a long failure segment " * 35],
                      "recommended_actions": ["RECOMMENDATION_ONLY try a different lowering"],
                      "applied_action": "ACTION_RECORDED changed a mask",
                      "patch_excerpt": "@@ -1 +1 @@\n-old\n+PATCH_RECORDED",
                      "test_summary": "VALIDATION_RECORDED 1 failed in 0.1s",
                      "command": "python -m pytest fixed_acceptance.py"})
        mid = self.ready(**{**defaults, **kwargs})
        self.select(mid, "evidence.error_excerpt[0]", part=1 if "evidence" not in kwargs else 0)
        return mid

    def select(self, mid, field, part=0):
        row = next(r for r in self.jobs.db.rows(chunks, chunks.c.memory_id == mid,
                   chunks.c.revision == self.entry(mid)["revision"])
                   if json.loads(r["provenance_json"])["source_field"] == field
                   and json.loads(r["provenance_json"])["unit_position"] == 0
                   and json.loads(r["provenance_json"])["part_index"] == part)
        point = self.vectors.points[row["point_id"]]
        self.hit_override = [{"id": point["id"], "payload": point["payload"], "score": .9}]
        return row

    def expand(self, stage="repair", items=None):
        return expand_cases(self.memory, self.read() if items is None else items, stage, trace=self.trace)

    def public(self, stage="repair"):
        self.cfg["memory"] = {"retrievalMode": "qdrant"}
        self.configure()
        with patch("codex_agent.indexing.retrieval.provider_for", return_value=FixtureEmbedding()), \
             patch("codex_agent.indexing.retrieval.Qdrant", return_value=self.vectors):
            return retrieve_memories(self.root, operator_name="tanh_and_mul", task_stage=stage, limit=1)

    def test_layers_keep_background_actions_results_and_neighbors(self):
        self.seed()
        expanded = self.expand()[0]
        entries = expanded["expanded_evidence"]["items"]
        self.assertEqual({e["layer"] for e in entries}, {0, 1, 2, 3})
        background = next(e for e in entries if e["layer"] == 1)
        self.assertIn("riscv64", background["text"])
        self.assertIn("torch.tanh", background["text"])
        self.assertEqual([e["part_index"] for e in entries if e["layer"] == 3], [0, 2])
        self.assertEqual(self.trace["expansion"]["neighbors"], 2)

    def test_task_stage_changes_context_not_ranking_or_query(self):
        self.seed()
        original = self.read()
        before = copy.deepcopy(original)
        generation = self.expand("generation", original)
        repair = self.expand("repair", original)
        self.assertEqual(original, before)
        self.assertEqual([i["retrieval"] for i in generation], [i["retrieval"] for i in repair])
        gen = json.dumps(generation[0]["expanded_evidence"])
        fix = json.dumps(repair[0]["expanded_evidence"])
        self.assertNotIn("PATCH_RECORDED", gen)
        self.assertNotIn("ACTION_RECORDED", gen)
        self.assertIn("PATCH_RECORDED", fix)
        self.assertIn("RECOMMENDATION_ONLY", fix)

    def test_all_stages_and_invalid_stage(self):
        self.seed()
        self.assertEqual(resolve_stage(None, diagnostic=True), "diagnosis")
        self.assertEqual(resolve_stage(None), "lookup")
        for stage in ("lookup", "generation", "diagnosis", "repair", "validation"):
            self.assertEqual(self.expand(stage)[0]["expansion"]["stage"], stage)
        with self.assertRaises(ValueError):
            self.expand("guess")

    def test_short_complete_unit_adds_no_neighbor(self):
        self.seed(evidence={"error_excerpt": ["short complete error"]})
        self.assertFalse([e for e in self.expand()[0]["expanded_evidence"]["items"] if e["layer"] == 3])

    def test_neighbors_never_cross_units_even_in_same_field(self):
        self.seed(evidence={"error_excerpt": ["a" * 900 + "\n\n" + "FOREIGN_PARAGRAPH " * 60]})
        result = self.expand()[0]
        text = json.dumps([e for e in result["expanded_evidence"]["items"] if e["layer"] in (0, 3)])
        self.assertNotIn("FOREIGN_PARAGRAPH", text)

    def test_old_missing_lineage_does_not_guess_neighbors(self):
        mid = self.seed()
        for row in self.jobs.db.rows(chunks, chunks.c.memory_id == mid):
            p = json.loads(row["provenance_json"])
            p.pop("part_index", None)
            with self.jobs.db.transaction() as conn:
                self.jobs.db.update(conn, chunks, {"provenance_json": json.dumps(p)}, chunks.c.chunk_id == row["chunk_id"])
        self.assertFalse([e for e in self.expand()[0]["expanded_evidence"]["items"] if e["layer"] == 3])
        self.assertGreater(self.trace["expansion"]["neighbor_unavailable"], 0)

    def test_corrupt_neighbor_is_omitted_not_used(self):
        mid = self.seed()
        target = next(r for r in self.jobs.db.rows(chunks, chunks.c.memory_id == mid)
                      if json.loads(r["provenance_json"])["source_field"] == "evidence.error_excerpt[0]"
                      and json.loads(r["provenance_json"])["part_index"] == 0)
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, chunks, {"text": "CORRUPT_NEIGHBOR"}, chunks.c.chunk_id == target["chunk_id"])
        self.assertNotIn("CORRUPT_NEIGHBOR", json.dumps(self.expand()))
        self.assertEqual(self.trace["expansion"]["rejected_neighbors"], 1)

    def test_missing_unit_metadata_does_not_guess_neighbors(self):
        mid = self.seed()
        for row in self.jobs.db.rows(chunks, chunks.c.memory_id == mid):
            p = json.loads(row["provenance_json"])
            p.pop("unit_sha256", None)
            with self.jobs.db.transaction() as conn:
                self.jobs.db.update(conn, chunks, {"provenance_json": json.dumps(p)}, chunks.c.chunk_id == row["chunk_id"])
        result = self.expand()[0]
        self.assertEqual(result["expansion"]["neighbor_parts_included"], 0)
        self.assertEqual(result["expansion"]["neighbor_unavailable"], 1)

    def test_pending_update_after_recall_drops_old_case(self):
        mid = self.seed()
        items = self.read()
        updated = self.add(outcome="failed", summary="Hypothesis, not verified.",
                           evidence={"test_summary": "new source, not indexed yet"})
        self.assertEqual(updated, mid)
        self.assertEqual(self.expand(items=items), [])

    def test_update_archive_and_generation_switch_after_recall_fail_closed(self):
        mid = self.seed()
        items = self.read()
        self.memory.archive(mid, "changed between recall and expansion")
        self.assertEqual(self.expand(items=items), [])
        self.assertEqual(self.trace["expansion"]["rejected_cases"], 1)
        self.seed()
        items = self.read()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, indexes, {"status": "retired"}, indexes.c.generation == "test-v1")
        self.assertEqual(self.expand(items=items), [])

    def test_direct_sql_source_corruption_after_recall_fails_closed(self):
        mid = self.seed()
        items = self.read()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, memories, {"semantics": "CORRUPTED"}, memories.c.id == mid)
        self.assertEqual(self.expand(items=items), [])

    def test_missing_fields_remain_unknown(self):
        self.seed(semantics="", pytorch_reference="", environment={})
        background = next(e for e in self.expand()[0]["expanded_evidence"]["items"] if e["layer"] == 1)
        facts = json.loads(background["text"])
        self.assertIsNone(facts["semantics"])
        self.assertEqual(facts["environment"], {})
        self.assertIn("environment", facts["missing_fields"])

    def test_long_patch_is_projected_as_whole_hunks_not_silent_cut(self):
        patch_text = "\n".join(f"@@ -{i} +{i} @@\n-old{i}\n+new{i}" for i in range(100))
        self.seed(evidence={"error_excerpt": ["short error"], "patch_excerpt": patch_text})
        evidence = next(e for e in self.expand()[0]["expanded_evidence"]["items"] if e["kind"] == "patch")
        self.assertTrue(evidence["truncated"])
        self.assertEqual(evidence["state"], "application-unknown")
        for i in range(100):
            self.assertEqual(f"-old{i}\n" in evidence["text"] + "\n", f"+new{i}\n" in evidence["text"] + "\n")

    def test_public_context_keeps_evidence_states_and_no_unselected_flat_payload(self):
        self.seed()
        result = self.public()
        self.assertEqual(result["status"], "found", result)
        typed = MemoryRetrievalToolResult.model_validate(result).model_dump()
        doc = json.loads(typed["context_excerpt"])
        self.assertEqual(doc["cases"][0]["expansion"]["stage"], "repair")
        self.assertEqual({e["layer"] for e in doc["evidence"]}, {0, 1, 2, 3})
        text = typed["context_excerpt"]
        for marker in ("PATCH_RECORDED", "VALIDATION_RECORDED", "RECOMMENDATION_ONLY", "not-executed", "application-unknown", "not-established"):
            self.assertIn(marker, text)
        self.assertIsNone(typed["items"][0]["applied_action"])
        self.assertIsNone(typed["items"][0]["reported_cause"])
        self.assertEqual(typed["retrieval_trace"]["expansion"]["schema"], "three-layer-v1")

    def test_budget_unicode_and_many_cases_are_bounded_and_omissions_visible(self):
        self.seed(semantics="中文😀" * 200)
        public = [_public_item(i) for i in self.expand()]
        for budget in (0, 10, 100, 500, 1200, 2400, 6000):
            result = render_memory_context(public, budget)
            self.assertLessEqual(len(result), budget)
        cases = []
        for i in range(5):
            item = copy.deepcopy(public[0])
            item["memory_id"] = f"case-{i}"
            item["case_id"] = f"case-{i}"
            for entry in item["evidence_chain"]["items"]:
                entry["source"] = f"source-{i}"
                entry["case_version"] = str(i)
            cases.append(item)
        text = render_memory_context(cases, 6000)
        self.assertLessEqual(len(text), 6000)
        self.assertGreater(json.loads(text).get("omitted", {}).get("evidence", 0), 0)

    def test_shared_environment_is_interned_without_losing_version_fields(self):
        self.seed()
        items = [_public_item(i) for i in self.expand()]
        second = copy.deepcopy(items[0])
        second["memory_id"] = 999
        for entry in second["evidence_chain"]["items"]:
            entry["source"] = "another-source"
            entry["case_version"] = "another-version"
        doc = json.loads(render_memory_context([*items, second], 20000))
        backgrounds = [e for e in doc["evidence"] if e["layer"] == 1]
        self.assertEqual(len(backgrounds), 2)
        self.assertEqual(len(doc["environments"]), 1)
        for entry in backgrounds:
            env = doc["environments"][entry["environment"]]
            self.assertEqual(env["triton"], "3.4.0")
            self.assertEqual(env["architecture"], "riscv64")

    def test_actual_mcp_and_bridge_return_same_expanded_context(self):
        self.seed()
        self.cfg["memory"] = {"retrievalMode": "qdrant"}
        self.configure()
        from mcp import Client
        from codex_agent.harness.mcp_server import server
        from codex_agent.harness.native_bridge import dispatch
        query = dict(operator_name="tanh_and_mul", task_stage="repair", limit=1)
        async def call():
            async with Client(server) as client:
                invalid = await client.call_tool("retrieve_operator_memory", {**query, "task_stage": "invalid"})
                self.assertTrue(invalid.is_error)
                result = await client.call_tool("retrieve_operator_memory", query)
                self.assertFalse(result.is_error)
                return result.structured_content
        with patch("codex_agent.indexing.retrieval.provider_for", return_value=FixtureEmbedding()), \
             patch("codex_agent.indexing.retrieval.Qdrant", return_value=self.vectors):
            tool = asyncio.run(call())
            bridge = dispatch(self.root, {"action": "memory", "query": query})
        self.assertEqual(tool["status"], "found", tool)
        self.assertEqual(tool["context_excerpt"], bridge["context_excerpt"])
        self.assertIn("PATCH_RECORDED", tool["context_excerpt"])
