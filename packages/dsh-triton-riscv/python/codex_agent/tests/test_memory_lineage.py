"""Synthetic lineage invariants; these are not retrieval efficacy measurements."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import uuid

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from codex_agent.memory import MemoryStore, memory_chunks as legacy_chunks, memory_fingerprint
from codex_agent.memory_chunking import ChunkPolicy, embedding_input, case_chunk_parts as memory_chunks
from codex_agent.memory_lineage import (
    case_payload, chunk_lineage, digest, policy_metadata, source_binding,
)
from codex_agent.storage.schema import memories, chunks, memory_versions
from codex_agent.storage.database import StorageError
from codex_agent.tests.test_memory import FakeEmbeddingProvider, record


class ChunkLineageTests(unittest.TestCase):
    def test_legacy_four_field_chunk_interface_is_unchanged(self):
        item = asdict(record())
        old = [(kind, position, value, source) for kind, position, value, source in legacy_chunks(item)]
        new = [(p.kind, p.position, p.text, p.source_field) for p in memory_chunks(item)]
        self.assertEqual(old, new)

    def test_short_field_is_complete_single_unit(self):
        parts = memory_chunks(asdict(record()))
        part = next(c for c in parts if c.source_field == "semantics")
        self.assertEqual(part.text, "Semantics: " + record().semantics)
        self.assertEqual((part.part_index, part.part_count), (0, 1))
        self.assertEqual(len(part.unit_sha256), 64)

    def test_long_chinese_and_unspaced_text_respect_budget_and_part_order(self):
        provider = FakeEmbeddingProvider()
        policy = ChunkPolicy(max_tokens=150, overlap_tokens=12)
        for value in ("无空格错误" * 400, "x" * 2500):
            with self.subTest(value=value[:10]):
                item = asdict(record(evidence={"error_excerpt": [value]}))
                parts = memory_chunks(item, token_count=provider.count_tokens, policy=policy)
                errors = [p for p in parts if p.source_field == "evidence.error_excerpt[0]"]
                self.assertGreater(len(errors), 1)
                self.assertEqual(len({p.unit_sha256 for p in errors}), 1)
                self.assertEqual([p.part_index for p in errors], list(range(len(errors))))
                self.assertEqual({p.part_count for p in errors}, {len(errors)})
                self.assertTrue(all(provider.count_tokens(embedding_input(item["operator"], p)) <= 150
                                    for p in errors))

    def test_paragraphs_events_and_patch_hunks_are_separate_units(self):
        evidence = {"error_excerpt": ["A" * 900 + "\n\n" + "B" * 900, "C" * 900],
                    "patch_excerpt": "diff --git a/a b/a\n@@ -1 +1 @@\n-" + "old" * 300 +
                    "\n+new\n@@ -9 +9 @@\n-second\n+third"}
        parts = memory_chunks(asdict(record(evidence=evidence)))
        errors = [p for p in parts if p.source_field == "evidence.error_excerpt[0]"]
        self.assertEqual({p.unit_position for p in errors}, {0, 1})
        self.assertFalse(any("A" in p.text and "B" in p.text for p in errors))
        patches = [p for p in parts if p.source_field == "evidence.patch_excerpt"]
        self.assertGreater(len({p.unit_position for p in patches}), 1)

    def test_ids_include_workspace_parent_version_policy_and_part(self):
        item = asdict(record())
        part = memory_chunks(item)[0]
        policy = policy_metadata(ChunkPolicy())
        make = lambda workspace, mid, version, settings=policy: chunk_lineage(
            workspace, mid, version, item, part, settings)
        first = make("workspace-a", 1, "v1")
        self.assertEqual(first, make("workspace-a", 1, "v1"))
        variants = [make("workspace-b", 1, "v1"), make("workspace-a", 2, "v1"),
                    make("workspace-a", 1, "v2"), make("workspace-a", 1, "v1", {**policy, "max_chars": 300})]
        self.assertTrue(all(v["chunk_id"] != first["chunk_id"] for v in variants))
        self.assertEqual(json.loads(first["provenance_json"])["policy"]["counting"], "characters-only")

    def test_source_states_are_copied_not_inferred_from_pass(self):
        entries = [dict(evidence_id="e1", kind="recommendation", state="not-executed",
                        run_id="run-a", proposal_id="p-a", attempt=2, source="receipt.json",
                        pointer="/diagnosis/actions", lines=[3, 5], source_sha256="hash"),
                   dict(evidence_id="e2", kind="patch", state="proposed-not-applied", run_id="run-b")]
        item = asdict(record(outcome="passed", evidence={"chain": {"items": entries}}))
        source = source_binding(item, "evidence.chain.e1")
        for key in ("state", "run_id", "proposal_id", "attempt", "source", "pointer", "lines", "source_sha256"):
            self.assertEqual(source[key], entries[0][key])
        self.assertEqual(source_binding(item, "evidence.chain.e2")["state"], "proposed-not-applied")

    def test_missing_and_ambiguous_sources_do_not_get_invented_links(self):
        item = asdict(record(source_run="", evidence={}))
        source = source_binding(item, "evidence.recommended_actions[0]")
        self.assertEqual(source["state"], "not-executed")
        for key in ("source", "source_run", "run_id", "proposal_id", "attempt", "source_sha256"):
            self.assertIsNone(source[key])
        self.assertEqual(source["pointer"], "/evidence/recommended_actions/0")
        item["evidence"] = {"chain": {"items": [{"evidence_id": "same", "text": "one"},
                                                   {"evidence_id": "same", "text": "two"}]}}
        source = source_binding(item, "evidence.chain.same")
        self.assertEqual(source["binding"], "ambiguous-evidence-entry")
        self.assertIsNone(source["run_id"])

    def test_version_ignores_access_counters_but_tracks_evidence_environment(self):
        item = asdict(record())
        version = digest(case_payload(item))
        self.assertEqual(version, digest(case_payload({**item, "last_used_at": "now", "active": False,
                                                       "embedding": [1, 2], "useful_count": 5})))
        self.assertNotEqual(version, digest(case_payload({**item, "environment": {"triton": "other"}})))
        self.assertNotEqual(version, digest(case_payload({**item, "evidence": {"attempt": 2}})))


class StoredLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = MemoryStore(self.root)

    def test_new_records_persist_parent_snapshot_and_complete_chunk_links(self):
        mid, _ = self.store.add(record(evidence={"error_excerpt": ["error " * 1000]}))
        parent = self.store.get(mid)
        parts = self.store.chunk_rows(mid)
        self.assertTrue(parts)
        groups = {}
        for part in parts:
            self.assertEqual(part["case_version"], parent["case_version"])
            provenance = json.loads(part["provenance_json"])
            self.assertEqual(provenance["case_id"], parent["case_id"])
            groups.setdefault(part["unit_id"], []).append(part)
        self.assertEqual(len({p["chunk_id"] for p in parts}), len(parts))
        for group in groups.values():
            self.assertEqual(sorted(p["part_index"] for p in group), list(range(len(group))))
            self.assertEqual({p["part_count"] for p in group}, {len(group)})
        self.assertEqual(self.store.case_snapshot(mid, parent["case_version"]), case_payload(parent))
        other = MemoryStore(self.root / "other")
        self.assertIsNone(other.case_snapshot(mid, parent["case_version"]))

    def test_duplicate_ingest_and_rebuild_keep_ids_versions_and_vectors(self):
        self.store = MemoryStore(self.root, FakeEmbeddingProvider())
        mid, _ = self.store.add(record())
        before = self.store.chunk_rows(mid)
        self.assertEqual(self.store.add(record()), (mid, False))
        self.assertEqual(self.store.rebuild_chunks(), 0)
        self.assertEqual(before, self.store.chunk_rows(mid))
        self.assertEqual(len(self.store.db.rows(memory_versions)), 1)

    def test_same_summary_from_different_run_or_attempt_never_overwrites(self):
        self.store.add(record(source_run="run-a"))
        self.store.add(record(source_run="run-b"))
        self.store.add(record(source_run="run-b", evidence={"attempt": 1}))
        self.store.add(record(source_run="run-b", evidence={"attempt": 2}))
        self.assertEqual(len(self.store.list()), 4)
        self.assertEqual(len({p["unit_id"] for p in self.store.chunk_rows()}), len(self.store.chunk_rows()))

    def test_explicit_episode_update_keeps_old_snapshot_and_replaces_child_version(self):
        evidence = {"chain": {"schema": "evidence-chain-v1", "identity": "run-a", "items": []}}
        original = record(evidence=evidence)
        mid, _ = self.store.add(original)
        old = self.store.get(mid)
        old_ids = {p["chunk_id"] for p in self.store.chunk_rows(mid)}
        updated = record(summary="new observation", evidence={**evidence, "recommended_actions": ["try masking"]})
        self.assertEqual(self.store.add(updated), (mid, False))
        new = self.store.get(mid)
        self.assertEqual(old["case_id"], new["case_id"])
        self.assertNotEqual(old["case_version"], new["case_version"])
        self.assertEqual(self.store.case_snapshot(mid, old["case_version"]), case_payload(old))
        self.assertEqual(len(self.store.db.rows(memory_versions)), 2)
        self.assertFalse(old_ids & {p["chunk_id"] for p in self.store.chunk_rows(mid)})
        self.assertEqual({p["case_version"] for p in self.store.chunk_rows(mid)}, {new["case_version"]})

    def test_explicit_rebuild_backfills_unknown_metadata_without_discarding_vectors(self):
        self.store = MemoryStore(self.root, FakeEmbeddingProvider())
        mid, _ = self.store.add(record())
        with self.store.db.transaction() as conn:
            self.store.db.update(conn, memories, {"case_version": None}, memories.c.id == mid)
            self.store.db.update(conn, chunks, dict(case_version=None, chunk_id=None, unit_id=None,
                part_index=None, part_count=None, provenance_json=None), chunks.c.memory_id == mid)
        before = self.store.chunk_rows(mid)
        self.assertIsNone(MemoryStore(self.root).get(mid)["case_version"])
        self.assertEqual(self.store.rebuild_chunks(), 1)
        after = self.store.chunk_rows(mid)
        self.assertTrue(all(p["unit_id"] for p in after))
        for left, right in zip(before, after):
            for key in ("text", "embedding_json", "embedding_provider", "embedding_model", "embedding_dim"):
                self.assertEqual(left[key], right[key])
        self.assertEqual(self.store.rebuild_chunks(), 0)

    def test_legacy_fingerprint_is_reused_only_for_its_original_run(self):
        original = record(source_run="run-a")
        mid, _ = self.store.add(original)
        with self.store.db.transaction() as conn:
            self.store.db.update(conn, memories, {"fingerprint": memory_fingerprint(original.normalized(), bind_source=False)},
                                 memories.c.id == mid)
        self.assertEqual(self.store.add(original), (mid, False))
        new_id, created = self.store.add(record(source_run="run-b"))
        self.assertTrue(created)
        self.assertNotEqual(new_id, mid)
        self.assertEqual(self.store.get(mid)["source_run"], "run-a")

    def test_failed_update_rolls_back_snapshot_parent_and_children(self):
        mid, _ = self.store.add(record())
        before = self.store.get(mid), self.store.chunk_rows(mid), self.store.db.rows(memory_versions)
        with patch.object(self.store, "_insert_chunks", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.store.add(record(evidence={"recommended_actions": ["new"]}))
        self.assertEqual(before, (self.store.get(mid), self.store.chunk_rows(mid), self.store.db.rows(memory_versions)))

    def test_lineage_never_promotes_recommendation_to_verified_repair(self):
        mid, _ = self.store.add(record(outcome="passed", evidence={"recommended_actions": ["try X"]}))
        part = next(p for p in self.store.chunk_rows(mid) if p["source_field"] == "evidence.recommended_actions[0]")
        self.assertEqual(json.loads(part["provenance_json"])["source"]["state"], "not-executed")

    def test_duplicate_evidence_ids_stay_ambiguous_and_do_not_collide(self):
        entries = [{"evidence_id": "same", "text": "same", "kind": "log"}] * 2
        mid, _ = self.store.add(record(evidence={"chain": {"items": entries}}))
        parts = [p for p in self.store.chunk_rows(mid) if p["source_field"] == "evidence.chain.same"]
        self.assertEqual(len(parts), 2)
        self.assertNotEqual(parts[0]["unit_id"], parts[1]["unit_id"])
        self.assertTrue(all(json.loads(p["provenance_json"])["source"]["binding"] ==
                            "ambiguous-evidence-entry" for p in parts))

    def test_changed_test_hash_or_stage_updates_explicit_episode_version(self):
        evidence = {"chain": {"schema": "evidence-chain-v1", "identity": "episode", "items": []}}
        mid, _ = self.store.add(record(evidence=evidence))
        old = self.store.get(mid)
        self.store.add(record(evidence=evidence, test_sha256="new-test", failure_stage="runtime",
                              error_signature="new-error"))
        new = self.store.get(mid)
        self.assertEqual(new["test_sha256"], "new-test")
        self.assertEqual(new["failure_stage"], "runtime")
        self.assertNotEqual(old["case_version"], new["case_version"])
        self.assertEqual(self.store.case_snapshot(mid, old["case_version"])["test_sha256"], old["test_sha256"])

    def test_database_rejects_nonexistent_parent_version(self):
        mid, _ = self.store.add(record())
        before = self.store.chunk_rows(mid)
        with self.assertRaises(StorageError):
            with self.store.db.transaction() as conn:
                self.store.db.update(conn, chunks, {"case_version": "0" * 64}, chunks.c.memory_id == mid)
        self.assertEqual(before, self.store.chunk_rows(mid))


class LineageMigrationTests(unittest.TestCase):
    def test_upgrade_old_rows_and_retry_partial_ddl_without_rewriting_content(self):
        # The parent test service must be disposable. Use a separate schema for old DDL.
        url = make_url(os.environ["TRITON_MYSQL_URL"])
        name = "triton_lineage_test_" + uuid.uuid4().hex
        admin = create_engine(url.set(database=None), hide_parameters=True)
        target = create_engine(url.set(database=name), hide_parameters=True)
        try:
            with admin.begin() as conn:
                conn.execute(text(f"CREATE DATABASE {name} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"))
            config = Config()
            config.set_main_option("script_location", str(Path(__file__).parents[1] / "storage/migrations"))
            with target.connect() as conn:
                config.attributes["connection"] = conn
                command.upgrade(config, "0002_tasks")
            sample = record().normalized()
            values = asdict(sample)
            for key in ("tl_ops", "environment", "evidence"):
                values[key + "_json"] = json.dumps(values.pop(key))
            values.update(workspace_id="legacy-workspace", id=1, fingerprint=memory_fingerprint(sample, bind_source=False),
                          searchable_text=sample.searchable_text(), confidence=1.0,
                          created_at="2026-09-01T00:00:00Z", updated_at="2026-09-01T00:00:00Z")
            with target.begin() as conn:
                conn.execute(memories.insert().values(**values))
                conn.execute(chunks.insert().values(workspace_id="legacy-workspace", memory_id=1,
                    kind="contract", position=0, text="legacy unchanged", source_field="semantics",
                    embedding_json="[1,0]", embedding_provider="fixture", embedding_model="old", embedding_dim=2))
                before = dict(conn.execute(text("SELECT * FROM memories")).mappings().one())
                before_chunk = dict(conn.execute(text("SELECT * FROM memory_chunks")).mappings().one())
                conn.execute(text("ALTER TABLE memories ADD COLUMN case_version VARCHAR(64) NULL"))
            for _ in range(2):
                with target.connect() as conn:
                    config.attributes["connection"] = conn
                    command.upgrade(config, "head")
            with target.connect() as conn:
                parent = dict(conn.execute(text("SELECT * FROM memories")).mappings().one())
                part = dict(conn.execute(text("SELECT * FROM memory_chunks")).mappings().one())
                self.assertEqual({k: parent[k] for k in before}, before)
                self.assertEqual({k: part[k] for k in before_chunk}, before_chunk)
                self.assertIsNone(parent["case_version"])
                self.assertIsNone(part["unit_id"])
                self.assertEqual(conn.execute(text("SELECT COUNT(*) FROM memory_versions")).scalar(), 0)
        finally:
            target.dispose()
            with admin.begin() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS {name}"))
            admin.dispose()


if __name__ == "__main__":
    unittest.main()
