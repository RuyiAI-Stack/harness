from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryStore,
    ingest_results,
    memory_chunks,
    query_embedding_sections,
    records_from_run,
    render_memory_context,
    split_chunk_text,
)
from codex_agent.memory_chunking import ChunkPolicy, embedding_input
from codex_agent.embeddings import OpenAICompatibleEmbeddingProvider


class FakeEmbeddingProvider:
    name = "fake"
    model = "fake-v1"
    max_input_tokens = 2048

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0]
            if "exponential formulation" in text
            else [0.0, 1.0, 0.0, 0.0]
            if "pointer masking" in text
            else [0.0, 0.0, 1.0, 0.0]
            if "unique compiler dialect" in text
            else [0.0, 0.0, 0.0, 1.0]
            for text in texts
        ]


def record(**overrides: object) -> MemoryRecord:
    values = {
        "memory_type": "successful-repair",
        "operator": "tanh_and_mul",
        "semantics": "Compute tanh(x) multiplied by y.",
        "pytorch_reference": "torch.tanh(x) * y",
        "summary": "Use an exponential formulation to avoid unsupported lowering.",
        "outcome": "passed",
        "confidence_grade": "A",
        "source_run": "agent-results/development/run-1",
        "tl_ops": ["exp", "load", "store"],
        "failure_stage": "llvm-ir",
        "error_signature": "linalg-signature",
        "environment": {
            "architecture": "riscv64",
            "execution_mode": "native-riscv",
            "triton": "3.4.0",
        },
        "evidence": {"test_summary": "12 passed in 30.86s"},
        "test_sha256": "abc",
    }
    values.update(overrides)
    return MemoryRecord(**values)


class MemoryTests(unittest.TestCase):
    def test_deduplicates_redacts_and_soft_archives_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            with MemoryStore(path) as store:
                first_id, first_created = store.add(
                    record(summary="Successful repair api_key=do-not-store")
                )
                second_id, second_created = store.add(
                    record(summary="Successful repair api_key=do-not-store")
                )

                self.assertTrue(first_created)
                self.assertFalse(second_created)
                self.assertEqual(first_id, second_id)
                self.assertNotIn("do-not-store", store.list()[0]["summary"])
                self.assertTrue(store.archive(first_id, "superseded by new evidence"))
                self.assertEqual(store.list(), [])
                self.assertEqual(len(store.list(active_only=False)), 1)

    def test_maintenance_archives_superseded_error_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryStore(Path(temp_dir) / "memory.sqlite3") as store:
                first_id, _ = store.add(
                    record(
                        memory_type="failure-diagnosis",
                        summary="First observation of the same compiler failure.",
                        outcome="failed",
                        confidence_grade="C",
                    )
                )
                second_id, _ = store.add(
                    record(
                        memory_type="failure-diagnosis",
                        summary="Better evidence for the same compiler failure.",
                        outcome="failed",
                        confidence_grade="A",
                        source_run="run-new",
                    )
                )

                result = store.maintain()

                self.assertEqual(result["archived"], 1)
                self.assertEqual(store.list()[0]["id"], second_id)
                archived = {
                    item["id"]: item for item in store.list(active_only=False)
                }
                self.assertFalse(archived[first_id]["active"])

    def test_structured_and_embedding_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            with MemoryStore(path, FakeEmbeddingProvider()) as store:
                expected_id, _ = store.add(record())
                store.add(
                    record(
                        operator="masked_load",
                        semantics="Load values with an explicit mask.",
                        pytorch_reference="torch.where(mask, x, 0)",
                        summary="Repair pointer masking and boundary offsets.",
                        tl_ops=["load", "where"],
                        failure_stage="runtime",
                        error_signature="segfault",
                        source_run="run-2",
                        test_sha256="def",
                    )
                )
                query = MemoryQuery(
                    operator="new_tanh_and_mul",
                    semantics="Use an exponential formulation for tanh and multiply.",
                    pytorch_reference="torch.tanh(x) * y",
                    tl_ops=["exp", "load", "store"],
                    failure_stage="llvm-ir",
                    error_signature="linalg-signature",
                    environment={
                        "architecture": "riscv64",
                        "execution_mode": "native-riscv",
                        "triton": "3.4.0",
                    },
                )

                results = store.retrieve(query, limit=2)

                self.assertEqual(results[0]["id"], expected_id)
                self.assertIsNotNone(results[0]["retrieval"]["semantic_score"])
                self.assertGreater(
                    results[0]["retrieval"]["score"],
                    results[1]["retrieval"]["score"],
                )

    def test_ingests_verified_run_as_episode_diagnosis_and_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "development/20260809-tanh"
            run_dir.mkdir(parents=True)
            (run_dir / "operator-spec.json").write_text(
                json.dumps(
                    {
                        "name": "tanh_and_mul",
                        "semantics": "Compute tanh(x) multiplied by y.",
                        "pytorch_reference": "torch.tanh(x) * y",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "environment.json").write_text(
                json.dumps(
                    {
                        "architecture": "riscv64",
                        "execution": {"mode": "native-riscv"},
                        "triton": {"version": "3.4.0"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "final-result.json").write_text(
                json.dumps(
                    {
                        "operator": "tanh_and_mul",
                        "status": "passed",
                        "locked_test_sha256": "abc",
                        "repair_attempts": 1,
                        "validations": [
                            {
                                "iteration": 1,
                                "status": "failed",
                                "first_failure_stage": "llvm-ir",
                                "likely_reason": "unsupported linalg operation",
                                "error_excerpt": ["linalg.generic remained"],
                            },
                            {
                                "iteration": 2,
                                "status": "passed",
                                "test_summary": "12 passed in 30.86s",
                            },
                        ],
                        "repair_history": [
                            {
                                "attempt": 1,
                                "accepted": True,
                                "outcome": "passed",
                                "reason": "locked test passed",
                                "decision": {
                                    "stage": "llvm-ir",
                                    "strategy": "use equivalent supported primitives",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            records = records_from_run(run_dir)
            self.assertEqual(
                {item.memory_type for item in records},
                {"failure-diagnosis", "successful-repair", "successful-run"},
            )
            self.assertTrue(all(item.confidence_grade in {"A", "C"} for item in records))

            with MemoryStore(root / "memory.sqlite3") as store:
                first = ingest_results(store, root / "development")
                second = ingest_results(store, root / "development")
                self.assertEqual(first["added"], 3)
                self.assertEqual(second["duplicates"], 3)
                self.assertEqual(store.stats()["total"], 3)

    def test_embeds_existing_lexical_memories_and_limits_prompt_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            with MemoryStore(path) as store:
                memory_id, _ = store.add(record(summary="exponential formulation " * 100))
                self.assertEqual(store.stats()["embedded"], 0)

            with MemoryStore(path, FakeEmbeddingProvider()) as store:
                result = store.embed_missing()
                self.assertEqual(result["embedded"], 0)
                self.assertGreaterEqual(result["embedded_chunks"], 2)
                item = store.list()[0]
                item["retrieval"] = {"score": 0.8}
                context = render_memory_context([item], max_chars=800)
                self.assertLessEqual(len(context), 800)
                self.assertIn(f"memory #{memory_id}", context)
                self.assertIn("source=agent-results/development/run-1", context)

    def test_semantic_chunks_keep_case_identity_and_exclude_environment(self) -> None:
        chunks = memory_chunks(record().normalized().__dict__)
        self.assertEqual({chunk.kind for chunk in chunks}, {"contract", "diagnosis", "outcome"})
        self.assertTrue(all(len(chunk.text) <= 600 for chunk in chunks))
        self.assertTrue(all("riscv64" not in chunk.text for chunk in chunks))
        long_chunks = split_chunk_text("evidence " * 200)
        self.assertGreater(len(long_chunks), 1)
        self.assertTrue(all(len(text) <= 600 for text in long_chunks))
        self.assertTrue(all(len(text) <= 600 for text in split_chunk_text("x" * 1300)))

    def test_three_retrieval_modes_require_matching_chunk_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            query = MemoryQuery(
                operator="tanh_and_mul",
                semantics="Compute tanh(x) multiplied by y.",
                pytorch_reference="torch.tanh(x) * y",
                diagnostic_text="unsupported lowering",
            )
            with MemoryStore(path) as store:
                expected_id, _ = store.add(record())
                self.assertEqual(store.retrieve(query, score_mode="jaccard")[0]["id"], expected_id)
                with self.assertRaisesRegex(RuntimeError, "embedding provider"):
                    store.retrieve(query, score_mode="embedding")
            with MemoryStore(path, FakeEmbeddingProvider()) as store:
                with self.assertRaisesRegex(RuntimeError, "embed-missing"):
                    store.retrieve(query, score_mode="fusion")
                self.assertGreater(store.embed_missing()["embedded_chunks"], 0)
                for mode in ("jaccard", "embedding", "fusion"):
                    result = store.retrieve(query, score_mode=mode)
                    self.assertEqual(result[0]["id"], expected_id)
                    self.assertEqual(result[0]["retrieval"]["mode"], mode)
                with self.assertRaises(ValueError):
                    store.retrieve(query, score_mode="fusion", lexical_weight=1.5)
                store.connection.execute("DELETE FROM memory_chunks")
                store.connection.commit()
            with MemoryStore(path, FakeEmbeddingProvider()) as store:
                self.assertGreater(store.embed_missing()["embedded_chunks"], 0)
                self.assertEqual(store.retrieve(query, score_mode="embedding")[0]["id"], expected_id)

    def test_old_relevant_case_is_retrieved_after_1000_newer_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            with MemoryStore(path) as store:
                old_id, _ = store.add(record(
                    operator="old_exact_case",
                    semantics="unique compiler dialect failure in reduction",
                    pytorch_reference="torch.sum(x)",
                    summary="unique compiler dialect failure in reduction",
                ))
                for index in range(1001):
                    store.add(record(
                        operator=f"unrelated_{index}",
                        semantics="matrix allocation",
                        pytorch_reference="torch.zeros(x)",
                        summary=f"unrelated memory record {index}",
                        source_run=f"run-{index}",
                    ))
                query = MemoryQuery(
                    operator="old_exact_case",
                    semantics="unique compiler dialect failure in reduction",
                    pytorch_reference="torch.sum(x)",
                )
                self.assertEqual(store.retrieve(query, score_mode="jaccard")[0]["id"], old_id)
            with MemoryStore(path, FakeEmbeddingProvider()) as store:
                store.embed_missing(batch_size=128)
                for mode in ("embedding", "fusion"):
                    self.assertEqual(store.retrieve(query, score_mode=mode)[0]["id"], old_id)

    def test_short_multiline_evidence_stays_intact_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryStore(Path(temp_dir) / "memory.sqlite3") as store:
                memory_id, _ = store.add(record(evidence={
                    "error_excerpt": ["ll.mlir: error: dialect missing\n  linalg.generic\n  ^"],
                }))
                item = store.list()[0]
                chunks = store.connection.execute(
                    "SELECT * FROM memory_chunks WHERE memory_id = ?", (memory_id,)
                ).fetchall()
                excerpt = next(row for row in chunks if row["source_field"] == "evidence.error_excerpt[0]")
                self.assertIn("\n  linalg.generic\n  ^", excerpt["text"])
                self.assertEqual(item["evidence"]["error_excerpt"][0],
                                 "ll.mlir: error: dialect missing\n  linalg.generic\n  ^")
                self.assertEqual(item["source_run"], "agent-results/development/run-1")

    def test_recursive_chunks_cover_chinese_unspaced_text_and_patch_without_mixing_fields(self) -> None:
        long_chinese = "编译错误无法识别方言" * 80
        patch = "@@ -1,2 +1,2 @@\n-old\n+new\n" * 40
        item = record(evidence={
            "error_excerpt": [long_chinese, "another error event"],
            "patch_excerpt": patch,
            "accepted": True,
        }).normalized().__dict__
        provider = FakeEmbeddingProvider()
        policy = ChunkPolicy(max_tokens=120, overlap_tokens=10)
        chunks = memory_chunks(item, token_count=provider.count_tokens, policy=policy)
        errors = [part for part in chunks if part.source_field == "evidence.error_excerpt[0]"]
        self.assertGreater(len(errors), 1)
        self.assertTrue(all(provider.count_tokens(embedding_input(item["operator"], part)) <= 120 for part in chunks))
        self.assertTrue(all(len(part.text) <= 600 for part in chunks))
        self.assertTrue(any(part.source_field == "evidence.patch_excerpt" for part in chunks))
        self.assertFalse(any("another error event" in part.text for part in errors))
        patch_chunks = [part for part in chunks if part.source_field == "evidence.patch_excerpt"]
        self.assertIn("@@ -1,2 +1,2 @@", "".join(part.text for part in patch_chunks))
        self.assertTrue(all(part.text.count("@@ -1,2 +1,2 @@") <= 1 for part in patch_chunks))
        query = MemoryQuery(
            operator="demo", semantics="", pytorch_reference="",
            diagnostic_text=long_chinese,
        )
        sections = query_embedding_sections(
            query, token_count=provider.count_tokens, policy=policy,
        )
        self.assertGreater(len(sections), 2)
        self.assertTrue(all(provider.count_tokens(
            f"operator: demo\nsection: {kind}\nsource: query\n" + text
        ) <= 120 for kind, text in sections))

    def test_missing_fields_do_not_create_actions_or_validation(self) -> None:
        item = record(
            memory_type="failure-diagnosis", outcome="failed", summary="possibly bad mask",
            evidence={"error_excerpt": ["out of bounds"]},
        ).normalized().__dict__
        chunks = memory_chunks(item)
        joined = " ".join(part.text for part in chunks)
        self.assertNotIn("Recommended action", joined)
        self.assertNotIn("Validation result", joined)
        self.assertIn("Reported cause (not verified)", joined)

    def test_verified_repair_episode_keeps_failure_patch_and_regression_distinct(self) -> None:
        patch = "@@ -1 +1 @@\n-old\n+new"
        item = record(
            memory_type="successful-repair",
            outcome="passed",
            evidence={
                "repair_episode": {
                    "initial_failure": {
                        "stage": "llvm-ir",
                        "command": "python -m pytest test_demo.py",
                        "error_excerpt": ["Dialect linalg not found"],
                    },
                    "diagnosis": {
                        "verified_cause": "named math lowering left linalg.generic",
                    },
                    "repair": {
                        "applied_action": "replace unsupported math operation",
                        "patch": patch,
                    },
                    "validation": {
                        "summary": "3 passed",
                        "regression_command": "python -m pytest test_demo.py -q",
                    },
                }
            },
        ).normalized().__dict__
        chunks = memory_chunks(item)
        fields = {part.source_field for part in chunks}
        self.assertIn(
            "evidence.repair_episode.initial_failure.error_excerpt[0]", fields
        )
        self.assertIn(
            "evidence.repair_episode.diagnosis.verified_cause", fields
        )
        self.assertIn("evidence.repair_episode.repair.patch", fields)
        self.assertIn("evidence.repair_episode.validation.summary", fields)
        self.assertNotIn("evidence.patch_excerpt", fields)
        joined = "\n".join(part.text for part in chunks)
        self.assertIn("Cause verified by applied patch and passing regression", joined)
        self.assertIn("3 passed", joined)

    def test_retrieval_merges_chunks_by_parent_and_context_contains_key_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryStore(Path(temp_dir) / "memory.sqlite3", FakeEmbeddingProvider()) as store:
                memory_id, _ = store.add(record(evidence={
                    "error_excerpt": ["dialect missing\n  linalg.generic"],
                    "recommended_actions": ["Try lowering to supported operations"],
                    "applied_action": "replace unsupported lowering",
                    "accepted": True,
                    "patch_path": "repair-1.patch",
                    "patch_excerpt": "@@ -1 +1 @@\n-old\n+new",
                    "test_summary": "6 passed",
                }))
                found = store.retrieve(MemoryQuery(
                    operator="tanh_and_mul", semantics="Compute tanh(x) multiplied by y.",
                    pytorch_reference="torch.tanh(x) * y", diagnostic_text="dialect missing",
                ), score_mode="embedding", limit=5)
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["id"], memory_id)
                self.assertEqual(found[0]["matched_evidence"]["source_run"], found[0]["source_run"])
                context = render_memory_context(found, max_chars=900)
                self.assertLessEqual(len(context), 900)
                for phrase in ("observed_error=", "applied_action=replace unsupported lowering",
                               "validation_result=6 passed", "recommended_action_not_executed=",
                               "source=agent-results/development/run-1"):
                    self.assertIn(phrase, context)

    def test_schema_upgrade_keeps_parent_id_and_rebuilds_children(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            with MemoryStore(path) as store:
                memory_id, _ = store.add(record())
                store.connection.execute("UPDATE metadata SET value = '3' WHERE key = 'schema_version'")
                store.connection.execute("UPDATE memory_chunks SET text = 'old flattened chunk'")
                store.connection.commit()
            with MemoryStore(path) as store:
                self.assertEqual(store.list()[0]["id"], memory_id)
                self.assertEqual(store.list()[0]["source_run"], "agent-results/development/run-1")
                self.assertFalse(any(row["text"] == "old flattened chunk" for row in
                                     store.connection.execute("SELECT text FROM memory_chunks")))

    def test_reingest_restores_available_line_breaks_without_new_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryStore(Path(temp_dir) / "memory.sqlite3") as store:
                old_id, _ = store.add(record(
                    summary="failure at line two",
                    evidence={"error_excerpt": ["failure at line two"]},
                ))
                new_id, created = store.add(record(
                    summary="failure at\nline two",
                    evidence={"error_excerpt": ["failure at\nline two"]},
                ))
                self.assertEqual(new_id, old_id)
                self.assertFalse(created)
                self.assertEqual(store.list()[0]["summary"], "failure at\nline two")
                self.assertIn("\nline two", " ".join(
                    row["text"] for row in store.connection.execute(
                        "SELECT text FROM memory_chunks WHERE memory_id = ?", (old_id,)
                    )
                ))

    def test_version_two_database_without_chunk_table_migrates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.sqlite3"
            with MemoryStore(path) as store:
                memory_id, _ = store.add(record())
                store.connection.execute("DROP TABLE memory_chunks")
                store.connection.execute("UPDATE metadata SET value = '2' WHERE key = 'schema_version'")
                store.connection.commit()
            with MemoryStore(path) as store:
                self.assertEqual(store.list()[0]["id"], memory_id)
                self.assertGreater(store.connection.execute(
                    "SELECT COUNT(*) FROM memory_chunks WHERE memory_id = ?", (memory_id,)
                ).fetchone()[0], 0)

    def test_openai_compatible_embedding_without_tokenizer_fails_closed(self) -> None:
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="http://localhost/v1", api_key="fake", model="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryStore(Path(temp_dir) / "memory.sqlite3", provider) as store:
                with self.assertRaisesRegex(RuntimeError, "tokenizer"):
                    store.add(record())

    def test_final_status_without_passed_validation_is_not_a_success_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir)
            (run / "operator-spec.json").write_text(
                json.dumps({"name": "demo", "semantics": "", "pytorch_reference": ""}),
                encoding="utf-8",
            )
            (run / "final-result.json").write_text(
                json.dumps({"operator": "demo", "status": "passed", "validations": []}),
                encoding="utf-8",
            )
            self.assertEqual(records_from_run(run), [])

    def test_missing_repair_history_does_not_claim_patch_was_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir)
            (run / "operator-spec.json").write_text(
                json.dumps({"name": "demo", "semantics": "", "pytorch_reference": ""}),
                encoding="utf-8",
            )
            (run / "repair-1.patch").write_text("@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
            (run / "final-result.json").write_text(json.dumps({
                "operator": "demo", "status": "passed", "repair_attempts": 1,
                "validations": [
                    {"status": "failed", "error_excerpt": ["compiler error"]},
                    {"status": "passed", "test_summary": "1 passed"},
                ],
            }), encoding="utf-8")
            cases = records_from_run(run)
            self.assertEqual({case.memory_type for case in cases},
                             {"failure-diagnosis", "successful-run"})


if __name__ == "__main__":
    unittest.main()
