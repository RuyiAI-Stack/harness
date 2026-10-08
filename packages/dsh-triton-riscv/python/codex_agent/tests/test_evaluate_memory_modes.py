from __future__ import annotations

import unittest
from pathlib import Path

from codex_agent.evaluate_memory_modes import evaluate_modes, load_fixture, render_markdown


class FakeEmbeddingProvider:
    name = "fake"
    model = "fake-v1"
    max_input_tokens = 2048

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float("linalg" in text.lower()), float("mask" in text.lower()), 1.0] for text in texts]


class RetrievalModeEvaluationTests(unittest.TestCase):
    def test_three_modes_use_same_queries_and_produce_auditable_rankings(self) -> None:
        fixture = load_fixture(Path(__file__).parent / "fixtures/memory_retrieval_cases.json")
        report = evaluate_modes(fixture, FakeEmbeddingProvider(), limit=3)

        self.assertEqual(report["query_count"], 8)
        self.assertEqual(report["corpus_count"], 8)
        self.assertGreater(report["embedding_backfill"]["embedded_chunks"], 8)
        self.assertEqual(set(report["modes"]), {"jaccard", "embedding", "fusion"})
        self.assertTrue(all(len(report["modes"][mode]["queries"]) == 8 for mode in report["modes"]))
        self.assertIn("Recall@k", render_markdown(report))

    def test_invalid_labels_do_not_silently_produce_scores(self) -> None:
        fixture = load_fixture(Path(__file__).parent / "fixtures/memory_retrieval_cases.json")
        fixture["queries"][0]["relevance"] = {"missing-case": 3}
        with self.assertRaisesRegex(ValueError, "unknown relevance source"):
            evaluate_modes(fixture, FakeEmbeddingProvider())


if __name__ == "__main__":
    unittest.main()
