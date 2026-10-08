from __future__ import annotations

import unittest
from pathlib import Path

from codex_agent.evaluate_memory_retrieval import evaluate_fixture, load_fixture


class MemoryRetrievalEvaluationTests(unittest.TestCase):
    def test_curated_queries_retrieve_the_expected_case(self) -> None:
        fixture = Path(__file__).parent / "fixtures/memory_retrieval_cases.json"
        report = evaluate_fixture(load_fixture(fixture))

        self.assertGreaterEqual(report["query_count"], 8)
        self.assertEqual(report["hit_at_1"], 1.0)
        self.assertEqual(report["hit_at_3"], 1.0)
        self.assertEqual(report["mean_reciprocal_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()
