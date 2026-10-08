from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex_agent.diagnostic_memory import (
    memory_database_path,
    record_from_validation_receipt,
    remember_validation,
    retrieve_memories,
)
from codex_agent.memory import MemoryRecord, MemoryStore


OPERATOR = {
    "name": "demo",
    "public_functions": ["demo"],
    "torch_references": ["torch.tanh"],
    "tl_ops": ["load", "store"],
}


def failed_receipt(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "operator": "demo",
        "status": "failed",
        "dry_run": False,
        "exit_code": 1,
        "failure_stage": "mlir-translate",
        "command": "python -m pytest test_demo.py -s",
        "diagnosis": {
            "rule_id": "unlowered-linalg-translation",
            "category": "compiler-capability",
            "failure_stage": "mlir-translate",
            "summary": "An unlowered linalg.generic reached translation.",
            "confidence": 0.99,
            "evidence": [
                {
                    "line_number": 7,
                    "text": "Dialect linalg not found for linalg.generic",
                }
            ],
            "recommended_actions": ["Preserve the failing IR."],
        },
        "error_excerpt": ["Dialect linalg not found for linalg.generic"],
        "receipt_path": f"receipts/{run_id}.json",
    }


class DiagnosticMemoryTests(unittest.TestCase):
    def test_only_executed_results_become_memory(self) -> None:
        planned = failed_receipt("run-plan")
        planned.update({"status": "planned", "dry_run": True, "exit_code": None})
        self.assertIsNone(record_from_validation_receipt(planned, OPERATOR))

        record = record_from_validation_receipt(failed_receipt("run-failed"), OPERATOR)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.memory_type, "failure-diagnosis")
        self.assertEqual(record.confidence_grade, "C")
        self.assertEqual(
            record.evidence["rule_id"],
            "unlowered-linalg-translation",
        )

    def test_memory_write_is_deduplicated_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = failed_receipt("run-failed")

            first = remember_validation(root, receipt, OPERATOR)
            second = remember_validation(root, receipt, OPERATOR)

            self.assertEqual(first["status"], "recorded")
            self.assertEqual(second["status"], "deduplicated")
            self.assertEqual(first["memory_id"], second["memory_id"])
            with MemoryStore(memory_database_path(root)) as store:
                stored = store.list()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["source_run"], "operator-lifecycle:run-failed")

    def test_retrieval_finds_cross_operator_evidence_and_excludes_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with MemoryStore(memory_database_path(root)) as store:
                expected_id, _ = store.add(
                    MemoryRecord(
                        memory_type="successful-repair",
                        operator="tanh_and_mul",
                        semantics="Compute tanh and multiply elementwise.",
                        pytorch_reference="torch.tanh(x) * y",
                        summary="Avoid an unsupported linalg.generic lowering.",
                        outcome="passed",
                        confidence_grade="A",
                        source_run="development:tanh-repair",
                        tl_ops=["load", "store"],
                        failure_stage="mlir-translate",
                        evidence={
                            "rule_id": "unlowered-linalg-translation",
                            "error_excerpt": ["Dialect linalg not found"],
                            "recommended_actions": ["Use supported primitives."],
                        },
                    )
                )
            remember_validation(root, failed_receipt("run-current"), OPERATOR)

            result = retrieve_memories(
                root,
                operator_name="demo",
                operator=OPERATOR,
                failure_stage="mlir-translate",
                rule_id="unlowered-linalg-translation",
                evidence=["Dialect linalg not found for linalg.generic"],
                exclude_source_runs=("operator-lifecycle:run-current",),
            )

            self.assertEqual(result["status"], "found")
            self.assertEqual(result["items"][0]["memory_id"], expected_id)
            self.assertEqual(result["items"][0]["outcome"], "passed")
            self.assertNotEqual(
                result["items"][0]["source_run"],
                "operator-lifecycle:run-current",
            )
            case = result["items"][0]
            self.assertEqual(case["recommended_actions"], ["Use supported primitives."])
            self.assertIsNone(case["applied_action"])
            self.assertIsNone(case["validation_result"])
            self.assertEqual(case["evidence"], ["Dialect linalg not found"])

    def test_passed_receipt_does_not_invent_missing_test_summary(self) -> None:
        receipt = {
            "run_id": "passed-without-summary", "operator": "demo", "status": "passed",
            "dry_run": False, "exit_code": 0,
        }
        remembered = record_from_validation_receipt(receipt, None)
        assert remembered is not None
        self.assertEqual(remembered.semantics, "")
        self.assertIsNone(remembered.evidence["test_summary"])


if __name__ == "__main__":
    unittest.main()
