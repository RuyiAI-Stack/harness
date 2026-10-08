from __future__ import annotations

import unittest
from pathlib import Path

from codex_agent.evaluate_diagnosis import evaluate_cases, load_cases
from codex_agent.failure_diagnosis import diagnose_log


FIXTURES = Path(__file__).parent / "fixtures" / "diagnosis_cases.json"


class FailureDiagnosisTests(unittest.TestCase):
    def test_curated_suite_covers_at_least_twenty_failure_cases(self) -> None:
        report = evaluate_cases(load_cases(FIXTURES))

        self.assertGreaterEqual(report["case_count"], 20)
        self.assertEqual(report["rule_accuracy"], 1.0)
        self.assertEqual(report["stage_accuracy"], 1.0)
        self.assertEqual(report["category_accuracy"], 1.0)
        self.assertEqual(report["evidence_recall"], 1.0)

    def test_evidence_keeps_line_numbers_and_deduplicates_temp_paths(self) -> None:
        diagnosis = diagnose_log(
            "/tmp/tmp123/ll.mlir:10: error: Dialect `linalg' not found\n"
            "/tmp/tmp456/ll.mlir:10: error: Dialect `linalg' not found\n",
            1,
        )

        self.assertEqual(diagnosis["rule_id"], "unlowered-linalg-translation")
        self.assertEqual(len(diagnosis["evidence"]), 1)
        self.assertEqual(diagnosis["evidence"][0]["line_number"], 1)

    def test_failed_stage_event_supplies_exact_command(self) -> None:
        diagnosis = diagnose_log(
            "mlir-translate: error: translation failed",
            1,
            stage_events=[
                {
                    "stage": "mlir-translate",
                    "status": "failed",
                    "command": ["mlir-translate", "--mlir-to-llvmir", "ll.mlir"],
                }
            ],
        )

        self.assertEqual(
            diagnosis["failed_command"],
            "mlir-translate --mlir-to-llvmir ll.mlir",
        )

    def test_stage_event_classifies_an_unrecognized_compiler_message(self) -> None:
        diagnosis = diagnose_log(
            "conversion stopped before producing output",
            1,
            stage_events=[
                {
                    "stage": "buddy-opt-lower",
                    "pipeline_stage": "llvm-mlir",
                    "status": "failed",
                    "command": ["buddy-opt", "input.mlir"],
                }
            ],
        )

        self.assertEqual(diagnosis["rule_id"], "failed-compiler-stage-event")
        self.assertEqual(diagnosis["failure_stage"], "buddy-opt")
        self.assertEqual(diagnosis["pipeline_stage"], "llvm-mlir")

    def test_success_and_plan_do_not_claim_a_failure(self) -> None:
        planned = diagnose_log("", None)
        passed = diagnose_log("1 passed", 0)

        self.assertEqual(planned["status"], "planned")
        self.assertIsNone(planned["failure_stage"])
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(passed["repair_scope"], "none")


if __name__ == "__main__":
    unittest.main()
