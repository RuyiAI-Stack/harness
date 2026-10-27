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

    def test_sandbox_compiler_paths_do_not_hide_real_numerical_failure(self) -> None:
        log = ('SANDBOX_READONLY=["/tools/triton-shared-opt", "/tools/buddy-opt"]\n'
               'E AssertionError: Tensor-likes are not close!\n'
               'E Mismatched elements: 17 / 17\n8 failed in 27.89s\n')
        diagnosis = diagnose_log(log, 1)
        self.assertEqual(diagnosis["rule_id"], "numerical-mismatch")
        self.assertEqual(diagnosis["repair_scope"], "operator-source")
        self.assertEqual(diagnosis["evidence"][0]["line_number"], 3)
        self.assertNotIn("SANDBOX_READONLY", str(diagnosis["evidence"]))

    def test_metadata_filter_preserves_actual_compiler_and_sandbox_errors(self) -> None:
        header = 'SANDBOX_READONLY=["/tools/buddy-opt", "/tools/triton-shared-opt"]\n'
        compiler = diagnose_log(header + 'triton-shared-opt: error: conversion failed\n', 1)
        self.assertEqual(compiler["rule_id"], "triton-shared-generic")
        self.assertEqual(compiler["evidence"][0]["line_number"], 2)
        failure = diagnose_log(header + 'SANDBOX_ERROR=namespace setup failed\n', 1)
        self.assertNotEqual(failure["rule_id"], "triton-shared-generic")
        self.assertIn("SANDBOX_ERROR", str(failure["evidence"]))


if __name__ == "__main__":
    unittest.main()
