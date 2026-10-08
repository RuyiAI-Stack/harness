from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.validate_operator import (
    classify_log,
    extract_error_excerpt,
    load_completed_operators,
    run_operator,
)


class ValidateOperatorTests(unittest.TestCase):
    def test_classifies_success_and_timeout(self) -> None:
        self.assertEqual(classify_log("20 passed", 0), ("passed", None, None))
        self.assertEqual(classify_log("TIMEOUT", 124)[1], "timeout")

    def test_classifies_environment_and_import_failures(self) -> None:
        environment = "scripts/triton-riscv-env.sh: No such file or directory"
        self.assertEqual(classify_log(environment, 1)[1], "environment")
        self.assertEqual(classify_log("ModuleNotFoundError: triton", 1)[1], "import")

    def test_classifies_build_failure(self) -> None:
        self.assertEqual(classify_log("ERROR: Failed building wheel for triton", 1)[1], "build")

    def test_classifies_target_capability_failure(self) -> None:
        log = "ValueError: type fp8e4nv not supported in this architecture"
        status, stage, reason = classify_log(log, 1)

        self.assertEqual(status, "failed")
        self.assertEqual(stage, "target-capability")
        self.assertIn("FP8", reason)

    def test_classifies_unavailable_triton_math_api(self) -> None:
        log = (
            "triton.compiler.errors.CompilationError\n"
            "AttributeError: module triton.language.math has no attribute acos"
        )

        self.assertEqual(classify_log(log, 1)[1], "triton-frontend")

    def test_classifies_buddy_tptr_failure(self) -> None:
        log = "error: Dialect `tptr' not found for custom op 'tptr.type_offset'"

        self.assertEqual(classify_log(log, 1)[1], "buddy-opt")

    def test_classifies_buddy_ttx_failure(self) -> None:
        log = "error: Dialect `ttx' not found for custom op 'ttx.cumsum'"
        status, stage, reason = classify_log(log, 1)

        self.assertEqual(status, "failed")
        self.assertEqual(stage, "buddy-opt")
        self.assertIn("ttx", reason)

    def test_classifies_unsupported_buddy_reduction(self) -> None:
        log = "error: unsupported linalg.reduce for -lower-linalg-to-vir"
        status, stage, reason = classify_log(log, 1)

        self.assertEqual(status, "failed")
        self.assertEqual(stage, "buddy-opt")
        self.assertIn("linalg.reduce", reason)

    def test_classifies_linalg_translation_failure(self) -> None:
        log = "error: Dialect `linalg' not found for custom op 'linalg.generic'"

        self.assertEqual(classify_log(log, 1)[1], "mlir-translate")

    def test_classifies_correctness_failure(self) -> None:
        log = "torch.testing.assert_close failed: Mismatched elements: 5 / 10"

        self.assertEqual(classify_log(log, 1)[1], "correctness")

    def test_classifies_link_hardware_and_runtime_failures(self) -> None:
        self.assertEqual(classify_log("ld: undefined reference to foo", 1)[1], "link")
        self.assertEqual(classify_log("Illegal instruction (core dumped)", 1)[1], "target-capability")
        self.assertEqual(classify_log("Segmentation fault", 1)[1], "runtime")

    def test_extracts_and_deduplicates_primary_errors(self) -> None:
        log = (
            "/tmp/tmp123/ll.mlir:10: error: Dialect `linalg' not found\n"
            "/tmp/tmp456/ll.mlir:10: error: Dialect `linalg' not found\n"
            "FAILED test_example.py::test_op\n"
        )

        excerpt = extract_error_excerpt(log)

        self.assertEqual(len(excerpt), 1)
        self.assertIn("Dialect `linalg'", excerpt[0])

    def test_run_operator_records_success_skip_and_timeout(self) -> None:
        operator = {
            "name": "smoke",
            "implementation_file": "smoke.py",
            "test_files": ["test_smoke.py"],
            "validation_command": "printf '1 passed in 0.01s'",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            passed = run_operator(
                operator,
                repo_root=root,
                results_dir=results_dir,
                source_env=False,
                dry_run=False,
                timeout_seconds=5,
            )
            self.assertEqual(passed.status, "passed")
            self.assertEqual(passed.exit_code, 0)
            self.assertIn("1 passed", Path(passed.log_path).read_text())
            self.assertTrue(Path(passed.pipeline_report_path).exists())
            self.assertTrue(
                (Path(passed.pipeline_report_path).parent / "result.json").exists()
            )
            self.assertEqual(passed.correctness, "not-established")
            self.assertEqual(passed.diagnosis["status"], "passed")
            self.assertIn("TRITON_SHARED_DUMP_PATH", passed.command)

            skipped = run_operator(
                {**operator, "validation_command": ""},
                repo_root=root,
                results_dir=results_dir,
                source_env=False,
                dry_run=False,
                timeout_seconds=5,
            )
            self.assertEqual(skipped.status, "skipped")
            self.assertEqual(skipped.failure_stage, "selection")

            timeout = subprocess.TimeoutExpired(
                cmd="slow",
                timeout=1,
                output=b"partial",
            )
            with patch(
                "codex_agent.validate_operator.run_bounded",
                side_effect=timeout,
            ):
                failed = run_operator(
                    operator,
                    repo_root=root,
                    results_dir=results_dir,
                    source_env=False,
                    dry_run=False,
                    timeout_seconds=1,
                )
            self.assertEqual(failed.exit_code, 124)
            self.assertEqual(failed.failure_stage, "timeout")
            self.assertEqual(failed.diagnosis["rule_id"], "validation-timeout")
            self.assertEqual(failed.diagnosis["repair_scope"], "retry-policy")
            self.assertIn("TIMEOUT", Path(failed.log_path).read_text())
            pipeline = json.loads(Path(failed.pipeline_report_path).read_text())
            self.assertEqual(pipeline["diagnosis"]["rule_id"], "validation-timeout")
            self.assertIn(
                "## Failure Diagnosis",
                (Path(failed.pipeline_report_path).parent / "report.md").read_text(),
            )

    def test_load_completed_operators_ignores_planned_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                '{"operator":"add","status":"passed"}\n'
                '{"operator":"mul","status":"planned"}\n'
                '{"operator":"sub","status":"failed"}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_completed_operators(path), {"add", "sub"})
            self.assertEqual(load_completed_operators(path.parent / "missing.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
