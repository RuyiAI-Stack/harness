from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from codex_agent.develop_operator import (
    audit_test_contract,
    choose_references,
    compare_validations,
    extract_pytest_summary,
    load_operator_spec,
    main,
    may_repair,
    plan_repair,
    restore_files,
    restore_workspace_paths,
    snapshot_workspace_bytes,
    update_task_validation_record,
    unauthorized_changes,
)


def valid_spec() -> dict:
    return {
        "schema_version": 1,
        "name": "tanh_and_mul",
        "semantics": "Compute tanh(x) multiplied elementwise by y.",
        "pytorch_reference": "torch.tanh(x) * y",
        "inputs": [
            {"name": "x", "description": "activation input"},
            {"name": "y", "description": "multiplication input"},
        ],
        "output": "tanh(x) * y",
        "shape_cases": [[512], [1023]],
        "input_shape_cases": [{"x": [4, 1], "y": [1, 8]}],
        "dtypes": ["torch.float32", "torch.float16"],
        "tolerances": {"rtol": 0.01, "atol": 0.01},
        "backward": True,
        "reference_operators": ["sigmoid_and_mul"],
    }


def valid_test_source() -> str:
    return """import pytest
import torch
from .tanh_and_mul import tanh_and_mul, tanh_and_mul_backward

@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    y = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)

def test_broadcast():
    x = torch.randn((4, 1), dtype=torch.float32)
    y = torch.randn((1, 8), dtype=torch.float32)
    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)

def test_backward():
    x = torch.randn((512,), requires_grad=True)
    y = torch.randn((512,), requires_grad=True)
    expected = torch.tanh(x) * y
    expected.backward(torch.ones_like(expected))
    dx, dy = tanh_and_mul_backward(torch.ones_like(expected), x.detach(), y.detach())
    torch.testing.assert_close(dx, x.grad, rtol=0.01, atol=0.01)
    torch.testing.assert_close(dy, y.grad, rtol=0.01, atol=0.01)
"""


class DevelopOperatorTests(unittest.TestCase):
    def test_loads_spec_and_applies_safe_default_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")

            spec = load_operator_spec(spec_path, root)

            self.assertEqual(
                spec.implementation_file,
                "python/examples/flaggems/tanh_and_mul.py",
            )
            self.assertEqual(
                spec.test_file,
                "python/examples/flaggems/test_tanh_and_mul.py",
            )

    def test_contract_audit_requires_reference_coverage_and_backward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            test_path = root / spec.test_file
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                """import pytest
import torch

from .tanh_and_mul import tanh_and_mul, tanh_and_mul_backward


@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tanh_and_mul_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    y = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


def test_tanh_and_mul_broadcast():
    x = torch.randn((4, 1), dtype=torch.float32)
    y = torch.randn((1, 8), dtype=torch.float32)
    expected = torch.tanh(x) * y
    actual = tanh_and_mul(x, y)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


def test_tanh_and_mul_backward():
    x = torch.randn((512,), requires_grad=True)
    y = torch.randn((512,), requires_grad=True)
    expected = torch.tanh(x) * y
    expected.backward(torch.ones_like(expected))
    dx, dy = tanh_and_mul_backward(torch.ones_like(expected), x.detach(), y.detach())
    torch.testing.assert_close(dx, x.grad, rtol=0.01, atol=0.01)
    torch.testing.assert_close(dy, y.grad, rtol=0.01, atol=0.01)
""",
                encoding="utf-8",
            )

            audit = audit_test_contract(spec, root)

            self.assertEqual(audit.status, "passed")
            self.assertEqual(audit.errors, [])
            self.assertIsNotNone(audit.test_sha256)

    def test_contract_audit_rejects_weakened_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_data = valid_spec()
            spec_data["backward"] = False
            spec_data["input_shape_cases"] = []
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            test_path = root / spec.test_file
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                """import pytest
import torch
from .tanh_and_mul import tanh_and_mul
@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * x
    actual = tanh_and_mul(x, x)
    torch.testing.assert_close(actual, expected, rtol=1.0, atol=1.0)
""",
                encoding="utf-8",
            )

            audit = audit_test_contract(spec, root)

            self.assertEqual(audit.status, "failed")
            self.assertTrue(any("exceeds" in error for error in audit.errors))

    def test_reference_selection_prefers_explicit_and_fused_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            inventory = {
                "operators": [
                    {
                        "name": "add",
                        "implementation_file": "add.py",
                        "torch_references": ["torch.add"],
                        "tl_ops": ["load", "store"],
                        "public_functions": ["add"],
                    },
                    {
                        "name": "sigmoid_and_mul",
                        "implementation_file": "sigmoid_and_mul.py",
                        "torch_references": ["torch.sigmoid"],
                        "tl_ops": ["exp", "load", "store"],
                        "public_functions": ["sigmoid_and_mul_backward"],
                    },
                ]
            }

            references = choose_references(spec, inventory)

            self.assertEqual(references[0]["name"], "sigmoid_and_mul")

    def test_repair_policy_stops_for_environment_and_locks_compiler_by_default(self) -> None:
        self.assertTrue(may_repair("correctness", False))
        self.assertFalse(may_repair("environment", True))
        self.assertFalse(may_repair("buddy-opt", False))
        self.assertTrue(may_repair("buddy-opt", True))

        compiler = plan_repair(
            {
                "status": "failed",
                "failure_stage": "mlir-translate",
                "first_failure_stage": "llvm-ir",
            },
            True,
        )
        hardware = plan_repair(
            {
                "status": "failed",
                "failure_stage": "target-capability",
                "first_failure_stage": "hardware-capability",
            },
            True,
        )

        self.assertEqual(compiler.action, "repair")
        self.assertEqual(compiler.category, "compiler-workaround")
        self.assertEqual(hardware.action, "stop")
        self.assertEqual(hardware.category, "external-blocker")

    def test_validation_comparison_accepts_progress_and_rejects_regression(self) -> None:
        correctness = {
            "status": "failed",
            "first_failure_stage": "correctness",
            "likely_reason": "mismatch",
            "error_excerpt": ["mismatched elements"],
        }
        frontend = {
            "status": "failed",
            "first_failure_stage": "ttir",
            "likely_reason": "frontend",
            "error_excerpt": ["CompilationError"],
        }
        passed = {"status": "passed", "first_failure_stage": None}

        self.assertEqual(compare_validations(correctness, passed)[:2], ("passed", True))
        self.assertEqual(
            compare_validations(correctness, frontend)[:2],
            ("regressed", False),
        )
        self.assertEqual(
            compare_validations(correctness, correctness)[:2],
            ("ineffective", False),
        )

    def test_restore_files_restores_content_and_removes_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "existing.py"
            created = root / "created.py"
            existing.write_text("before\n", encoding="utf-8")
            snapshot = {"existing.py": "before\n", "created.py": None}
            existing.write_text("after\n", encoding="utf-8")
            created.write_text("new\n", encoding="utf-8")

            restore_files(root, snapshot)

            self.assertEqual(existing.read_text(encoding="utf-8"), "before\n")
            self.assertFalse(created.exists())

    def test_workspace_snapshot_restores_unauthorized_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protected = root / "README.md"
            protected.write_bytes(b"user content\n")
            state = {"README.md": "before-hash"}
            snapshot = snapshot_workspace_bytes(root, state)
            protected.write_bytes(b"model content\n")

            failed = restore_workspace_paths(
                root,
                state,
                snapshot,
                ["README.md"],
            )

            self.assertEqual(failed, [])
            self.assertEqual(protected.read_bytes(), b"user content\n")

    def test_detects_changes_outside_allowlist(self) -> None:
        before = {"implementation.py": "a", "README.md": "old"}
        after = {"implementation.py": "b", "README.md": "new", "extra.py": "x"}

        changed = unauthorized_changes(before, after, {"implementation.py"})

        self.assertEqual(changed, ["README.md", "extra.py"])

    def test_extracts_latest_pytest_summary(self) -> None:
        text = "first\n2 failed, 3 passed in 1.25s\nretry\n20 passed in 16.79s\n"

        self.assertEqual(extract_pytest_summary(text), "20 passed in 16.79s")

    def test_rejects_operator_path_outside_allowlisted_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_data = valid_spec()
            spec_data["implementation_file"] = "../outside.py"
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec_data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "repository-relative"):
                load_operator_spec(spec_path, root)

    def test_contract_audit_requires_declared_broadcast_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            spec = load_operator_spec(spec_path, root)
            test_path = root / spec.test_file
            test_path.parent.mkdir(parents=True)
            test_path.write_text(
                """import pytest
import torch
from .tanh_and_mul import tanh_and_mul, tanh_and_mul_backward
@pytest.mark.parametrize("shape", [(512,), (1023,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    expected = torch.tanh(x) * x
    actual = tanh_and_mul(x, x)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
def test_backward():
    x = torch.randn((512,), requires_grad=True)
    expected = torch.tanh(x) * x
    expected.backward(torch.ones_like(expected))
    dx, dy = tanh_and_mul_backward(torch.ones_like(expected), x, x)
    torch.testing.assert_close(dx, x.grad, rtol=0.01, atol=0.01)
""",
                encoding="utf-8",
            )

            audit = audit_test_contract(spec, root)

            self.assertEqual(audit.status, "failed")
            self.assertTrue(any("input-shape case" in item for item in audit.errors))

    def test_task_validation_record_is_replaced_instead_of_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task_path = Path(temp_dir) / "task.md"
            task_path.write_text("# Task\n", encoding="utf-8")
            final = {
                "status": "passed",
                "locked_test_sha256": "abc",
                "repair_attempts": 1,
                "validations": [
                    {
                        "iteration": 1,
                        "status": "passed",
                        "failure_stage": None,
                        "test_summary": "2 passed in 1.00s",
                        "error_excerpt": [],
                    }
                ],
            }

            update_task_validation_record(task_path, final)
            update_task_validation_record(task_path, final)
            content = task_path.read_text(encoding="utf-8")

            self.assertEqual(content.count("Autonomous Validation Record"), 1)

    def test_closed_loop_rolls_back_regression_then_accepts_passing_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("results/\n", encoding="utf-8")
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            implementation_path = root / "python/examples/flaggems/tanh_and_mul.py"
            test_path = root / "python/examples/flaggems/test_tanh_and_mul.py"
            implementation_path.parent.mkdir(parents=True)
            implementation_path.write_text("state = 'broken'\n", encoding="utf-8")
            test_path.write_text(valid_test_source(), encoding="utf-8")
            locked_test = test_path.read_text(encoding="utf-8")

            def fake_codex(_root, _prompt, _run_dir, label, _timeout):
                if label == "codex-repair-1":
                    implementation_path.write_text("state = 'worse'\n", encoding="utf-8")
                elif label == "codex-repair-2":
                    self.assertEqual(
                        implementation_path.read_text(encoding="utf-8"),
                        "state = 'broken'\n",
                    )
                    implementation_path.write_text("state = 'fixed'\n", encoding="utf-8")
                return {"status": "passed", "exit_code": 0}

            validations = [
                {
                    "operator": "tanh_and_mul",
                    "iteration": 1,
                    "command": "pytest",
                    "status": "failed",
                    "exit_code": 1,
                    "failure_stage": "correctness",
                    "first_failure_stage": "correctness",
                    "likely_reason": "result differs from reference",
                    "error_excerpt": ["mismatched elements"],
                    "test_summary": "1 failed in 1.0s",
                    "correctness": "basic-tests-failed",
                    "stages": [],
                },
                {
                    "operator": "tanh_and_mul",
                    "iteration": 2,
                    "command": "pytest",
                    "status": "failed",
                    "exit_code": 1,
                    "failure_stage": "triton-frontend",
                    "first_failure_stage": "ttir",
                    "likely_reason": "frontend compilation failed",
                    "error_excerpt": ["CompilationError"],
                    "test_summary": "1 failed in 1.0s",
                    "correctness": "not-established",
                    "stages": [],
                },
                {
                    "operator": "tanh_and_mul",
                    "iteration": 3,
                    "command": "pytest",
                    "status": "passed",
                    "exit_code": 0,
                    "failure_stage": None,
                    "first_failure_stage": None,
                    "likely_reason": None,
                    "error_excerpt": [],
                    "test_summary": "20 passed in 3.0s",
                    "correctness": "basic-tests-passed",
                    "stages": [],
                },
            ]

            with patch(
                "sys.argv",
                [
                    "develop-operator",
                    "--spec",
                    spec_path.as_posix(),
                    "--repo-root",
                    temp_dir,
                    "--results-dir",
                    "results",
                    "--allow-existing",
                    "--max-generation-attempts",
                    "1",
                    "--max-repairs",
                    "2",
                ],
            ), patch(
                "codex_agent.develop_operator.discover_operators",
                return_value={"operators": []},
            ), patch(
                "codex_agent.develop_operator.run_preflight",
                return_value={"status": "passed", "exit_code": 0},
            ), patch(
                "codex_agent.develop_operator.collect_environment",
                return_value={
                    "status": "ready",
                    "architecture": "riscv64",
                    "execution": {"mode": "native-riscv"},
                },
            ), patch(
                "codex_agent.develop_operator.run_codex",
                side_effect=fake_codex,
            ), patch(
                "codex_agent.develop_operator.run_validation",
                side_effect=validations,
            ), redirect_stdout(StringIO()):
                exit_code = main()

            final_path = next((root / "results").glob("*/final-result.json"))
            final = json.loads(final_path.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(final["status"], "passed")
            self.assertEqual(final["accepted_repairs"], 1)
            self.assertEqual(final["rejected_repairs"], 1)
            self.assertEqual(final["repair_history"][0]["outcome"], "regressed")
            self.assertEqual(final["repair_history"][1]["outcome"], "passed")
            self.assertEqual(
                implementation_path.read_text(encoding="utf-8"),
                "state = 'fixed'\n",
            )
            self.assertEqual(test_path.read_text(encoding="utf-8"), locked_test)
            self.assertTrue((final_path.parent / "repair-summary.md").exists())

    def test_closed_loop_restores_locked_test_modified_by_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("results/\n", encoding="utf-8")
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            implementation_path = root / "python/examples/flaggems/tanh_and_mul.py"
            test_path = root / "python/examples/flaggems/test_tanh_and_mul.py"
            implementation_path.parent.mkdir(parents=True)
            implementation_path.write_text("state = 'broken'\n", encoding="utf-8")
            test_path.write_text(valid_test_source(), encoding="utf-8")
            locked_test = test_path.read_text(encoding="utf-8")

            def fake_codex(_root, _prompt, _run_dir, label, _timeout):
                if label == "codex-repair-1":
                    implementation_path.write_text("state = 'cheat'\n", encoding="utf-8")
                    test_path.write_text("def test_weakened(): pass\n", encoding="utf-8")
                return {"status": "passed", "exit_code": 0}

            failed_validation = {
                "operator": "tanh_and_mul",
                "iteration": 1,
                "command": "pytest",
                "status": "failed",
                "exit_code": 1,
                "failure_stage": "correctness",
                "first_failure_stage": "correctness",
                "likely_reason": "result differs from reference",
                "error_excerpt": ["mismatched elements"],
                "test_summary": "1 failed in 1.0s",
                "correctness": "basic-tests-failed",
                "stages": [],
            }

            with patch(
                "sys.argv",
                [
                    "develop-operator",
                    "--spec",
                    spec_path.as_posix(),
                    "--repo-root",
                    temp_dir,
                    "--results-dir",
                    "results",
                    "--allow-existing",
                    "--max-generation-attempts",
                    "1",
                    "--max-repairs",
                    "1",
                ],
            ), patch(
                "codex_agent.develop_operator.discover_operators",
                return_value={"operators": []},
            ), patch(
                "codex_agent.develop_operator.run_preflight",
                return_value={"status": "passed", "exit_code": 0},
            ), patch(
                "codex_agent.develop_operator.collect_environment",
                return_value={"status": "ready", "execution": {}},
            ), patch(
                "codex_agent.develop_operator.run_codex",
                side_effect=fake_codex,
            ), patch(
                "codex_agent.develop_operator.run_validation",
                return_value=failed_validation,
            ), redirect_stdout(StringIO()):
                exit_code = main()

            final_path = next((root / "results").glob("*/final-result.json"))
            final = json.loads(final_path.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 5)
            self.assertEqual(final["status"], "repair-exhausted")
            self.assertEqual(
                final["repair_history"][0]["outcome"],
                "locked-test-modified",
            )
            self.assertEqual(
                implementation_path.read_text(encoding="utf-8"),
                "state = 'broken'\n",
            )
            self.assertEqual(test_path.read_text(encoding="utf-8"), locked_test)

    def test_prepare_only_generates_task_and_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "develop-operator",
                    "--spec",
                    spec_path.as_posix(),
                    "--repo-root",
                    temp_dir,
                    "--results-dir",
                    "results",
                    "--prepare-only",
                ],
            ), patch(
                "codex_agent.develop_operator.run_preflight",
                return_value={"status": "skipped", "reason": "test"},
            ), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            task = root / "tasks/operators/tanh_and_mul.md"
            self.assertTrue(task.exists())
            self.assertIn("tanh_and_mul", task.read_text())
            run_dirs = list((root / "results").iterdir())
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "operator-spec.json").exists())
            self.assertTrue((run_dirs[0] / "references.json").exists())


if __name__ == "__main__":
    unittest.main()
