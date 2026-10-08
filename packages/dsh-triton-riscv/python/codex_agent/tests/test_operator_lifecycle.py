from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.operator_lifecycle import (
    apply_operator_repair,
    decide_repair_proposal,
    decide_validation_plan,
    diagnose_failure_run,
    propose_operator_repair,
    validate_operator_target,
)
from codex_agent.remote_executor import RemotePreflightResult
from codex_agent.validate_operator import OperatorValidationResult


ORIGINAL_SOURCE = """import triton
import triton.language as tl

@triton.jit
def demo_kernel(x):
    return x
"""

REPLACEMENT_SOURCE = """import triton
import triton.language as tl

@triton.jit
def demo_kernel(x):
    return x + 1
"""

TEST_SOURCE = """from .demo import demo_kernel

def test_demo():
    assert demo_kernel is not None
"""


class OperatorLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        operator_root = self.root / "python/examples/flaggems"
        operator_root.mkdir(parents=True)
        self.implementation = operator_root / "demo.py"
        self.test_file = operator_root / "test_demo.py"
        self.implementation.write_text(ORIGINAL_SOURCE, encoding="utf-8")
        self.test_file.write_text(TEST_SOURCE, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def failed_receipt(self) -> str:
        planned = validate_operator_target(self.root, "demo")
        receipt_path = self.root / planned.receipt_path
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update(
            {
                "status": "failed",
                "exit_code": 1,
                "failure_stage": "correctness",
                "likely_reason": "result differs from PyTorch reference",
                "error_excerpt": ["mismatched elements: 8 / 8"],
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return planned.run_id

    def test_validation_plans_by_default_and_live_run_needs_host_switch(self) -> None:
        result = validate_operator_target(self.root, "demo")

        self.assertEqual(result.status, "planned")
        self.assertIn("test_demo.py::test_demo", result.command or "")
        self.assertTrue((self.root / result.receipt_path).is_file())
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(PermissionError, "ALLOW_VALIDATION"):
                validate_operator_target(self.root, "demo", execute=True)

    def test_isolation_failure_never_requests_operator_source_repair(self):
        run_id = self.failed_receipt()
        receipt_path = next((self.root / "agent-results/operator-lifecycle/receipts").glob(f"{run_id}.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt["failure_stage"] = "isolation"
        receipt_path.write_text(json.dumps(receipt))
        diagnosis = diagnose_failure_run(self.root, run_id)
        self.assertFalse(diagnosis.source_repair_allowed)
        self.assertIn("do not edit operator source or disable isolation", diagnosis.recommended_action)

    def test_capacity_failure_never_requests_source_repair(self):
        run_id = self.failed_receipt()
        path = self.root / f"agent-results/operator-lifecycle/receipts/{run_id}.json"
        receipt = json.loads(path.read_text())
        receipt["failure_stage"] = "capacity"
        path.write_text(json.dumps(receipt))
        diagnosis = diagnose_failure_run(self.root, run_id)
        self.assertFalse(diagnosis.source_repair_allowed)
        self.assertIn("do not edit operator source", diagnosis.recommended_action)
        self.assertIn("not started", diagnosis.stop_reason)

    def test_capacity_receipt_never_becomes_operator_failure_memory(self):
        from codex_agent.validation_evidence import ValidationEvidence
        result = OperatorValidationResult(operator="demo",
            implementation_file="python/examples/flaggems/demo.py",
            test_files=["python/examples/flaggems/test_demo.py"], command="fixture",
            dry_run=False, exit_code=81, status="failed", failure_stage="capacity",
            likely_reason="no slot", error_excerpt=[], duration_seconds=30, log_path=None)
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_VALIDATION": "1"}, clear=True), \
             patch("codex_agent.operator_lifecycle.run_operator", return_value=result), \
             patch("codex_agent.operator_lifecycle.audit_validation_receipt", return_value=
                   ValidationEvidence(verdict="verified-failed", trusted=True, success=False)), \
             patch("codex_agent.operator_lifecycle.remember_validation") as remember:
            output = validate_operator_target(self.root, "demo", execute=True)
        remember.assert_not_called()
        self.assertEqual(output.memory_write["status"], "not-recorded")

    def test_cancellation_is_not_a_source_failure_or_failure_memory(self):
        run_id = self.failed_receipt()
        path = self.root / f"agent-results/operator-lifecycle/receipts/{run_id}.json"
        receipt = json.loads(path.read_text())
        receipt.update(status="cancelled", failure_stage="cancellation", exit_code=130)
        path.write_text(json.dumps(receipt))
        diagnosis = diagnose_failure_run(self.root, run_id)
        self.assertFalse(diagnosis.source_repair_allowed)
        self.assertIn("cancelled", diagnosis.stop_reason)
        result = OperatorValidationResult(operator="demo",
            implementation_file="python/examples/flaggems/demo.py",
            test_files=["python/examples/flaggems/test_demo.py"], command="fixture",
            dry_run=False, exit_code=130, status="cancelled", failure_stage="cancellation",
            likely_reason="host cancelled", error_excerpt=[], duration_seconds=1, log_path=None)
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_VALIDATION": "1"}, clear=True), \
             patch("codex_agent.operator_lifecycle.run_operator", return_value=result), \
             patch("codex_agent.operator_lifecycle.remember_validation") as remember:
            output = validate_operator_target(self.root, "demo", execute=True)
        remember.assert_not_called()
        self.assertEqual(output.memory_write["status"], "not-recorded")

    def test_live_validation_uses_host_configured_remote_executor(self) -> None:
        remote_result = OperatorValidationResult(
            operator="demo",
            implementation_file="python/examples/flaggems/demo.py",
            test_files=["python/examples/flaggems/test_demo.py"],
            command="ssh sg2044 -- pytest demo",
            dry_run=False,
            exit_code=0,
            status="passed",
            failure_stage=None,
            likely_reason=None,
            error_excerpt=[],
            duration_seconds=1.0,
            log_path="remote.log",
        )
        preflight = RemotePreflightResult(
            configured=True,
            status="passed",
            host="sg2044",
            repository="/home/lichunbo/work/triton-riscv",
            architecture="riscv64",
            exit_code=0,
        )
        environment = {
            "TRITON_RISCV_ALLOW_VALIDATION": "1",
            "TRITON_RISCV_REQUIRE_REMOTE": "1",
            "RISCV_HOST": "sg2044",
            "RISCV_REPO": "/home/lichunbo/work/triton-riscv",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "codex_agent.operator_lifecycle.run_remote_operator",
            return_value=(remote_result, preflight),
        ) as remote:
            result = validate_operator_target(self.root, "demo", execute=True)

        remote.assert_called_once()
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.execution_target, "remote")
        self.assertEqual(result.remote_preflight.architecture, "riscv64")

    def test_risky_candidate_is_blocked_before_local_or_remote_execution(self) -> None:
        self.implementation.write_text(ORIGINAL_SOURCE + '\nimport os\nos.system("true")\n')
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_VALIDATION": "1"}, clear=True), patch(
            "codex_agent.operator_lifecycle.run_operator"
        ) as local, patch("codex_agent.operator_lifecycle.run_remote_operator") as remote:
            with self.assertRaisesRegex(ValueError, "risk screening"):
                validate_operator_target(self.root, "demo", execute=True)
        local.assert_not_called()
        remote.assert_not_called()

    def test_risky_repair_is_not_stored_as_a_proposal(self) -> None:
        run_id = self.failed_receipt()
        with self.assertRaisesRegex(ValueError, "risk screening"):
            propose_operator_repair(self.root, run_id, REPLACEMENT_SOURCE + '\nimport socket\n', "fixture")

    def test_remote_plan_displays_the_command_that_will_run(self) -> None:
        environment = {
            "RISCV_HOST": "sg2044",
            "RISCV_REPO": "/home/lichunbo/work/triton-riscv",
        }
        with patch.dict(os.environ, environment, clear=True):
            result = validate_operator_target(self.root, "demo")

        self.assertEqual(result.execution_target, "remote")
        self.assertTrue((result.command or "").startswith("ssh sg2044 --"))
        self.assertIn("test_demo.py::test_demo", result.command or "")

    def test_diagnosis_and_approved_repair_lifecycle(self) -> None:
        run_id = self.failed_receipt()
        diagnosis = diagnose_failure_run(self.root, run_id)

        self.assertTrue(diagnosis.source_repair_allowed)
        proposal = propose_operator_repair(
            self.root,
            run_id,
            REPLACEMENT_SOURCE,
            "Correct the deliberately wrong kernel expression.",
        )
        self.assertEqual(proposal.status, "pending_approval")
        self.assertEqual(self.implementation.read_text(encoding="utf-8"), ORIGINAL_SOURCE)
        self.assertEqual(apply_operator_repair(self.root, proposal.proposal_id).status, "not_approved")

        decide_repair_proposal(
            self.root,
            proposal.proposal_id,
            approve=True,
            reviewer="unit-test",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                apply_operator_repair(self.root, proposal.proposal_id).status,
                "blocked",
            )
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}):
            applied = apply_operator_repair(self.root, proposal.proposal_id)

        self.assertEqual(applied.status, "applied")
        self.assertEqual(
            self.implementation.read_text(encoding="utf-8"),
            REPLACEMENT_SOURCE,
        )
        self.assertEqual(self.test_file.read_text(encoding="utf-8"), TEST_SOURCE)
        self.assertTrue((self.root / (applied.patch_path or "")).is_file())

    def test_changed_acceptance_test_blocks_an_approved_repair(self) -> None:
        run_id = self.failed_receipt()
        proposal = propose_operator_repair(
            self.root,
            run_id,
            REPLACEMENT_SOURCE,
            "Repair implementation only.",
        )
        decide_repair_proposal(
            self.root,
            proposal.proposal_id,
            approve=True,
            reviewer="unit-test",
        )
        self.test_file.write_text(TEST_SOURCE + "\n# changed\n", encoding="utf-8")

        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}):
            with self.assertRaisesRegex(RuntimeError, "acceptance test changed"):
                apply_operator_repair(self.root, proposal.proposal_id)

        self.assertEqual(self.implementation.read_text(encoding="utf-8"), ORIGINAL_SOURCE)

    def test_compiler_failure_allows_a_bounded_source_workaround(self) -> None:
        run_id = self.failed_receipt()
        receipt_path = (
            self.root / "agent-results/operator-lifecycle/receipts" / f"{run_id}.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["failure_stage"] = "buddy-opt"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        diagnosis = diagnose_failure_run(self.root, run_id)
        self.assertTrue(diagnosis.source_repair_allowed)
        self.assertEqual(diagnosis.repair_strategy, "compiler-workaround")
        proposal = propose_operator_repair(
            self.root,
            run_id,
            REPLACEMENT_SOURCE,
            "Use a semantics-preserving expression that avoids unsupported lowering.",
        )
        self.assertEqual(proposal.status, "pending_approval")
        self.assertEqual(self.implementation.read_text(encoding="utf-8"), ORIGINAL_SOURCE)

    def test_repair_stops_after_three_proposals(self) -> None:
        run_id = self.failed_receipt()
        for attempt in range(3):
            proposal = propose_operator_repair(
                self.root,
                run_id,
                REPLACEMENT_SOURCE,
                f"Repair attempt {attempt + 1}.",
            )
            self.assertEqual(proposal.status, "pending_approval")

        diagnosis = diagnose_failure_run(self.root, run_id)
        self.assertFalse(diagnosis.source_repair_allowed)
        self.assertEqual(diagnosis.repair_attempts, 3)
        self.assertEqual(diagnosis.attempts_remaining, 0)
        self.assertIn("repair limit", diagnosis.stop_reason or "")
        self.assertIn("compiler owner", diagnosis.user_action or "")

    def test_live_validation_requires_and_consumes_exact_approved_plan(self) -> None:
        plan = validate_operator_target(self.root, "demo")
        decide_validation_plan(
            self.root,
            plan.run_id,
            approve=True,
            reviewer="unit-test",
        )
        result = OperatorValidationResult(
            operator="demo",
            implementation_file="python/examples/flaggems/demo.py",
            test_files=["python/examples/flaggems/test_demo.py"],
            command=plan.command or "",
            dry_run=False,
            exit_code=0,
            status="passed",
            failure_stage=None,
            likely_reason=None,
            error_excerpt=[],
            duration_seconds=1.0,
            log_path="validation.log",
        )
        environment = {
            "TRITON_RISCV_ALLOW_VALIDATION": "1",
            "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "1",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "codex_agent.operator_lifecycle.run_operator",
            return_value=result,
        ) as runner:
            executed = validate_operator_target(
                self.root,
                "demo",
                execute=True,
                approved_run_id=plan.run_id,
            )

        self.assertEqual(executed.status, "passed")
        self.assertEqual(runner.call_count, 2)
        receipt = json.loads(
            (
                self.root
                / "agent-results/operator-lifecycle/receipts"
                / f"{plan.run_id}.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["approval"]["execution_run_id"], executed.run_id)

    def test_changed_validation_command_invalidates_approval(self) -> None:
        plan = validate_operator_target(self.root, "demo")
        decide_validation_plan(
            self.root,
            plan.run_id,
            approve=True,
            reviewer="unit-test",
        )
        self.test_file.write_text(
            TEST_SOURCE + "\ndef test_demo_second_case():\n    assert True\n",
            encoding="utf-8",
        )
        environment = {
            "TRITON_RISCV_ALLOW_VALIDATION": "1",
            "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "1",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(PermissionError, "command changed"):
                validate_operator_target(
                    self.root,
                    "demo",
                    execute=True,
                    approved_run_id=plan.run_id,
                )

    def test_changed_source_invalidates_approval_even_when_command_is_unchanged(self) -> None:
        plan = validate_operator_target(self.root, "demo")
        decide_validation_plan(
            self.root,
            plan.run_id,
            approve=True,
            reviewer="unit-test",
        )
        self.implementation.write_text(REPLACEMENT_SOURCE, encoding="utf-8")
        environment = {
            "TRITON_RISCV_ALLOW_VALIDATION": "1",
            "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "1",
        }

        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(PermissionError, "source or tests changed"):
                validate_operator_target(
                    self.root,
                    "demo",
                    execute=True,
                    approved_run_id=plan.run_id,
                )


if __name__ == "__main__":
    unittest.main()
