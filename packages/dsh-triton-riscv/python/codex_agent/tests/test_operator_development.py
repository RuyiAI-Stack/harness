from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.operator_development import (
    apply_operator_implementation,
    decide_operator_development_proposal,
    get_operator_development_proposal,
    prepare_operator_development,
    propose_operator_implementation,
)


def valid_spec() -> dict:
    return {
        "schema_version": 1,
        "name": "square_new",
        "semantics": "Compute the elementwise square of the input tensor.",
        "pytorch_reference": "torch.square(x)",
        "inputs": [{"name": "x", "description": "Floating-point input tensor."}],
        "output": "A tensor with the same shape as x containing x squared.",
        "shape_cases": [[8], [17]],
        "input_shape_cases": [],
        "dtypes": ["torch.float32"],
        "tolerances": {"rtol": 0.0001, "atol": 0.0001},
        "backward": False,
        "reference_operators": [],
        "notes": "Use explicit masks for the non-power-of-two case.",
    }


IMPLEMENTATION = """import torch
import triton
import triton.language as tl


@triton.jit
def square_new_kernel(x_ptr, out_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    values = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, values * values)


def square_new(x):
    output = torch.empty_like(x)
    square_new_kernel[(1,)](x, output, x.numel())
    return output
"""


TEST_SOURCE = """import pytest
import torch

from .square_new import square_new


@pytest.mark.parametrize("shape", [(8,), (17,)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_square_new(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    expected = torch.square(x)
    actual = square_new(x)
    torch.testing.assert_close(actual, expected, rtol=0.0001, atol=0.0001)
"""


class OperatorDevelopmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "python/examples/flaggems").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_propose_approve_and_apply_new_operator(self) -> None:
        plan = prepare_operator_development(self.root, valid_spec())

        self.assertEqual(plan.status, "planned")
        self.assertFalse((self.root / plan.implementation_file).exists())
        self.assertFalse((self.root / plan.test_file).exists())
        proposal = propose_operator_implementation(
            self.root,
            plan.development_id,
            IMPLEMENTATION,
            TEST_SOURCE,
            "Implement a masked elementwise kernel and independent PyTorch test.",
        )
        self.assertEqual(proposal.status, "pending_approval")
        self.assertFalse((self.root / plan.implementation_file).exists())
        self.assertEqual(
            apply_operator_implementation(self.root, proposal.proposal_id).status,
            "not_approved",
        )

        reviewed = decide_operator_development_proposal(
            self.root,
            proposal.proposal_id,
            approve=True,
            reviewer="unit-test",
        )
        self.assertEqual(reviewed["status"], "approved")
        self.assertNotIn("implementation_source", reviewed)
        self.assertEqual(
            apply_operator_implementation(self.root, proposal.proposal_id).status,
            "blocked",
        )
        with patch.dict(
            "os.environ",
            {"TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": "1"},
        ):
            applied = apply_operator_implementation(self.root, proposal.proposal_id)

        self.assertEqual(applied.status, "applied")
        self.assertEqual(
            set(applied.created_files),
            {
                "python/examples/flaggems/square_new.py",
                "python/examples/flaggems/test_square_new.py",
                "tasks/operators/square_new.md",
            },
        )
        self.assertIn("@triton.jit", (self.root / plan.implementation_file).read_text())
        self.assertIn("Immutable Contract", (self.root / plan.task_file).read_text())
        with patch.dict("os.environ", {"TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": "1"}):
            repeated = apply_operator_implementation(self.root, proposal.proposal_id)
        self.assertEqual(repeated.status, "applied")
        self.assertIn("validate_operator", repeated.message)

    def test_contract_audit_rejects_a_weakened_generated_test(self) -> None:
        plan = prepare_operator_development(self.root, valid_spec())
        weak_test = TEST_SOURCE.replace("torch.square(x)", "x * x").replace(
            "rtol=0.0001", "rtol=1.0"
        )

        with self.assertRaisesRegex(ValueError, "test contract audit failed"):
            propose_operator_implementation(
                self.root,
                plan.development_id,
                IMPLEMENTATION,
                weak_test,
                "This should be rejected.",
            )

    def test_risky_generated_implementation_or_test_is_rejected(self) -> None:
        plan = prepare_operator_development(self.root, valid_spec())
        for implementation, test in (
            (IMPLEMENTATION + '\nimport subprocess\n', TEST_SOURCE),
            (IMPLEMENTATION, TEST_SOURCE + '\nfrom pathlib import Path\nPath("target").unlink()\n'),
        ):
            with self.subTest(implementation=implementation), self.assertRaisesRegex(ValueError, "risk screening"):
                propose_operator_implementation(self.root, plan.development_id, implementation, test, "fixture")

    def test_existing_target_is_not_treated_as_a_new_operator(self) -> None:
        path = self.root / "python/examples/flaggems/square_new.py"
        path.write_text("# existing\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "refuses existing files"):
            prepare_operator_development(self.root, valid_spec())

    def test_changed_target_blocks_an_approved_proposal(self) -> None:
        plan = prepare_operator_development(self.root, valid_spec())
        proposal = propose_operator_implementation(
            self.root,
            plan.development_id,
            IMPLEMENTATION,
            TEST_SOURCE,
            "Create a new implementation.",
        )
        decide_operator_development_proposal(
            self.root,
            proposal.proposal_id,
            approve=True,
            reviewer="unit-test",
        )
        target = self.root / plan.implementation_file
        target.write_text("# created concurrently\n", encoding="utf-8")

        with patch.dict(
            "os.environ",
            {"TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": "1"},
        ):
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                apply_operator_implementation(self.root, proposal.proposal_id)

    def test_review_view_hides_generated_sources(self) -> None:
        plan = prepare_operator_development(self.root, valid_spec())
        proposal = propose_operator_implementation(
            self.root,
            plan.development_id,
            IMPLEMENTATION,
            TEST_SOURCE,
            "Create a new implementation.",
        )

        review = get_operator_development_proposal(self.root, proposal.proposal_id)

        self.assertNotIn("implementation_source", review)
        self.assertNotIn("test_source", review)
        self.assertIn("diff", review)


if __name__ == "__main__":
    unittest.main()
