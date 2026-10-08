from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex_agent.discover_operators import (
    build_validation_command,
    discover_operators,
    infer_risk_hints,
    select_relevant_tests,
)


class DiscoverOperatorsTests(unittest.TestCase):
    def test_discovers_operator_contract_and_skips_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            operator_dir = root / "python/examples/flaggems"
            operator_dir.mkdir(parents=True)
            (operator_dir / "sigmoid_and_mul.py").write_text(
                """import triton
import triton.language as tl

@triton.jit
def sigmoid_and_mul_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    values = tl.load(x + offsets)
    tl.store(out + offsets, tl.where(values > 0, tl.exp(values), values))

def sigmoid_and_mul(x, y):
    return x
""",
                encoding="utf-8",
            )
            (operator_dir / "test_sigmoid_and_mul.py").write_text(
                """import pytest
import torch
from .sigmoid_and_mul import sigmoid_and_mul

@pytest.mark.parametrize("shape", [(16,), (31,)])
def test_sigmoid_and_mul_forward(shape):
    expected = torch.sigmoid(torch.ones(shape)) * torch.ones(shape)
    assert expected.shape == shape

def test_unrelated():
    pass
""",
                encoding="utf-8",
            )
            (operator_dir / "helper.py").write_text("def helper():\n    pass\n", encoding="utf-8")
            (operator_dir / "bad-name.py").write_text(
                "import triton\n@triton.jit\ndef kernel():\n    pass\n",
                encoding="utf-8",
            )

            result = discover_operators(root)

            self.assertEqual(result["summary"]["operators"], 1)
            self.assertEqual(result["summary"]["operators_with_tests"], 1)
            self.assertEqual(result["skipped_invalid_operator_files"], ["bad-name.py"])
            operator = result["operators"][0]
            self.assertEqual(operator["name"], "sigmoid_and_mul")
            self.assertEqual(operator["public_functions"], ["sigmoid_and_mul"])
            self.assertEqual(operator["test_nodes"], [
                "python/examples/flaggems/test_sigmoid_and_mul.py::test_sigmoid_and_mul_forward"
            ])
            self.assertTrue(operator["validation_command"].endswith(" -s"))
            self.assertTrue(any("transcendental" in hint for hint in operator["risk_hints"]))
            self.assertTrue(any("masks" in hint for hint in operator["risk_hints"]))
            self.assertTrue(any(ref.startswith("torch.sigmoid") for ref in operator["torch_references"]))
            self.assertTrue(operator["test_contract"]["pytorch_reference"])
            self.assertFalse(operator["test_contract"]["numerical_assertion"])
            self.assertEqual(operator["test_contract"]["selected_test_count"], 1)
            self.assertTrue(operator["test_contract"]["parameterized"])

    def test_selection_and_command_helpers_cover_empty_and_fused_cases(self) -> None:
        self.assertEqual(
            select_relevant_tests(
                "relu_and_mul",
                ["test_relu_forward", "test_relu_and_mul_backward", "test_add"],
            ),
            ["test_relu_and_mul_backward", "test_relu_forward"],
        )
        self.assertEqual(build_validation_command([], []), "")
        self.assertEqual(
            build_validation_command([], ["test_one.py"]),
            "python -m pytest -q test_one.py -s",
        )
        hints = infer_risk_hints(["dot", "load", "store"])
        self.assertTrue(any("matrix" in hint for hint in hints))
        self.assertTrue(any("boundary" in hint for hint in hints))

    def test_contract_does_not_treat_random_setup_as_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            operator_dir = root / "python/examples/flaggems"
            operator_dir.mkdir(parents=True)
            (operator_dir / "smoke.py").write_text(
                "import triton\n@triton.jit\ndef kernel():\n    pass\n",
                encoding="utf-8",
            )
            (operator_dir / "test_smoke.py").write_text(
                "import torch\nfrom .smoke import kernel\n"
                "def test_smoke():\n    torch.manual_seed(0)\n    assert True\n",
                encoding="utf-8",
            )

            contract = discover_operators(root)["operators"][0]["test_contract"]

        self.assertFalse(contract["pytorch_reference"])


if __name__ == "__main__":
    unittest.main()
