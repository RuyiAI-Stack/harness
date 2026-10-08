from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from codex_agent.discover import discover, main, render_markdown


class DiscoverTests(unittest.TestCase):
    def test_discovers_pytest_and_lit_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            examples = root / "python/examples/flaggems"
            examples.mkdir(parents=True)
            (examples / "add.py").write_text(
                """import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets) + tl.load(y + offsets))
""",
                encoding="utf-8",
            )
            (examples / "test_add.py").write_text(
                """import pytest
import torch
from .add import add_kernel

@pytest.mark.parametrize("shape", [(16,), (31,)])
def test_add(shape):
    assert torch.add(torch.ones(shape), torch.ones(shape)).shape == shape
""",
                encoding="utf-8",
            )
            (examples / "test_invalid.py").write_text("def broken(:\n", encoding="utf-8")
            lit_dir = root / "test/Conversion/TritonToStructured"
            lit_dir.mkdir(parents=True)
            (lit_dir / "add.mlir").write_text(
                """// RUN: triton-shared-opt %s --triton-to-structured | FileCheck %s
module { func.func @add() { %0 = memref.alloc() : memref<4xf32> return } }
""",
                encoding="utf-8",
            )

            result = discover(root)

            self.assertEqual(result["summary"]["pytest_targets"], 1)
            self.assertEqual(result["summary"]["lit_targets"], 1)
            pytest_target = result["pytest_targets"][0]
            self.assertEqual(pytest_target["test_functions"], ["test_add"])
            self.assertEqual(pytest_target["triton_kernels"], ["add_kernel"])
            self.assertEqual(
                pytest_target["tl_ops"],
                ["arange", "constexpr", "load", "store"],
            )
            self.assertEqual(pytest_target["parametrize_count"], 1)
            lit_target = result["lit_targets"][0]
            self.assertIn("triton-to-structured", lit_target["passes"])
            self.assertEqual(lit_target["likely_area"], "triton-to-structured")
            self.assertIn("memref", lit_target["dialect_keywords"])

    def test_cli_writes_to_path_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir) / "targets.json"
            report = Path(output_dir) / "targets.md"
            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "discover",
                    "--repo-root",
                    repo_dir,
                    "--output",
                    output.as_posix(),
                    "--report",
                    report.as_posix(),
                ],
            ), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            self.assertEqual(json.loads(output.read_text())["summary"]["total_targets"], 0)
            self.assertIn(output.as_posix(), stdout.getvalue())
            self.assertIn(report.as_posix(), stdout.getvalue())

    def test_discovers_project_targets_and_coverage_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            agent_dir = root / "codex_agent"
            agent_tests = agent_dir / "tests"
            agent_tests.mkdir(parents=True)
            (agent_dir / "tool.py").write_text("def check(): return True\n", encoding="utf-8")
            (agent_tests / "test_tool.py").write_text(
                "from codex_agent.tool import check\n\ndef test_check():\n    assert check()\n",
                encoding="utf-8",
            )

            backend = root / "backend"
            backend.mkdir()
            (backend / "orphan.py").write_text("VALUE = 1\n", encoding="utf-8")

            conversion = root / "lib/Conversion/Foo"
            conversion.mkdir(parents=True)
            (conversion / "FooPass.cpp").write_text("// pass\n", encoding="utf-8")
            (conversion / "CMakeLists.txt").write_text(
                """add_triton_library(FooPass
  FooPass.cpp
  LINK_LIBS MLIRPass
)
""",
                encoding="utf-8",
            )

            lit_dir = root / "test/Conversion/Foo"
            lit_dir.mkdir(parents=True)
            (lit_dir / "foo.mlir").write_text(
                "// RUN: triton-shared-opt %s --foo-pass | FileCheck %s\n",
                encoding="utf-8",
            )
            (root / "CMakeLists.txt").write_text(
                """add_lit_testsuite(check-foo "Run foo tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS FooPass
)
""",
                encoding="utf-8",
            )

            result = discover(root)

            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["summary"]["pytest_targets"], 1)
            self.assertEqual(result["summary"]["lit_targets"], 1)
            self.assertEqual(result["summary"]["build_targets"], 2)
            self.assertEqual(result["summary"]["total_targets"], 4)
            self.assertEqual(len({target["id"] for target in result["targets"]}), 4)

            build_by_name = {target["name"]: target for target in result["build_targets"]}
            self.assertEqual(build_by_name["check-foo"]["target_type"], "test-suite")
            self.assertEqual(build_by_name["check-foo"]["dependencies"], ["FooPass"])
            self.assertEqual(
                build_by_name["FooPass"]["source_files"],
                ["lib/Conversion/Foo/FooPass.cpp"],
            )

            mappings = {
                item["source"]: item for item in result["coverage"]["mappings"]
            }
            self.assertEqual(mappings["codex_agent/tool.py"]["status"], "tested")
            self.assertEqual(
                mappings["lib/Conversion/Foo/FooPass.cpp"]["status"], "tested"
            )
            self.assertEqual(mappings["backend/orphan.py"]["status"], "unmapped")
            report = render_markdown(result)
            self.assertIn("CMake/Ninja targets: 2", report)
            self.assertIn("backend/orphan.py", report)


if __name__ == "__main__":
    unittest.main()
