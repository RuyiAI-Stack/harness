from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from codex_agent.harness.mcp_demo import run_mcp_demo
from codex_agent.harness.mcp_server import server
from codex_agent.tests.test_operator_development import valid_spec
from codex_agent.operator_tools import discover_operator_evidence

FIXTURE_REPOSITORY = Path(__file__).parent / "fixtures/repository"


class OperatorToolTests(unittest.TestCase):
    def test_discovers_existing_operator_evidence(self) -> None:
        result = discover_operator_evidence(FIXTURE_REPOSITORY, "relu_and_mul")

        self.assertEqual(result.status, "found")
        self.assertIsNotNone(result.operator)
        assert result.operator is not None
        self.assertEqual(
            result.operator.implementation_file,
            "python/examples/flaggems/relu_and_mul.py",
        )
        self.assertIn(
            "python/examples/flaggems/test_relu_and_mul.py",
            result.operator.test_files,
        )
        self.assertIn("load", result.operator.tl_ops)
        self.assertTrue(
            result.operator.validation_command.startswith("python -m pytest")
        )

    def test_unknown_operator_returns_bounded_suggestions(self) -> None:
        result = discover_operator_evidence(FIXTURE_REPOSITORY, "relu_and_mu")

        self.assertEqual(result.status, "not_found")
        self.assertIn("relu_and_mul", result.suggestions)
        self.assertLessEqual(len(result.suggestions), 5)

    def test_rejects_path_traversal_as_an_operator_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Python identifier"):
            discover_operator_evidence(Path.cwd(), "../relu_and_mul")


class McpOperatorToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_lists_lifecycle_and_calls_discover_operator(self) -> None:
        with patch.dict(
            os.environ,
            {"TRITON_RISCV_REPO_ROOT": str(FIXTURE_REPOSITORY)},
        ):
            async with Client(server) as client:
                listed = await client.list_tools()
                names = {item.name for item in listed.tools}
                tool = next(
                    item for item in listed.tools if item.name == "discover_operator"
                )
                result = await client.call_tool(
                    "discover_operator",
                    {"operator_name": "relu_and_mul"},
                )

        self.assertEqual(
            names,
            {
                "check_validation_environment",
                "discover_operator",
                "read_operator_file",
                "validate_operator",
                "diagnose_failure",
                "propose_repair",
                "apply_repair",
                "prepare_operator_development",
                "propose_operator_implementation",
                "apply_operator_implementation",
                "apply_development_proposal",
                "execute_approved_validation",
                "retrieve_operator_memory",
                "get_operator_implementation_proposal",
                "inspect_project",
                "prepare_validation_job",
                "execute_validation_job",
                "get_validation_job",
                "get_validation_status",
                "inspect_queued_task",
                "evaluate_plugin_fixtures",
            },
        )
        self.assertTrue(tool.annotations.read_only_hint)
        self.assertFalse(tool.annotations.destructive_hint)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "found")
        self.assertEqual(
            result.structured_content["operator"]["name"],
            "relu_and_mul",
        )

    async def test_mcp_reads_source_without_execution(self) -> None:
        with patch.dict(os.environ, {"TRITON_RISCV_REPO_ROOT": str(FIXTURE_REPOSITORY)}):
            async with Client(server) as client:
                result = await client.call_tool("read_operator_file", {
                    "path": "python/examples/flaggems/relu_and_mul.py", "max_lines": 2,
                })
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["next_line"], 3)
        self.assertEqual(len(result.structured_content["sha256"]), 64)

    async def test_status_tool_is_read_only_and_cannot_override_the_host(self):
        with patch("codex_agent.remote_maintenance.validation_status", return_value={
            "run_id": "run-plan", "execution_state": "unknown", "test_status": None,
            "remote_state": "queued", "observation": "live-remote",
        }) as inspect:
            async with Client(server) as client:
                tools = await client.list_tools()
                tool = next(item for item in tools.tools if item.name == "get_validation_status")
                result = await client.call_tool("get_validation_status", {"run_id": "run-plan", "remote": True})
        self.assertTrue(tool.annotations.read_only_hint)
        self.assertFalse(tool.annotations.destructive_hint)
        self.assertEqual(set(tool.input_schema["properties"]), {"run_id", "remote"})
        self.assertFalse(result.is_error)
        self.assertIsNone(result.structured_content["test_status"])
        self.assertEqual(inspect.call_args.kwargs, {"remote": True})

    async def test_mcp_validation_defaults_to_a_nonexecuting_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator_root = root / "python/examples/flaggems"
            operator_root.mkdir(parents=True)
            (operator_root / "demo.py").write_text(
                "import triton\n@triton.jit\ndef demo_kernel(x):\n    return x\n",
                encoding="utf-8",
            )
            (operator_root / "test_demo.py").write_text(
                "def test_demo():\n    assert True\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TRITON_RISCV_REPO_ROOT": str(root)}):
                async with Client(server) as client:
                    result = await client.call_tool(
                        "validate_operator",
                        {"operator_name": "demo"},
                    )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "planned")

    async def test_mcp_prepares_a_new_operator_without_writing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "python/examples/flaggems").mkdir(parents=True)
            with patch.dict(os.environ, {"TRITON_RISCV_REPO_ROOT": str(root)}):
                async with Client(server) as client:
                    result = await client.call_tool(
                        "prepare_operator_development",
                        {"specification": valid_spec()},
                    )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "planned")
        self.assertEqual(result.structured_content["operator"], "square_new")

    async def test_mcp_redirects_existing_targets_instead_of_returning_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator_root = root / "python/examples/flaggems"
            operator_root.mkdir(parents=True)
            (operator_root / "square_new.py").write_text(
                "import triton\n@triton.jit\ndef square_new_kernel(x):\n    return x\n",
                encoding="utf-8",
            )
            (operator_root / "test_square_new.py").write_text(
                "def test_square_new():\n    assert True\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TRITON_RISCV_REPO_ROOT": str(root)}):
                async with Client(server) as client:
                    result = await client.call_tool(
                        "prepare_operator_development",
                        {"specification": valid_spec()},
                    )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "existing_target")
        self.assertIn(
            "python/examples/flaggems/square_new.py",
            result.structured_content["existing_files"],
        )
        self.assertIn("discover_operator", result.structured_content["next_action"])

    async def test_stdio_demo_exposes_and_calls_the_same_tool(self) -> None:
        payload = await run_mcp_demo(FIXTURE_REPOSITORY, "relu_and_mul")

        self.assertEqual(payload["transport"], "stdio")
        self.assertIn("discover_operator", payload["available_tools"])
        self.assertEqual(payload["result"]["status"], "found")
        self.assertEqual(payload["result"]["operator"]["name"], "relu_and_mul")


if __name__ == "__main__":
    unittest.main()
