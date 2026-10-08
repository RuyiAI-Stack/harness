"""Migration contracts. Fixtures are synthetic, not RISC-V performance evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from mcp import Client

import codex_agent
from codex_agent.diagnostic_memory import memory_database_path, remember_validation
from codex_agent.harness import HarnessAgent, HarnessSettings
from codex_agent.harness.mcp_server import server
from codex_agent.platform.api import create_app
from codex_agent.platform.executor import HarnessRunExecutor
from codex_agent.platform.store import PlatformStore
from codex_agent.tests.test_diagnostic_memory import failed_receipt, OPERATOR
from codex_agent.tests.test_platform_api import FakeHarnessBackend


class InstalledMigrationTests(unittest.TestCase):
    def test_installed_package_cannot_be_shadowed_by_target_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shadow = root / "codex_agent"
            shadow.mkdir()
            (shadow / "__init__.py").write_text("raise RuntimeError('old checkout imported')")
            completed = subprocess.run(
                [sys.executable, "-I", "-c", "import codex_agent; print(codex_agent.__file__)"],
                cwd=root, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(Path(completed.stdout.strip()).resolve(), Path(codex_agent.__file__).resolve())

    def test_ui_and_state_do_not_require_backend_inside_target_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "target"
            state = Path(temporary) / "state"
            root.mkdir()
            with patch.dict(os.environ, {"TRITON_RISCV_STATE_DIR": str(state)}):
                settings = HarnessSettings.from_env(root, {"TRITON_RISCV_STATE_DIR": str(state)})
                self.assertEqual(settings.session_root, (state / "deepseek-harness").resolve())
                app = create_app(root, HarnessAgent(settings, FakeHarnessBackend()))
                with TestClient(app) as client:
                    response = client.get("/")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("/ui-assets/assets/", response.text)
                    self.assertEqual(client.get("/api/health").json()["repo_root"], str(root.resolve()))
                    session = client.post("/api/sessions", json={"title": "migration"}).json()
                    self.assertEqual(client.delete(f"/api/sessions/{session['id']}").status_code, 204)
            self.assertTrue((state / "platform.sqlite3").is_file())
            self.assertFalse((root / "codex_agent").exists())
            self.assertFalse((root / "agent-results/platform.sqlite3").exists())

    def test_managed_context_reaches_executor_and_persists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = PlatformStore(root / "state.sqlite3")
            settings = HarnessSettings.from_env(root, {"TRITON_RISCV_CONTEXT_SUMMARY_MODE": "extractive"})
            executor = HarnessRunExecutor(root, store, HarnessAgent(settings, FakeHarnessBackend()))
            try:
                session = store.create_session("context")
                store.save_context_checkpoint(session["id"], summary="Keep acceptance tests unchanged.",
                    through_message_id=None, source_ids=[], metrics={})
                run = store.create_run(session["id"], "develop-operator", "demo", {"task": "Continue repair"})
                prompt = executor._managed_task(run)
                self.assertIn("Keep acceptance tests unchanged", prompt)
                self.assertIn("Continue repair", prompt)
                self.assertIn("context-prepared", [e["event_type"] for e in store.list_events(run["id"])])
            finally:
                executor.close()


class MemoryMcpMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_retrieves_existing_evidence_without_inventing_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"TRITON_RISCV_REPO_ROOT": str(root),
                    "TRITON_RISCV_MEMORY_DB": str(root / "history.sqlite3"),
                    "TRITON_RISCV_EMBEDDING_PROVIDER": "none",
                    "TRITON_RISCV_MEMORY_RETRIEVAL_MODE": "legacy"}):
                remember_validation(root, failed_receipt("previous"), OPERATOR)
                async with Client(server) as client:
                    result = await client.call_tool("retrieve_operator_memory", {
                        "operator_name": "demo", "failure_stage": "mlir-translate",
                        "error_text": "Dialect linalg not found for linalg.generic"})
                self.assertFalse(result.is_error)
                payload = result.structured_content
                self.assertEqual(payload["status"], "found")
                item = payload["items"][0]
                self.assertIn("linalg", json.dumps(item["evidence"]))
                self.assertEqual(item["recommended_actions"], ["Preserve the failing IR."])
                self.assertIsNone(item["applied_action"])
                self.assertEqual(item["source_run"], "operator-lifecycle:previous")
                self.assertEqual(Path(payload["database"]), memory_database_path(root))
