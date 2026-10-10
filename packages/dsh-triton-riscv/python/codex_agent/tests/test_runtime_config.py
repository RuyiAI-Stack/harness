"""Offline configuration contract tests; no model or SSH requests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from codex_agent.runtime_config import (
    CONFIG_ENV, approval_required, permission_enabled, remote_execution_identity,
    runtime_config,
)


def document(**settings):
    return json.dumps({"schemaVersion": 1, "repoRoot": "/tmp/operator repo",
                       "stateDir": "/tmp/operator-state", **settings})


class RuntimeConfigTests(unittest.TestCase):
    def test_native_defaults_are_read_only_and_preserve_rag(self):
        cfg = runtime_config({CONFIG_ENV: document()})
        self.assertEqual(cfg.permissions.model_dump(),
                         {"validation": False, "development": False, "repair": False})
        self.assertEqual(cfg.memory.retrievalMode, "legacy")
        self.assertEqual(cfg.memory.contextFormat, "classic")
        self.assertEqual(cfg.memory.embedding.provider, "none")
        self.assertFalse(cfg.remote.requireTaskQuotas)

    def test_native_settings_are_authoritative_for_all_consumers(self):
        from codex_agent.diagnostic_memory import memory_workspace
        from codex_agent.paths import repository_root, state_root
        from codex_agent.remote_executor import RemoteValidationConfig
        raw = document(permissions={"validation": True},
                       storage={"urlEnv": "WORKSPACE_DB_URL"})
        env = {CONFIG_ENV: raw, "TRITON_RISCV_REPO_ROOT": "/wrong",
               "TRITON_RISCV_STATE_DIR": "/wrong-state", "TRITON_RISCV_MEMORY_DB": "/wrong-db",
               "TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": "1", "TRITON_RISCV_ALLOW_REPAIR_APPLY": "1",
               "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "0",
               "RISCV_HOST": "wrong", "RISCV_REPO": "/wrong"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(repository_root(), Path("/tmp/operator repo").resolve())
            self.assertEqual(state_root(Path("/unused")), Path("/tmp/operator-state").resolve())
            self.assertEqual(memory_workspace(repository_root()), repository_root())
            self.assertEqual(runtime_config().storage.urlEnv, "WORKSPACE_DB_URL")
            self.assertIsNone(RemoteValidationConfig.from_env())
            self.assertTrue(permission_enabled("validation"))
            self.assertTrue(permission_enabled("job"))
            self.assertFalse(permission_enabled("development"))
            self.assertFalse(permission_enabled("repair"))
            self.assertFalse(permission_enabled("unknown"))
            self.assertTrue(approval_required())

    def test_remote_target_and_quota_policy_use_native_settings(self):
        from codex_agent.remote_executor import RemoteValidationConfig
        from codex_agent.resource_probe import quota_policy_error
        raw = document(remote={"host": "fixture", "repository": "/work/operators",
                               "required": True, "requireTaskQuotas": True})
        with patch.dict(os.environ, {CONFIG_ENV: raw, "TRITON_RISCV_REQUIRE_TASK_QUOTAS": "0"}, clear=True):
            target = RemoteValidationConfig.from_env()
            self.assertEqual((target.host, target.repository), ("fixture", "/work/operators"))
            self.assertTrue(runtime_config().remote.requireTaskQuotas)
            self.assertIsNotNone(quota_policy_error())

    def test_rejects_invalid_fields_types_paths_and_remote_pairs(self):
        bad = [document(schemaVersion=2), document(schemaVersion=True), document(repoRoot="relative"),
               document(stateDir=""), document(permissions={"repair": "1"}),
               document(apiKey="private-value"), document(memory={"unexpected": True}),
               document(memory={"database": "relative.db"}),
               document(memory={"embedding": {"tokenBudget": True}}),
               document(memory={"embedding": {"tokenBudget": 0}}),
               document(memory={"embedding": {"tokenizerJson": "relative.json"}}),
               document(remote={"host": "host"}), document(remote={"required": True}),
               document(remote={"requireTaskQuotas": True}),
               document(remote={"host": "host;sh", "repository": "/repo"}),
               document(remote={"host": "host", "repository": "/repo/../other"}),
               document(memory={"embedding": {"model": "bad\nmodel"}})]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "Invalid TRITON_RISCV_CONFIG"):
                runtime_config({CONFIG_ENV: raw})

    def test_rejects_duplicate_keys_nonfinite_values_and_oversized_documents(self):
        for raw in ('{"schemaVersion":1,"schemaVersion":1}', '{"schemaVersion":NaN}',
                    '[]', '', 'not-json', 'x' * 65537):
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                runtime_config({CONFIG_ENV: raw})

    def test_invalid_native_document_never_falls_back_to_legacy_permission(self):
        with patch.dict(os.environ, {CONFIG_ENV: '{"apiKey":"private-value"}',
                                    "TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}, clear=True):
            with self.assertRaises(ValueError) as error:
                permission_enabled("repair")
            self.assertNotIn("private-value", str(error.exception))

    def test_credential_variable_cannot_override_control_fields(self):
        for name in (CONFIG_ENV, "RISCV_HOST", "PATH", "HOME", "PYTHONPATH", "bad-name"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime_config({CONFIG_ENV: document(memory={"embedding": {"apiKeyEnv": name}})})

    def test_cached_settings_cannot_be_mutated_or_leak_between_sessions(self):
        first = runtime_config({CONFIG_ENV: document(permissions={"repair": True})})
        second = runtime_config({CONFIG_ENV: document(repoRoot="/tmp/other")})
        with self.assertRaises(ValueError):
            first.permissions.repair = False
        self.assertFalse(second.permissions.repair)
        self.assertEqual(first.repoRoot, "/tmp/operator repo")
        self.assertEqual(second.repoRoot, "/tmp/other")

    def test_legacy_cli_configuration_remains_supported(self):
        env = {"TRITON_RISCV_CHECKOUT": "/tmp/old", "TRITON_RISCV_ALLOW_REPAIR_APPLY": "1",
               "TRITON_RISCV_MEMORY_DB": "history.sqlite3", "TRITON_RISCV_EMBEDDING_TOKEN_BUDGET": "256",
               "RISCV_HOST": "old", "RISCV_REPO": "/work/old", "TRITON_RISCV_REQUIRE_REMOTE": "1"}
        cfg = runtime_config(env)
        self.assertEqual(cfg.repoRoot, "/tmp/old")
        self.assertTrue(cfg.permissions.repair)
        self.assertEqual(cfg.storage.urlEnv, "TRITON_MYSQL_URL")
        self.assertEqual(cfg.memory.embedding.tokenBudget, 256)
        self.assertEqual(cfg.remote.host, "old")
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(approval_required())

    def test_journal_remote_identity_is_stable_across_native_transport_change(self):
        for remote in ({"host": "", "repository": "", "required": False},
                       {"host": "fixture", "repository": "/repo", "required": True}):
            legacy = {"RISCV_HOST": remote["host"], "RISCV_REPO": remote["repository"],
                      "TRITON_RISCV_REQUIRE_REMOTE": "1" if remote["required"] else "0"}
            with patch.dict(os.environ, legacy, clear=True):
                before = remote_execution_identity()
            with patch.dict(os.environ, {CONFIG_ENV: document(remote=remote)}, clear=True):
                self.assertEqual(before, remote_execution_identity())

    def test_test_children_receive_neither_host_authority_nor_custom_secret(self):
        from codex_agent.process_control import validation_environment
        raw = document(permissions={"repair": True}, memory={"embedding": {"apiKeyEnv": "MY_SECRET"}})
        env = {CONFIG_ENV: raw, "MY_SECRET": "private-value", "OPENAI_API_KEY": "another-value",
               "TRITON_RISCV_ALLOW_VALIDATION": "1", "PATH": "/bin", "OMP_NUM_THREADS": "2"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(validation_environment(), {"PATH": "/bin", "OMP_NUM_THREADS": "2"})

    def test_native_embedding_settings_do_not_inherit_ambient_defaults(self):
        from codex_agent.diagnostic_memory import build_configured_embedding_provider
        raw = document(memory={"embedding": {"provider": "ollama"}})
        with patch.dict(os.environ, {CONFIG_ENV: raw, "AGENT_EMBEDDING_MODEL": "stale-model"}, clear=True):
            with self.assertRaisesRegex(ValueError, "require a model"):
                build_configured_embedding_provider()
        with patch.dict(os.environ, {"TRITON_RISCV_EMBEDDING_PROVIDER": "ollama",
                                    "AGENT_EMBEDDING_MODEL": "legacy-model"}, clear=True), patch(
                "codex_agent.embeddings.OllamaEmbeddingProvider") as provider:
            build_configured_embedding_provider()
            self.assertEqual(provider.call_args.kwargs["model"], "legacy-model")

    def test_native_embedding_credentials_are_read_separately_without_network(self):
        from codex_agent.diagnostic_memory import build_configured_embedding_provider
        raw = document(memory={"embedding": {"provider": "openai-compatible", "model": "fixture",
                       "baseUrl": "https://example.invalid/v1", "apiKeyEnv": "MY_SECRET"}})
        with patch.dict(os.environ, {CONFIG_ENV: raw, "MY_SECRET": "private-value",
                                    "AGENT_EMBEDDING_TOKEN_BUDGET": "999"}, clear=True), patch(
                "codex_agent.embeddings.OpenAICompatibleEmbeddingProvider") as provider:
            build_configured_embedding_provider()
            self.assertEqual(provider.call_args.kwargs["api_key"], "private-value")
            self.assertEqual(provider.call_args.kwargs["model"], "fixture")
            self.assertIsNone(provider.call_args.kwargs["token_budget"])
            self.assertNotIn("private-value", raw)

    def test_invalid_config_blocks_mcp_startup_without_echoing_payload(self):
        result = subprocess.run([sys.executable, "-I", "-m", "codex_agent.harness.mcp_server"],
                                env={**os.environ, CONFIG_ENV: '{"apiKey":"private-value"}'},
                                input="", capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid TRITON_RISCV_CONFIG", result.stderr)
        self.assertNotIn("private-value", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
