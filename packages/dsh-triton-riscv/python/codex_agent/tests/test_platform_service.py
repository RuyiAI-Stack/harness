from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Optional

from codex_agent.harness import HarnessAgent, HarnessRunResult, HarnessSettings
from codex_agent.platform.executor import HarnessRunExecutor
from codex_agent.platform.service import PlatformService
from codex_agent.platform.store import PlatformStore


class FakeHarnessBackend:
    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> HarnessRunResult:
        if on_event is not None:
            on_event(
                {
                    "method": "session.event",
                    "payload": {"event": {"type": "assistant/message"}},
                }
            )
        return HarnessRunResult(
            session_id=session_id or "generated-session",
            final_response="Harness inspected the requested operator.",
            finish_reason="completed",
            events=[{"type": "assistant/message"}],
        )


class ErrorHarnessBackend:
    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> HarnessRunResult:
        if on_event is not None:
            on_event(
                {
                    "method": "session.event",
                    "payload": {
                        "event": {
                            "data": {
                                "reason": {
                                    "error": {
                                        "code": "UNKNOWN",
                                        "message": "persisted session id collision",
                                    },
                                    "kind": "error",
                                }
                            },
                            "type": "turn/end",
                        }
                    },
                }
            )
        return HarnessRunResult(
            session_id=session_id or "generated-session",
            final_response="",
            finish_reason="error",
        )


class PlatformServiceTests(unittest.TestCase):
    def test_executor_extracts_human_approval_events(self) -> None:
        notification = {
            "payload": {
                "event": {
                    "type": "tool/result",
                    "data": {
                        "result": {
                            "proposal_id": "development-proposal-20260831-120000-abcdef12"
                        }
                    },
                }
            }
        }

        self.assertEqual(
            HarnessRunExecutor._proposal_references(notification),
            [
                {
                    "proposal_id": "development-proposal-20260831-120000-abcdef12",
                    "proposal_type": "development",
                }
            ],
        )
        self.assertEqual(
            HarnessRunExecutor._proposal_references(
                {
                    "payload": {
                        "event": {
                            "type": "user/message",
                            "data": {
                                "text": "继续 proposal development-proposal-20260831-120000-abcdef12"
                            },
                        }
                    }
                }
            ),
            [],
        )

    def test_executor_extracts_validation_plan_from_tool_json(self) -> None:
        notification = {
            "method": "session.event",
            "payload": {
                "tool_result": json.dumps(
                    {
                        "run_id": "run-20260901-120000-abcdef12",
                        "operator": "scaled_sigmoid",
                        "status": "planned",
                        "command": "ssh sg2044 -- python -m pytest test_scaled_sigmoid.py",
                    }
                )
            },
        }

        self.assertEqual(
            HarnessRunExecutor._validation_plan_references(notification),
            [
                {
                    "validation_run_id": "run-20260901-120000-abcdef12",
                    "operator": "scaled_sigmoid",
                    "command": "ssh sg2044 -- python -m pytest test_scaled_sigmoid.py",
                }
            ],
        )

    def test_executor_exposes_text_deltas_and_public_agent_steps(self) -> None:
        delta = HarnessRunExecutor._public_events(
            {
                "payload": {
                    "event": {
                        "type": "assistant/chunk",
                        "data": {
                            "step": 1,
                            "chunk": {"type": "text-delta", "text": "验"},
                        },
                    }
                }
            }
        )
        self.assertEqual(
            delta,
            [("assistant-delta", {"text": "验", "step": 1})],
        )

        commentary = HarnessRunExecutor._public_events(
            {
                "payload": {
                    "event": {
                        "type": "assistant/message",
                        "data": {
                            "step": 1,
                            "message": {
                                "content": [{"type": "text", "text": "准备执行验证。"}],
                                "source": {
                                    "replayState": {
                                        "blocks": [
                                            {
                                                "textSignature": json.dumps(
                                                    {"phase": "commentary"}
                                                )
                                            }
                                        ]
                                    }
                                },
                            },
                        },
                    }
                }
            }
        )
        self.assertEqual(commentary[0][0], "agent-step")
        self.assertEqual(commentary[0][1]["kind"], "model-progress")
        self.assertEqual(commentary[0][1]["message"], "准备执行验证。")

        tool_call = HarnessRunExecutor._public_events(
            {
                "payload": {
                    "event": {
                        "type": "tool/call",
                        "data": {
                            "step": 1,
                            "name": "mcp__triton_riscv__validate_operator",
                            "arguments": json.dumps(
                                {
                                    "operator_name": "scaled_sigmoid",
                                    "approved_run_id": "run-demo",
                                }
                            ),
                        },
                    }
                }
            }
        )
        self.assertEqual(tool_call[0][1]["kind"], "tool-call")
        self.assertIn("scaled_sigmoid", tool_call[0][1]["detail"])

        tool_result = HarnessRunExecutor._public_events(
            {
                "payload": {
                    "event": {
                        "type": "tool/result",
                        "data": {
                            "message": {
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "run_id": "run-result",
                                                "operator": "scaled_sigmoid",
                                                "status": "passed",
                                            }
                                        )
                                    }
                                ]
                            }
                        },
                    }
                }
            }
        )
        self.assertEqual(tool_result[0][1]["kind"], "tool-result")
        self.assertEqual(tool_result[0][1]["status"], "passed")

    def make_components(
        self, root: Path
    ) -> tuple[PlatformStore, PlatformService, HarnessRunExecutor]:
        results = root / "agent-results"
        results.mkdir()
        (results / "operators.json").write_text(
            json.dumps({"summary": {"operators": 1}, "operators": []}),
            encoding="utf-8",
        )
        (results / "project-targets.json").write_text(
            json.dumps({"summary": {"total_targets": 12}}),
            encoding="utf-8",
        )
        settings = HarnessSettings.from_env(
            root,
            {
                "ISRC_API_KEY": "test-key",
                "DSH_MODEL": "test-model",
            },
        )
        agent = HarnessAgent(settings, FakeHarnessBackend())
        store = PlatformStore(results / "platform.sqlite3")
        service = PlatformService(root, store, agent)
        executor = HarnessRunExecutor(root, store, agent, workers=1)
        return store, service, executor

    def test_every_message_prepares_a_harness_run_without_model_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            session = service.create_session()

            result = service.handle_message(session["id"], "验证 relu_and_mul")

            self.assertEqual(result["run"]["status"], "awaiting-confirmation")
            self.assertEqual(result["run"]["request"]["action"], "deepseek-harness")
            self.assertFalse(result["run"]["request"]["requires_confirmation"])
            self.assertEqual(result["task"]["runtime"], "deepseek-harness")
            self.assertIsNone(result["assistant_message"])

    def test_schedule_runs_harness_without_an_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            session = service.create_session()
            planned = service.handle_message(session["id"], "分析 gelu 失败")

            executor.schedule(planned["run"]["id"])
            deadline = time.time() + 2
            while time.time() < deadline:
                run = store.get_run(planned["run"]["id"])
                if run["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)

            self.assertEqual(run["status"], "completed")
            event_types = [item["event_type"] for item in store.list_events(run["id"])]
            self.assertIn("scheduled", event_types)
            self.assertNotIn("confirmed", event_types)

    def test_confirm_runs_harness_and_persists_stream_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            session = service.create_session()
            planned = service.handle_message(session["id"], "分析 gelu 失败")

            executor.confirm(planned["run"]["id"])
            deadline = time.time() + 2
            while time.time() < deadline:
                run = store.get_run(planned["run"]["id"])
                if run["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)

            self.assertEqual(run["status"], "completed")
            expected_session_id = planned["run"]["request"]["harness_session_id"]
            self.assertEqual(run["result"]["harness_session_id"], expected_session_id)
            self.assertIn(session["id"], expected_session_id)
            event_types = [item["event_type"] for item in store.list_events(run["id"])]
            self.assertIn("harness-event", event_types)
            self.assertIn("completed", event_types)
            self.assertIn(
                "Harness inspected",
                store.list_messages(session["id"])[-1]["content"],
            )

    def test_harness_error_never_persists_an_empty_assistant_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results = root / "agent-results"
            results.mkdir()
            settings = HarnessSettings.from_env(root, {"ISRC_API_KEY": "test-key"})
            agent = HarnessAgent(settings, ErrorHarnessBackend())
            store = PlatformStore(results / "platform.sqlite3")
            service = PlatformService(root, store, agent)
            executor = HarnessRunExecutor(root, store, agent, workers=1)
            self.addCleanup(executor.close)
            session = service.create_session()
            planned = service.handle_message(session["id"], "北京今天的天气")

            executor.schedule(planned["run"]["id"])
            deadline = time.time() + 2
            while time.time() < deadline:
                run = store.get_run(planned["run"]["id"])
                if run["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)

            self.assertEqual(run["status"], "failed")
            message = store.list_messages(session["id"])[-1]["content"]
            self.assertIn("persisted session id collision", message)
            self.assertTrue(message.strip())

    def test_executor_rejects_old_python_command_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a DeepSeek Harness task"):
            HarnessRunExecutor._validate_request(
                {"action": "validate-operator", "argv": ["-m", "os"]}
            )

    def test_bootstrap_reports_harness_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, service, executor = self.make_components(Path(temp_dir))
            self.addCleanup(executor.close)
            bootstrap = service.bootstrap()

            self.assertEqual(bootstrap["harness"]["runtime"], "DeepSeek Harness")
            self.assertEqual(
                bootstrap["harness"]["plugin"]["name"],
                "dsh-triton-riscv",
            )
            self.assertTrue(bootstrap["harness"]["plugin"]["loaded"])
            self.assertEqual(bootstrap["harness"]["model"], "test-model")
            self.assertTrue(bootstrap["harness"]["api_configured"])


if __name__ == "__main__":
    unittest.main()
