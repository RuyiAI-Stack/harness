from __future__ import annotations

import tempfile
import time
import unittest
import json
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi.testclient import TestClient

from codex_agent.harness import HarnessAgent, HarnessRunResult, HarnessSettings
from codex_agent.operator_development import (
    prepare_operator_development,
    propose_operator_implementation,
)
from codex_agent.operator_lifecycle import (
    propose_operator_repair,
    validate_operator_target,
)
from codex_agent.platform.api import create_app
from codex_agent.tests.test_operator_development import (
    IMPLEMENTATION,
    TEST_SOURCE,
    valid_spec,
)


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
                    "payload": {"event": {"type": "tool/result"}},
                }
            )
        return HarnessRunResult(
            session_id=session_id or "generated-session",
            final_response="HTTP bridge reached the Harness backend.",
            finish_reason="completed",
        )


class PlatformApiTests(unittest.TestCase):
    def test_session_controls_and_cross_run_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = HarnessSettings.from_env(root, {})
            app = create_app(root, harness_agent=HarnessAgent(settings, FakeHarnessBackend()))

            with TestClient(app) as client:
                first = client.post(
                    "/api/sessions", json={"title": "First session"}
                ).json()
                second = client.post(
                    "/api/sessions", json={"title": "Second session"}
                ).json()
                pinned = client.patch(
                    f"/api/sessions/{first['id']}", json={"pinned": True}
                )
                self.assertEqual(pinned.status_code, 200)
                self.assertEqual(pinned.json()["pinned"], 1)
                self.assertEqual(client.get("/api/sessions").json()[0]["id"], first["id"])

                run = app.state.store.create_run(
                    first["id"], "develop-operator", "demo", {}
                )
                app.state.store.add_event(
                    run["id"],
                    "approval-required",
                    {
                        "proposal_id": "development-proposal-demo",
                        "proposal_type": "development",
                    },
                )
                approvals = client.get(
                    f"/api/sessions/{first['id']}/approvals"
                ).json()
                self.assertEqual(len(approvals), 1)
                self.assertEqual(
                    approvals[0]["payload"]["proposal_id"],
                    "development-proposal-demo",
                )

                deleted = client.delete(f"/api/sessions/{second['id']}")
                self.assertEqual(deleted.status_code, 204)
                self.assertEqual(
                    client.get(f"/api/sessions/{second['id']}").status_code,
                    404,
                )

    def test_message_auto_starts_and_harness_result_crosses_http_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = HarnessSettings.from_env(
                root,
                {"ISRC_API_KEY": "test-key", "DSH_MODEL": "test-model"},
            )
            agent = HarnessAgent(settings, FakeHarnessBackend())
            app = create_app(root, harness_agent=agent)

            with TestClient(app) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                session = client.post("/api/sessions", json={"title": "HTTP test"}).json()
                planned = client.post(
                    f"/api/sessions/{session['id']}/messages",
                    json={"content": "分析一个没有关键词模板的新任务"},
                ).json()
                self.assertNotEqual(planned["run"]["status"], "awaiting-confirmation")
                self.assertIsNone(planned["assistant_message"])

                run_id = planned["run"]["id"]
                deadline = time.time() + 2
                while time.time() < deadline:
                    run = client.get(f"/api/runs/{run_id}").json()
                    if run["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertEqual(run["status"], "completed")
                events = client.get(f"/api/runs/{run_id}/events").json()
                self.assertIn("harness-event", [event["event_type"] for event in events])
                bundle = client.get(f"/api/sessions/{session['id']}").json()
                self.assertIn("HTTP bridge reached", bundle["messages"][-1]["content"])

    def test_validation_command_decision_is_a_separate_http_boundary(self) -> None:
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
            plan = validate_operator_target(root, "demo")
            settings = HarnessSettings.from_env(root, {})
            app = create_app(root, harness_agent=HarnessAgent(settings, FakeHarnessBackend()))

            with TestClient(app) as client:
                fetched = client.get(f"/api/validation-plans/{plan.run_id}")
                self.assertEqual(fetched.status_code, 200)
                self.assertEqual(
                    fetched.json()["approval"]["status"],
                    "pending_approval",
                )
                decided = client.post(
                    f"/api/validation-plans/{plan.run_id}/decision",
                    json={"approve": True, "reviewer": "api-test"},
                )

        self.assertEqual(decided.status_code, 200)
        self.assertEqual(decided.json()["approval"]["status"], "approved")

    def test_repair_decision_is_a_separate_http_boundary(self) -> None:
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
            planned = validate_operator_target(root, "demo")
            receipt_path = root / planned.receipt_path
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.update({"status": "failed", "failure_stage": "correctness"})
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            proposal = propose_operator_repair(
                root,
                planned.run_id,
                "import triton\n@triton.jit\ndef demo_kernel(x):\n    return x + 1\n",
                "Fix the implementation.",
            )
            settings = HarnessSettings.from_env(root, {})
            app = create_app(root, harness_agent=HarnessAgent(settings, FakeHarnessBackend()))

            with TestClient(app) as client:
                fetched = client.get(
                    f"/api/repair-proposals/{proposal.proposal_id}"
                )
                self.assertEqual(fetched.status_code, 200)
                self.assertNotIn("replacement_source", fetched.json())
                decided = client.post(
                    f"/api/repair-proposals/{proposal.proposal_id}/decision",
                    json={"approve": True, "reviewer": "api-test"},
                )

        self.assertEqual(decided.status_code, 200)
        self.assertEqual(decided.json()["status"], "approved")

    def test_development_decision_is_a_separate_http_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "python/examples/flaggems").mkdir(parents=True)
            plan = prepare_operator_development(root, valid_spec())
            proposal = propose_operator_implementation(
                root,
                plan.development_id,
                IMPLEMENTATION,
                TEST_SOURCE,
                "Create a reviewed new operator.",
            )
            settings = HarnessSettings.from_env(root, {})
            app = create_app(root, harness_agent=HarnessAgent(settings, FakeHarnessBackend()))

            with TestClient(app) as client:
                fetched = client.get(
                    f"/api/development-proposals/{proposal.proposal_id}"
                )
                self.assertEqual(fetched.status_code, 200)
                self.assertNotIn("implementation_source", fetched.json())
                decided = client.post(
                    f"/api/development-proposals/{proposal.proposal_id}/decision",
                    json={"approve": True, "reviewer": "api-test"},
                )

        self.assertEqual(decided.status_code, 200)
        self.assertEqual(decided.json()["status"], "approved")


if __name__ == "__main__":
    unittest.main()
