from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.core import (
    WorkflowPhase,
    WorkflowSession,
    WorkflowTransitionError,
)


class WorkflowTests(unittest.TestCase):
    def test_persists_state_and_append_only_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            session = WorkflowSession(run_dir, "run-1", "square_and_mul")
            session.transition(
                WorkflowPhase.CONTEXT_READY,
                "operator context prepared",
                data={"references": 2},
                memory_ids=[3, 5],
            )
            session.transition(
                WorkflowPhase.ENVIRONMENT_READY,
                "preflight passed",
            )

            state = json.loads((run_dir / "workflow-state.json").read_text())
            events = [
                json.loads(line)
                for line in (run_dir / "workflow-events.jsonl").read_text().splitlines()
            ]

            self.assertEqual(state["phase"], "environment-ready")
            self.assertEqual(state["sequence"], 2)
            self.assertEqual(state["memory_ids"], [3, 5])
            self.assertEqual([event["sequence"] for event in events], [0, 1, 2])
            self.assertEqual(events[1]["data"], {"references": 2})

    def test_rejects_invalid_and_terminal_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = WorkflowSession(Path(temp_dir), "run-2", "relu")
            with self.assertRaises(WorkflowTransitionError):
                session.transition(WorkflowPhase.VALIDATING, "skip required phases")

            session.transition(WorkflowPhase.FAILED, "unrecoverable setup error")
            with self.assertRaises(WorkflowTransitionError):
                session.transition(WorkflowPhase.CONTEXT_READY, "cannot resume terminal")


if __name__ == "__main__":
    unittest.main()
