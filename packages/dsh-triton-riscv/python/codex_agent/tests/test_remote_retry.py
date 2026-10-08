import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.remote_retry import observe_with_retry, TransportUnavailable
from codex_agent.process_control import execution_budget, ExecutionCancelled


class RemoteRetryTests(unittest.TestCase):
    def test_transient_read_error_retries_same_operation_only(self):
        from unittest.mock import Mock
        call = Mock(side_effect=[TransportUnavailable("offline"), "original-result"])
        with patch("codex_agent.remote_retry.RETRY_DELAYS", (0, 0)):
            events = []
            self.assertEqual(observe_with_retry("inspect", call, on_retry=lambda *args: events.append(args)), "original-result")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(events, [(1, "TransportUnavailable")])

    def test_read_timeouts_have_three_attempts_not_an_unbounded_loop(self):
        from unittest.mock import Mock
        call = Mock(side_effect=subprocess.TimeoutExpired("ssh", 25))
        with patch("codex_agent.remote_retry.RETRY_DELAYS", (0, 0)):
            with self.assertRaises(subprocess.TimeoutExpired):
                observe_with_retry("collect", call)
        self.assertEqual(call.call_count, 3)

    def test_mutating_actions_are_never_automatically_retried(self):
        from unittest.mock import Mock
        for action in ("start", "ack", "cancel", "invented"):
            with self.subTest(action=action):
                call = Mock(side_effect=TransportUnavailable("lost acknowledgment"))
                with self.assertRaises(TransportUnavailable):
                    observe_with_retry(action, call)
                self.assertEqual(call.call_count, 1)

    def test_bad_evidence_and_missing_executable_are_not_network_retries(self):
        from unittest.mock import Mock
        for error in (ValueError("wrong task ID"), FileNotFoundError("ssh"), RuntimeError("corrupt log")):
            with self.subTest(error=type(error).__name__):
                call = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    observe_with_retry("inspect", call)
                self.assertEqual(call.call_count, 1)

    def test_cancel_during_backoff_prevents_the_second_call(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            cancel = Path(folder) / "cancel"
            call = Mock(side_effect=TransportUnavailable("offline"))
            with execution_budget(5, cancel), self.assertRaises(ExecutionCancelled):
                observe_with_retry("inspect", call, on_retry=lambda *args: cancel.touch())
            self.assertEqual(call.call_count, 1)

    def test_wall_clock_budget_also_bounds_backoff(self):
        from unittest.mock import Mock
        call = Mock(side_effect=TransportUnavailable("offline"))
        with execution_budget(0.02), self.assertRaises(subprocess.TimeoutExpired):
            observe_with_retry("inspect", call)
        self.assertEqual(call.call_count, 1)
