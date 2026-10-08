import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

from codex_agent import remote_process as control


class RemoteProcessTests(unittest.TestCase):
    def test_pidfd_is_rechecked_before_signalling(self):
        for children, expected in (([42], 1), ([], 0)):
            with self.subTest(children=children), patch.object(control.os, "pidfd_open", return_value=7, create=True), \
                 patch.object(control, "child_ids", return_value=children), \
                 patch.object(control.signal, "pidfd_send_signal", create=True) as send, \
                 patch.object(control.os, "close") as close:
                control.signal_owned_child(42, signal.SIGTERM)
                self.assertEqual(send.call_count, expected)
                close.assert_called_once_with(7)

    def test_unconfirmed_drain_cannot_return_success(self):
        with patch.object(control, "child_ids", return_value=[42]), \
             patch.object(control, "signal_owned_child"), \
             patch.object(control.os, "waitpid", return_value=(0, 0)), \
             patch.object(control.time, "monotonic", side_effect=[0, 4]):
            with self.assertRaisesRegex(RuntimeError, "shutdown not confirmed"):
                control.drain_children()

    def test_natural_completion_also_reaps_descendants(self):
        process = Mock(pid=42)
        process.wait.return_value = 0
        with patch.object(control, "enable_subreaper", return_value=True), \
             patch.object(control.subprocess, "Popen", return_value=process), \
             patch.object(control, "drain_children") as drain:
            result = control.run_controlled(["fixture"], stdin=None, stdout=None, cancelled=lambda: False)
        drain.assert_called_once_with()
        self.assertFalse(result["cancel_applied"])
        self.assertTrue(result["descendants_stopped"])

    def test_unsupported_platform_does_not_claim_shutdown(self):
        process = Mock(pid=42)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", .1), 0]
        with patch.object(control, "enable_subreaper", return_value=False), \
             patch.object(control.subprocess, "Popen", return_value=process), \
             patch.object(control, "signal_owned_child") as send:
            result = control.run_controlled(["fixture"], stdin=None, stdout=None, cancelled=lambda: True)
        send.assert_not_called()
        self.assertFalse(result["cancel_applied"])
        self.assertFalse(result["descendants_stopped"])

    def test_cancellation_requires_drain_before_confirmation(self):
        process = Mock(pid=42)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", .1), -15]
        with patch.object(control, "enable_subreaper", return_value=True), \
             patch.object(control.subprocess, "Popen", return_value=process), \
             patch.object(control, "signal_owned_child") as send, \
             patch.object(control, "drain_children", side_effect=RuntimeError("cannot prove shutdown")):
            with self.assertRaisesRegex(RuntimeError, "cannot prove"):
                control.run_controlled(["fixture"], stdin=None, stdout=None, cancelled=lambda: True)
        send.assert_called_once_with(42, signal.SIGTERM)
