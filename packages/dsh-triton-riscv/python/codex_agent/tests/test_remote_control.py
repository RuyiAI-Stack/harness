import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from codex_agent.execution_guard import atomic_json, journal_path
from codex_agent.remote_control import request_remote_stop


class RemoteControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_id = "run-cancel-fixture"
        self.job = {"host": "host", "repository": "/repo", "job_id": "a" * 32,
                    "stage": "/tmp/triton-riscv-agent/" + "a" * 32, "request_digest": "b" * 64}
        self.journal = {"kind": "validation", "id": self.run_id, "state": "unknown",
                        "execution": {"remote_job": self.job}}
        self.path = journal_path(self.root, "validation", self.run_id)
        atomic_json(self.path, self.journal)
        self.env = patch.dict(os.environ, {"RISCV_HOST": "host", "RISCV_REPO": "/repo"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_confirmed_stop_preserves_original_journal_and_has_separate_evidence(self):
        terminal = {"state": "completed", "exit_code": 130, "cancellation": {"confirmed": True}}
        with patch("codex_agent.remote_control._job_rpc", side_effect=[{"state": "running"}, terminal]) as rpc:
            result = request_remote_stop(self.root, "validation", self.run_id)
        self.assertEqual([call.args[2] for call in rpc.call_args_list], ["cancel", "inspect"])
        self.assertTrue(result["remote_shutdown_confirmed"])
        self.assertEqual(json.loads(self.path.read_text()), self.journal)
        self.assertEqual(json.loads(self.path.with_suffix(".remote-cancel.json").read_text()), result)

    def test_lost_cancel_ack_is_unknown_and_never_retried(self):
        with patch("codex_agent.remote_control._job_rpc", side_effect=RuntimeError("offline")) as rpc:
            result = request_remote_stop(self.root, "validation", self.run_id)
        self.assertFalse(result["remote_shutdown_confirmed"])
        self.assertEqual(rpc.call_count, 1)

    def test_total_budget_timeout_is_saved_not_lost_as_a_bridge_exception(self):
        with patch("codex_agent.remote_control._job_rpc", side_effect=subprocess.TimeoutExpired("ssh", 10)):
            result = request_remote_stop(self.root, "validation", self.run_id)
        self.assertFalse(result["remote_shutdown_confirmed"])
        self.assertTrue(self.path.with_suffix(".remote-cancel.json").exists())

    def test_finished_test_wins_cancel_race_without_relabeling_it_cancelled(self):
        with patch("codex_agent.remote_control._job_rpc", return_value={"state": "completed", "exit_code": 0}):
            result = request_remote_stop(self.root, "validation", self.run_id)
        self.assertEqual(result["status"], "already_completed")
        self.assertFalse(result["remote_shutdown_confirmed"])

    def test_wrong_host_or_journal_identity_cannot_signal_another_task(self):
        with patch.dict(os.environ, {"RISCV_HOST": "wrong"}), patch("codex_agent.remote_control._job_rpc") as rpc:
            result = request_remote_stop(self.root, "validation", self.run_id)
            self.assertFalse(result["remote_shutdown_confirmed"])
            rpc.assert_not_called()
        self.journal["id"] = "wrong"
        atomic_json(self.path, self.journal)
        with self.assertRaisesRegex(PermissionError, "identity mismatch"):
            request_remote_stop(self.root, "validation", self.run_id)

    def test_completed_or_untracked_task_does_not_send_remote_cancel(self):
        for state in ("completed", "unknown"):
            with self.subTest(state=state):
                self.journal.update(state=state, execution={})
                atomic_json(self.path, self.journal)
                with patch("codex_agent.remote_control._job_rpc") as rpc:
                    result = request_remote_stop(self.root, "validation", self.run_id)
                    self.assertFalse(result["remote_shutdown_confirmed"])
                    rpc.assert_not_called()
