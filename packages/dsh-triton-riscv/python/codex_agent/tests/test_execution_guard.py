"""Local fault injection, not evidence of real compiler/remote success."""
import json
import base64
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from codex_agent.execution_guard import atomic_json, inspect_execution, journal_path, resource_locks
from codex_agent.operator_lifecycle import validate_operator_target, decide_validation_plan
from codex_agent.process_control import ExecutionCancelled, execution_budget, run_bounded, validation_environment
from codex_agent.tests.test_operator_lifecycle import ORIGINAL_SOURCE, TEST_SOURCE
from codex_agent.validate_operator import OperatorValidationResult


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "python/examples/flaggems"
        folder.mkdir(parents=True)
        self.source = folder / "demo.py"
        self.source.write_text(ORIGINAL_SOURCE)
        (folder / "test_demo.py").write_text(TEST_SOURCE)
        self.environment = patch.dict(os.environ, {
            "TRITON_RISCV_ALLOW_VALIDATION": "1", "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "1",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def plan(self):
        plan = validate_operator_target(self.root, "demo")
        decide_validation_plan(self.root, plan.run_id, approve=True, reviewer="test")
        return plan

    def execute(self, plan, **kwargs):
        return validate_operator_target(self.root, "demo", execute=True, approved_run_id=plan.run_id, **kwargs)

    def runner(self, plan):
        return OperatorValidationResult(operator="demo", implementation_file="python/examples/flaggems/demo.py",
            test_files=["python/examples/flaggems/test_demo.py"], command=plan.command,
            dry_run=False, exit_code=0, status="passed", failure_stage=None, likely_reason=None,
            error_excerpt=[], duration_seconds=0.01, log_path=None)

    def test_completed_request_replays_from_disk_without_a_second_execution(self):
        plan = self.plan()
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)) as run:
            first = self.execute(plan)
            second = self.execute(plan)
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(run.call_count, 2)  # one dry plan check, one execution
        journal = inspect_execution(self.root, "validation", plan.run_id)
        self.assertEqual(journal["state"], "completed")
        self.assertTrue(journal["effects_started"])

    def test_changed_sandbox_launcher_invalidates_remote_replay(self):
        from codex_agent.remote_executor import RemotePreflightResult
        with patch.dict(os.environ, {"RISCV_HOST": "fixture", "RISCV_REPO": "/repo"}), patch(
            "codex_agent.linux_sandbox.launcher_digest", return_value="a" * 64
        ), patch("codex_agent.remote_executor.launcher_digest", return_value="a" * 64):
            plan = self.plan()
            with patch("codex_agent.operator_lifecycle.run_remote_operator", return_value=(
                self.runner(plan), RemotePreflightResult(configured=True, status="passed", architecture="riscv64")
            )) as remote:
                self.execute(plan)
                with patch("codex_agent.linux_sandbox.launcher_digest", return_value="b" * 64):
                    with self.assertRaisesRegex(PermissionError, "different arguments or environment"):
                        self.execute(plan)
                remote.assert_called_once()

    def test_approval_content_tampering_is_blocked_before_execution(self):
        plan = self.plan()
        path = self.root / plan.receipt_path
        value = json.loads(path.read_text())
        value["command"] = "echo tampered"
        atomic_json(path, value)
        with patch("codex_agent.operator_lifecycle.run_operator") as run:
            with self.assertRaisesRegex(PermissionError, "Approved content changed"):
                self.execute(plan)
            run.assert_not_called()

    def test_completed_failure_is_replayed_as_failure_not_success(self):
        plan = self.plan()
        failed = self.runner(plan)
        failed.status, failed.exit_code = "failed", 1
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=failed):
            self.assertEqual(self.execute(plan).status, "failed")
            self.assertEqual(self.execute(plan).status, "failed")
        journal = inspect_execution(self.root, "validation", plan.run_id)
        self.assertEqual(journal["state"], "completed")
        self.assertEqual(journal["result"]["status"], "failed")

    def test_remote_recovery_reads_original_job_and_never_uploads_or_starts_again(self):
        from codex_agent.remote_executor import RemotePreflightResult
        for code in (0, 1):
            with self.subTest(exit_code=code), patch.dict(os.environ, {"RISCV_HOST": "fixture", "RISCV_REPO": "/repo"}):
                plan = self.plan()
                actions = []
                broken = True
                raw = ("1 passed in 0.1s\n" if code == 0 else "1 failed in 0.1s\n").encode()
                def rpc(config, job, action, *args):
                    nonlocal broken
                    actions.append((action, job["job_id"]))
                    if action == "start" and broken:
                        broken = False
                        raise RuntimeError("Remote outcome unknown; simulated lost start reply")
                    return {"state": "completed", "exit_code": code, "duration_seconds": 0.1,
                            "log_bytes": len(raw), "log_sha256": hashlib.sha256(raw).hexdigest(),
                            "log_base64": base64.b64encode(raw).decode()}
                with patch("codex_agent.remote_executor.check_remote_environment", return_value=
                           RemotePreflightResult(configured=True, status="passed", architecture="riscv64")) as preflight, \
                     patch("codex_agent.remote_executor._run_ssh_script", return_value=subprocess.CompletedProcess([], 0, "")) as ssh, \
                     patch("codex_agent.remote_executor.run_bounded", return_value=subprocess.CompletedProcess([], 0, "")) as upload, \
                     patch("codex_agent.remote_executor._job_rpc", side_effect=rpc):
                    with self.assertRaisesRegex(RuntimeError, "unknown"):
                        self.execute(plan)
                    before = inspect_execution(self.root, "validation", plan.run_id)
                    self.assertEqual(before["state"], "unknown")
                    transferred = upload.call_count
                    second = self.execute(plan)
                    replay = self.execute(plan)
                    self.assertEqual(second.model_dump(), replay.model_dump())
                    self.assertEqual(second.run_id, before["execution"]["validation_result_run_id"])
                    self.assertEqual(second.exit_code, code)
                    self.assertEqual(second.status, "passed" if code == 0 else "failed")
                    self.assertEqual(upload.call_count, transferred)
                    preflight.assert_called_once()
                    ssh.assert_called_once()
                    self.assertEqual([a for a, _ in actions].count("start"), 1)
                    self.assertEqual(len({identifier for _, identifier in actions}), 1)
                    self.assertEqual(inspect_execution(self.root, "validation", plan.run_id)["state"], "completed")

    def test_changed_source_blocks_recovery_before_remote_query(self):
        from codex_agent.execution_guard import record_execution_detail
        with patch.dict(os.environ, {"RISCV_HOST": "fixture", "RISCV_REPO": "/repo"}):
            plan = self.plan()
            def interrupted(*args, **kwargs):
                record_execution_detail(remote_job={"job_id": "fixture"})
                raise RuntimeError("connection lost")
            with patch("codex_agent.operator_lifecycle.run_remote_operator", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    self.execute(plan)
            self.source.write_text(ORIGINAL_SOURCE + "\n# modified during disconnect\n")
            with patch("codex_agent.operator_lifecycle.run_remote_operator") as remote:
                with self.assertRaisesRegex(PermissionError, "Files changed during disconnect"):
                    self.execute(plan)
                remote.assert_not_called()

    def test_unkeyed_batch_children_cannot_reuse_parent_result_id(self):
        plan = self.plan()
        with patch.dict(os.environ, {"TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "0"}), \
             patch("codex_agent.execution_guard.execution_kind", return_value="job"), \
             patch("codex_agent.execution_guard.execution_details", return_value={"validation_result_run_id": "parent-id"}), \
             patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)):
            first = validate_operator_target(self.root, "demo", execute=True)
            second = validate_operator_target(self.root, "demo", execute=True)
        self.assertNotEqual(first.run_id, "parent-id")
        self.assertNotEqual(first.run_id, second.run_id)

    def test_failure_to_persist_completion_never_allows_replay(self):
        plan = self.plan()
        def save(path, value):
            if value.get("state") == "completed":
                raise OSError("simulated disk write failure")
            atomic_json(path, value)
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)), \
             patch("codex_agent.execution_guard.atomic_json", side_effect=save):
            with self.assertRaises(OSError):
                self.execute(plan)
        self.assertEqual(inspect_execution(self.root, "validation", plan.run_id)["state"], "unknown")
        with self.assertRaisesRegex(PermissionError, "do not replay"):
            self.execute(plan)

    def test_unknown_outcome_blocks_same_id_and_new_id_for_the_same_files(self):
        plan = self.plan()
        def run(*args, **kwargs):
            if kwargs["dry_run"]:
                return self.runner(plan)
            raise ConnectionError("response lost after remote launch")
        with patch("codex_agent.operator_lifecycle.run_operator", side_effect=run):
            with self.assertRaises(ConnectionError):
                self.execute(plan)
        self.assertEqual(inspect_execution(self.root, "validation", plan.run_id)["state"], "unknown")
        with self.assertRaisesRegex(PermissionError, "do not replay"):
            self.execute(plan)
        newer = self.plan()
        with self.assertRaisesRegex(PermissionError, "unresolved execution"):
            self.execute(newer)

    def test_crashed_running_journal_is_not_assumed_safe_to_repeat(self):
        plan = self.plan()
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)):
            self.execute(plan)
        path = journal_path(self.root, "validation", plan.run_id)
        value = json.loads(path.read_text())
        value["state"] = "running"
        atomic_json(path, value)
        with self.assertRaisesRegex(PermissionError, "outcome is running"):
            self.execute(plan)

    def test_same_key_different_args_or_changed_code_cannot_replay(self):
        plan = self.plan()
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)):
            self.execute(plan)
        with self.assertRaisesRegex(PermissionError, "different arguments"):
            self.execute(plan, timeout_seconds=10)
        self.source.write_text(ORIGINAL_SOURCE + "\n# changed\n")
        with self.assertRaisesRegex(PermissionError, "Files changed since completion"):
            self.execute(plan)

    def test_modified_receipt_cannot_replay_old_trusted_evidence(self):
        plan = self.plan()
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)):
            result = self.execute(plan)
        (self.root / result.receipt_path).write_text("{}")
        with self.assertRaisesRegex(PermissionError, "Stored receipt, log or patch changed"):
            self.execute(plan)

    def test_repair_replay_does_not_write_twice(self):
        from codex_agent.operator_lifecycle import propose_operator_repair, decide_repair_proposal, apply_operator_repair
        from codex_agent.tests.test_operator_lifecycle import REPLACEMENT_SOURCE
        plan = validate_operator_target(self.root, "demo")
        path = self.root / plan.receipt_path
        record = json.loads(path.read_text())
        record.update(status="failed", exit_code=1, failure_stage="correctness")
        atomic_json(path, record)
        proposal = propose_operator_repair(self.root, plan.run_id, REPLACEMENT_SOURCE, "fixture")
        decide_repair_proposal(self.root, proposal.proposal_id, approve=True, reviewer="test")
        with patch.dict(os.environ, {"TRITON_RISCV_ALLOW_REPAIR_APPLY": "1"}):
            first = apply_operator_repair(self.root, proposal.proposal_id)
            timestamp = self.source.stat().st_mtime_ns
            second = apply_operator_repair(self.root, proposal.proposal_id)
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(self.source.stat().st_mtime_ns, timestamp)

    def test_nested_operator_batch_uses_shared_locks_without_deadlock(self):
        from codex_agent.project_tools import prepare_validation_job, decide_validation_job, execute_validation_job
        job = prepare_validation_job(self.root, ["demo"], source_env=False, timeout_seconds=10)
        decide_validation_job(self.root, job["job_id"], approve=True, reviewer="test")
        from types import SimpleNamespace
        plan = SimpleNamespace(command=job["items"][0]["plan"]["command"])
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(plan)) as run:
            first = execute_validation_job(self.root, job["job_id"])
            second = execute_validation_job(self.root, job["job_id"])
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "passed", first)
        self.assertEqual(run.call_count, 2)

    def test_preexecution_rejection_does_not_claim_an_unknown_remote_run(self):
        plan = self.plan()
        self.source.write_text(ORIGINAL_SOURCE + "\n# changed\n")
        with self.assertRaisesRegex(PermissionError, "source or tests changed"):
            self.execute(plan)
        self.assertEqual(inspect_execution(self.root, "validation", plan.run_id)["state"], "rejected")
        newer = self.plan()
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=self.runner(newer)):
            self.assertEqual(self.execute(newer).status, "passed")

    def test_cross_process_resource_lock_and_unrelated_resource(self):
        code = """from pathlib import Path
import sys
from codex_agent.execution_guard import resource_locks
try:
    with resource_locks(Path(sys.argv[1]), [sys.argv[2]]): pass
except PermissionError:
    raise SystemExit(23)
"""
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        with resource_locks(self.root, ["file:shared"]):
            busy = subprocess.run([sys.executable, "-c", code, str(self.root), "file:shared"], env=env)
            other = subprocess.run([sys.executable, "-c", code, str(self.root), "file:other"], env=env)
        self.assertEqual(busy.returncode, 23)
        self.assertEqual(other.returncode, 0)

    def test_native_timeout_cannot_exceed_transport_budget(self):
        with self.assertRaisesRegex(ValueError, "900 second"):
            validate_operator_target(self.root, "demo", timeout_seconds=3600)

    def test_approval_wrapper_preserves_keyword_calling_contract(self):
        plan = validate_operator_target(self.root, "demo")
        result = decide_validation_plan(repo_root=self.root, run_id=plan.run_id, approve=True, reviewer="test")
        self.assertEqual(result["approval"]["status"], "approved")


class ProcessControlTests(unittest.TestCase):
    def test_file_lock_waits_until_other_task_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, release, entered = threading.Event(), threading.Event(), threading.Event()
            def task_a():
                with resource_locks(root, ["file:demo"]):
                    ready.set()
                    release.wait(2)
            def task_b():
                with resource_locks(root, ["file:demo"], wait_seconds=1):
                    entered.set()
            a = threading.Thread(target=task_a)
            a.start()
            self.assertTrue(ready.wait(1))
            b = threading.Thread(target=task_b)
            b.start()
            self.assertFalse(entered.wait(0.1))
            release.set()
            a.join(2)
            b.join(2)
            self.assertTrue(entered.is_set())

    def test_lock_wait_is_bounded_and_cancellable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, release = threading.Event(), threading.Event()
            def holder():
                with resource_locks(root, ["file:demo"]):
                    ready.set()
                    release.wait(2)
            thread = threading.Thread(target=holder)
            thread.start()
            self.assertTrue(ready.wait(1))
            try:
                with self.assertRaisesRegex(PermissionError, "budget exhausted"):
                    with resource_locks(root, ["file:demo"], wait_seconds=0.05):
                        self.fail("lock was not acquired")
                cancel = root / "cancel"
                cancel.touch()
                with execution_budget(5, cancel), self.assertRaises(ExecutionCancelled):
                    with resource_locks(root, ["file:demo"], wait_seconds=1):
                        self.fail("cancelled lock was acquired")
            finally:
                release.set()
                thread.join(2)
    def test_validation_does_not_inherit_model_keys_or_mutation_permissions(self):
        with patch.dict(os.environ, {"ISRC_API_KEY": "fixture-not-a-secret", "TRITON_RISCV_ALLOW_REPAIR_APPLY": "1",
                                    "TRITON_CACHE_DIR": "/tmp/cache", "SSH_AUTH_SOCK": "/fixture"}, clear=True):
            self.assertEqual(validation_environment(), {"TRITON_CACHE_DIR": "/tmp/cache"})

    def test_cancellation_terminates_process(self):
        with tempfile.TemporaryDirectory() as directory:
            cancel = Path(directory) / "cancel"
            timer = threading.Timer(0.1, lambda: cancel.touch())
            timer.start()
            try:
                with execution_budget(10, cancel), self.assertRaises(ExecutionCancelled):
                    run_bounded([sys.executable, "-c", "import time; time.sleep(20)"], timeout=10,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            finally:
                timer.join()

    def test_deadline_is_shared_across_subprocess_calls(self):
        with execution_budget(0.25):
            run_bounded([sys.executable, "-c", "import time; time.sleep(0.10)"], timeout=10)
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable, "-c", "import time; time.sleep(2)"], timeout=10)

    def test_timeout_kills_grandchild_even_when_parent_exits_on_term(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "escaped"
            child = ("import signal,time; from pathlib import Path; "
                     "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(1); "
                     f"Path({str(marker)!r}).touch()")
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(20)"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable, "-c", parent], timeout=0.2,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.1)
            self.assertFalse(marker.exists())
