"""Stage-one checkpoints on disposable MySQL; no model or remote execution."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from codex_agent.orchestration.models import InputVersions
from codex_agent.orchestration.schema import events, steps, tasks
from codex_agent.orchestration.store import WorkflowStore
from codex_agent.platform.store import PlatformStore
from codex_agent.storage.database import StorageError, engine
from codex_agent.tasks.schema import jobs, outbox
from codex_agent.tasks.store import Conflict, JobStore


CONTRACT = "a" * 64
TESTS = "b" * 64
SOURCE = "c" * 64


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        env = patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps({
            "schemaVersion": 1, "repoRoot": str(self.root), "stateDir": str(self.root / "agent-results")})})
        env.start()
        self.addCleanup(env.stop)
        self.platform = PlatformStore(self.root)
        self.session = self.platform.create_session("workflow fixture")
        self.store = WorkflowStore(self.root)
        self.request = dict(session_id=self.session["id"], kind="create", operator="fixture_operator",
                            requirement="Implement the specified operator; synthetic orchestration fixture only.")

    def create(self, key="request-1", **changes):
        return self.store.create({**self.request, **changes}, request_id=key)

    def start(self, task, role=None, **changes):
        role = role or {"preparing": "main", "planning_tests": "reviewer",
                        "implementing": "implementer", "reviewing": "reviewer"}[task["state"]]
        return self.store.start_step(task["id"], {"role": role, "instruction": "Fixture step", **changes},
            expected_revision=task["revision"], event_id="start-" + str(task["revision"]))

    def finish(self, task, *, status="completed", summary="Fixture result, not actual execution"):
        return self.store.finish_step(task["id"], task["steps"][-1]["id"],
            {"status": status, "summary": summary}, expected_revision=task["revision"],
            event_id="finish-" + str(task["revision"]))

    def advance(self, task, target, **versions):
        return self.store.advance(task["id"], target, {**task["versions"], **versions},
            expected_revision=task["revision"], event_id="advance-" + str(task["revision"]))

    def preparing_done(self):
        return self.finish(self.start(self.create()))

    def waiting_approval(self):
        task = self.advance(self.preparing_done(), "planning_tests", contract_sha256=CONTRACT)
        task = self.advance(self.finish(self.start(task)), "implementing", tests_sha256=TESTS)
        task = self.advance(self.finish(self.start(task)), "reviewing", source_sha256=SOURCE)
        return self.advance(self.finish(self.start(task)), "waiting_approval")

    def test_new_workflow_has_unknown_versions_no_jobs_and_no_fake_success(self):
        task = self.create()
        self.assertEqual(task["state"], "preparing")
        self.assertEqual(task["versions"], InputVersions().model_dump())
        self.assertEqual((task["revision"], task["repair_count"], task["steps"]), (1, 0, []))
        self.assertEqual(self.store.db.rows(jobs), [])
        self.assertEqual(self.store.db.rows(outbox), [])

    def test_concurrent_duplicate_submit_creates_one_workflow_and_event(self):
        with ThreadPoolExecutor(6) as pool:
            result = list(pool.map(lambda _: WorkflowStore(self.root).create(self.request, request_id="same"), range(18)))
        self.assertEqual(len({r["id"] for r in result}), 1)
        self.assertEqual(len(self.store.db.rows(tasks)), 1)
        self.assertEqual(len(self.store.db.rows(events)), 1)

    def test_submit_key_cannot_change_requirement_session_or_task_type(self):
        self.create()
        other = self.platform.create_session()
        for changes in ({"requirement": "other"}, {"kind": "repair"}, {"session_id": other["id"]}, {"max_repairs": 9}):
            with self.subTest(changes=changes), self.assertRaises(Conflict):
                self.create(**changes)

    def test_create_rejects_unknown_and_foreign_workspace_session(self):
        foreign = PlatformStore(self.root / "other").create_session()
        for session_id in ("missing", foreign["id"]):
            with self.subTest(session_id=session_id), self.assertRaises(KeyError):
                self.create(session_id=session_id)
        self.assertEqual(self.store.db.rows(tasks), [])

    def test_input_validation_rejects_invalid_keys_extra_authority_and_bad_versions(self):
        for changes in ({"kind": "shell"}, {"operator": "../outside"}, {"requirement": " "},
                        {"max_repairs": True}, {"approved": True}, {"versions": {"source_sha256": "not-a-hash"}}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.create(**changes)
        for key in ("", " ", "x" * 161, 1):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.create(key=key)

    def test_workspace_boundaries_cover_get_history_and_mutation(self):
        task = self.create()
        other = WorkflowStore(self.root / "other")
        for call in (lambda: other.get(task["id"]), lambda: other.history(task["id"]),
                     lambda: other.close(task["id"], "cancelled", reason="fixture", expected_revision=1, event_id="cancel")):
            with self.assertRaises(KeyError):
                call()
        self.assertEqual(other.pending(), [])

    def test_create_and_event_are_atomic(self):
        with patch.object(self.store, "_event", side_effect=RuntimeError("event storage failed")):
            with self.assertRaises(RuntimeError):
                self.create()
        self.assertEqual(self.store.db.rows(tasks), [])

    def test_pre_execution_lifecycle_stops_at_approval(self):
        task = self.waiting_approval()
        self.assertEqual(task["state"], "waiting_approval")
        self.assertEqual(task["recovery_action"], "wait")
        self.assertEqual([s["role"] for s in task["steps"]], ["main", "reviewer", "implementer", "reviewer"])
        self.assertEqual(task["versions"], dict(contract_sha256=CONTRACT, source_sha256=SOURCE, tests_sha256=TESTS))
        self.assertIn(task["id"], [r["id"] for r in self.store.pending()])

    def test_repair_task_is_recorded_without_faking_reproduction(self):
        task = self.create(kind="repair", versions={"source_sha256": SOURCE, "tests_sha256": TESTS})
        self.assertEqual(task["kind"], "repair")
        self.assertEqual(task["state"], "preparing")
        self.assertEqual(task["repair_count"], 0)
        self.assertEqual(task["result"], {})

    def test_no_stage_skipping_or_advancing_without_completed_work(self):
        task = self.create()
        for target in ("planning_tests", "implementing", "reviewing", "waiting_approval", "unknown"):
            with self.subTest(target=target), self.assertRaises(Conflict):
                self.advance(task, target, contract_sha256=CONTRACT, tests_sha256=TESTS, source_sha256=SOURCE)
        self.assertEqual(self.store.get(task["id"])["revision"], 1)

    def test_no_model_text_can_approve_or_declare_validation_success(self):
        task = self.waiting_approval()
        for target in ("waiting_validation", "repairing", "succeeded"):
            with self.subTest(target=target), self.assertRaises(PermissionError):
                self.advance(task, target)
        with self.assertRaises(ValueError):
            self.store.close(task["id"], "succeeded", reason="model claims passed",
                             expected_revision=task["revision"], event_id="fake")
        self.assertEqual(self.store.get(task["id"])["state"], "waiting_approval")

    def test_finished_agent_run_is_not_a_finished_workflow(self):
        task = self.create()
        run = self.platform.create_run(self.session["id"], "harness-agent", "fixture_operator", {})
        task = self.start(task, run_id=run["id"])
        self.platform.update_run(run["id"], status="completed", result={"status": "passed"})
        task = self.finish(task, summary="All tests passed (unverified model text)")
        self.assertEqual(task["state"], "preparing")
        self.assertEqual(task["result"], {})

    def test_roles_are_scoped_to_stage_and_steps_are_sequential(self):
        task = self.create()
        for role in ("implementer", "reviewer"):
            with self.subTest(role=role), self.assertRaises(Conflict):
                self.start(task, role=role)
        task = self.start(task, role="semantic")
        with self.assertRaises(Conflict):
            self.start(task)

    def test_step_submission_and_result_replay_are_idempotent(self):
        task = self.create()
        started = self.start(task)
        self.assertEqual(self.start(task), started)
        finished = self.finish(started)
        self.assertEqual(self.finish(started), finished)
        with self.assertRaises(Conflict):
            self.finish(started, summary="changed result under same event key")
        self.assertEqual(len(self.store.history(task["id"])), 3)

    def test_concurrent_result_delivery_records_one_completion(self):
        task = self.start(self.create())
        with ThreadPoolExecutor(5) as pool:
            results = list(pool.map(lambda _: self.finish(task), range(10)))
        self.assertEqual({r["revision"] for r in results}, {3})
        self.assertEqual(len(self.store.history(task["id"])), 3)

    def test_optimistic_revision_fences_competing_decisions(self):
        task = self.create()
        def attempt(n):
            try:
                return self.store.start_step(task["id"], {"role": "main", "instruction": "same snapshot"},
                    expected_revision=1, event_id="independent-" + str(n))
            except Conflict:
                return None
        with ThreadPoolExecutor(5) as pool:
            results = list(pool.map(attempt, range(10)))
        self.assertEqual(sum(r is not None for r in results), 1)

    def test_other_task_step_cannot_be_completed_even_with_latest_revision(self):
        left, right = self.start(self.create("left")), self.start(self.create("right"))
        with self.assertRaises(Conflict):
            self.store.finish_step(right["id"], left["steps"][-1]["id"],
                {"status": "completed", "summary": "wrong task"}, expected_revision=right["revision"], event_id="wrong")
        self.assertEqual(self.store.get(right["id"])["steps"][0]["status"], "running")

    def test_input_change_invalidates_late_result_even_if_caller_reloads_revision(self):
        task = self.start(self.create())
        updated = self.store.revise_inputs(task["id"], {"source_sha256": SOURCE}, reason="source edited",
                                          expected_revision=task["revision"], event_id="edit")
        with self.assertRaises(Conflict):
            self.store.finish_step(task["id"], task["steps"][0]["id"], {"status": "completed", "summary": "late"},
                                   expected_revision=updated["revision"], event_id="late")
        self.assertEqual(updated["steps"][0]["status"], "superseded")
        self.assertNotEqual(updated["input_epoch"], task["input_epoch"])

    def test_change_while_awaiting_approval_requires_new_implementation_and_review(self):
        task = self.waiting_approval()
        changed = self.store.revise_inputs(task["id"], {**task["versions"], "source_sha256": "d" * 64},
            reason="source edited during user review", expected_revision=task["revision"], event_id="edit")
        self.assertEqual(changed["state"], "implementing")
        self.assertTrue(all(s["status"] == "superseded" for s in changed["steps"]))
        with self.assertRaises(Conflict):
            self.advance(changed, "reviewing")
        self.assertEqual(self.store.history(task["id"], after=task["revision"])[0]["snapshot"]["versions"]["source_sha256"], "d" * 64)

    def test_contract_and_test_changes_rewind_to_the_correct_stage(self):
        task = self.waiting_approval()
        changed = self.store.revise_inputs(task["id"], {**task["versions"], "tests_sha256": "d" * 64},
            reason="new acceptance specification", expected_revision=task["revision"], event_id="tests")
        self.assertEqual(changed["state"], "planning_tests")
        changed = self.store.revise_inputs(task["id"], {**changed["versions"], "contract_sha256": "e" * 64},
            reason="user clarified semantics", expected_revision=changed["revision"], event_id="contract")
        self.assertEqual(changed["state"], "preparing")

    def test_advancing_cannot_silently_rewrite_frozen_upstream_inputs(self):
        task = self.advance(self.preparing_done(), "planning_tests", contract_sha256=CONTRACT)
        task = self.finish(self.start(task))
        with self.assertRaises(Conflict):
            self.advance(task, "implementing", contract_sha256="d" * 64, tests_sha256=TESTS)

    def test_missing_versions_and_failed_results_do_not_advance(self):
        task = self.preparing_done()
        with self.assertRaises(Conflict):
            self.advance(task, "planning_tests")
        task = self.finish(self.start(task), status="insufficient_evidence")
        with self.assertRaises(Conflict):
            self.advance(task, "planning_tests", contract_sha256=CONTRACT)

    def test_newest_failed_step_cannot_reuse_an_older_completed_step(self):
        task = self.finish(self.start(self.preparing_done()), status="failed")
        with self.assertRaises(Conflict):
            self.advance(task, "planning_tests", contract_sha256=CONTRACT)

    def test_step_and_checkpoint_event_rollback_together(self):
        task = self.create()
        with patch.object(self.store, "_event", side_effect=RuntimeError("event write failed")):
            with self.assertRaises(RuntimeError):
                self.start(task)
        self.assertEqual(self.store.get(task["id"]), task)

    def test_restart_reads_waiting_checkpoint_without_new_jobs(self):
        task = self.waiting_approval()
        source = str(Path(__file__).resolve().parents[2])
        code = ("import sys,json; sys.path.insert(0,sys.argv[1]); "
                "from codex_agent.orchestration.store import WorkflowStore; "
                "print(json.dumps(WorkflowStore(sys.argv[2]).get(sys.argv[3])))")
        result = subprocess.run([sys.executable, "-I", "-c", code, source, str(self.root), task["id"]],
                                capture_output=True, text=True, check=True, timeout=20)
        restored = json.loads(result.stdout)
        self.assertEqual(restored, task)
        self.assertEqual(self.store.db.rows(jobs), [])

    def test_unfinished_step_is_visible_after_restart_not_replayed(self):
        task = self.start(self.create())
        restored = WorkflowStore(self.root).get(task["id"])
        self.assertEqual(restored["recovery_action"], "inspect_unfinished_steps")
        with self.assertRaises(Conflict):
            self.start(restored)
        with self.assertRaises(Conflict):
            self.store.close(task["id"], "cancelled", reason="needs reconciliation",
                             expected_revision=restored["revision"], event_id="cancel")

    def test_terminal_states_are_explicit_and_no_longer_pending(self):
        for status in ("cancelled", "blocked", "failed"):
            task = self.create(status)
            closed = self.store.close(task["id"], status, reason="fixture stopping reason",
                                      expected_revision=task["revision"], event_id="stop")
            self.assertEqual(closed["result"]["reason"], "fixture stopping reason")
            self.assertEqual(self.store.close(task["id"], status, reason="fixture stopping reason",
                expected_revision=task["revision"], event_id="stop"), closed)
            with self.assertRaises(Conflict):
                self.start(closed, role="main")
        self.assertEqual(self.store.pending(), [])

    def test_unknown_workflow_and_bounded_cursors(self):
        with self.assertRaises(KeyError):
            self.store.get("missing")
        task = self.preparing_done()
        self.assertEqual([r["revision"] for r in self.store.history(task["id"], after=1, limit=1)], [2])
        for limit in (0, 201, True):
            with self.assertRaises(ValueError):
                self.store.pending(limit=limit)
        with self.assertRaises(ValueError):
            self.store.history(task["id"], after=-1)

    def test_run_and_job_links_are_checked_without_trusting_their_status(self):
        task = self.create()
        foreign_session = self.platform.create_session()
        foreign_run = self.platform.create_run(foreign_session["id"], "harness-agent", "fixture_operator", {})
        with self.assertRaises(Conflict):
            self.start(task, run_id=foreign_run["id"])
        run = self.platform.create_run(self.session["id"], "harness-agent", "fixture_operator", {})
        queue = JobStore(self.root)
        wrong = queue.submit("validation", {}, "wrong", run_id=run["id"])
        with self.assertRaises(Conflict):
            self.start(task, run_id=run["id"], job_id=wrong["id"])
        correct = queue.submit("agent", {}, "correct", run_id=run["id"])
        task = self.start(task, run_id=run["id"], job_id=correct["id"])
        self.assertEqual(task["steps"][0]["job_id"], correct["id"])

    def test_two_workflows_in_same_session_cannot_share_a_run(self):
        left, right = self.create("left"), self.create("right")
        run = self.platform.create_run(self.session["id"], "harness-agent", "fixture_operator", {})
        self.start(left, run_id=run["id"])
        with self.assertRaises(Conflict):
            self.start(right, run_id=run["id"])

    def test_concurrent_run_binding_has_one_owner(self):
        tasks = [self.create("task-" + str(n)) for n in range(4)]
        run = self.platform.create_run(self.session["id"], "harness-agent", "fixture_operator", {})
        def attempt(task):
            try:
                return self.start(task, run_id=run["id"])
            except Conflict:
                return None
        with ThreadPoolExecutor(4) as pool:
            result = list(pool.map(attempt, tasks))
        self.assertEqual(sum(r is not None for r in result), 1)

    def test_run_for_another_operator_is_rejected(self):
        task = self.create()
        run = self.platform.create_run(self.session["id"], "harness-agent", "another_operator", {})
        with self.assertRaises(Conflict):
            self.start(task, run_id=run["id"])

    def test_mutation_rolls_back_if_event_write_fails_after_step_completion(self):
        task = self.start(self.create())
        with patch.object(self.store, "_event", side_effect=RuntimeError("event write failed")):
            with self.assertRaises(RuntimeError):
                self.finish(task)
        self.assertEqual(self.store.get(task["id"]), task)

    def test_large_payload_bool_revision_and_repeated_final_result_are_rejected(self):
        task = self.start(self.create())
        with self.assertRaises(ValueError):
            self.store.finish_step(task["id"], task["steps"][0]["id"], {"status": "completed", "summary": "fixture"},
                                   expected_revision=True, event_id="bool")
        finished = self.finish(task)
        with self.assertRaises(Conflict):
            self.store.finish_step(task["id"], task["steps"][0]["id"], {"status": "completed", "summary": "fixture"},
                                   expected_revision=finished["revision"], event_id="another-completion")
        from codex_agent.orchestration.store import bounded
        with self.assertRaises(ValueError):
            bounded({"log": "x" * 65536})

    def test_migration_is_repeatable_and_schema_matches_declared_columns(self):
        cfg = Config()
        cfg.set_main_option("script_location", str(Path(__file__).parents[1] / "storage/migrations"))
        with engine().connect() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
            command.upgrade(cfg, "head")
            self.assertEqual(conn.execute(text("SELECT version_num FROM alembic_version")).scalar(), "0006_accounts")
            inspector = inspect(conn)
            for table in (tasks, steps, events):
                self.assertEqual({c["name"] for c in inspector.get_columns(table.name)}, set(table.c.keys()))
                self.assertEqual({idx["name"] for idx in inspector.get_indexes(table.name) if not idx["unique"]},
                                 {idx.name for idx in table.indexes})


class WorkflowMigrationTests(unittest.TestCase):
    def test_upgrade_preserves_old_data_and_resumes_partial_ddl_without_runtime_migration(self):
        url = make_url(os.environ["TRITON_MYSQL_URL"])
        name = "triton_workflow_test_" + uuid.uuid4().hex
        admin = create_engine(url.set(database=None), hide_parameters=True)
        target = create_engine(url.set(database=name), hide_parameters=True)
        try:
            with admin.begin() as conn:
                conn.execute(text(f"CREATE DATABASE {name} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"))
            cfg = Config()
            cfg.set_main_option("script_location", str(Path(__file__).parents[1] / "storage/migrations"))
            with target.connect() as conn:
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, "0004_memory_indexes")
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "TRITON_MYSQL_URL": url.set(database=name).render_as_string(hide_password=False),
                "TRITON_RISCV_CONFIG": json.dumps({"schemaVersion": 1, "repoRoot": folder, "stateDir": folder}),
            }):
                platform = PlatformStore(Path(folder))
                session = platform.create_session("preserve old session")
                run = platform.create_run(session["id"], "validate", "legacy_operator", {})
                platform.add_message(session["id"], "user", "preserve original requirement")
                job_store = JobStore(Path(folder))
                job = job_store.submit("agent", {"fixture": "unchanged"}, "legacy", run_id=run["id"])
                with target.connect() as conn:
                    before = {table: [dict(r) for r in conn.execute(text(f"SELECT * FROM {table}")).mappings()]
                              for table in ("sessions", "messages", "runs", "jobs", "outbox_events")}
                store = WorkflowStore(Path(folder))
                with self.assertRaises(StorageError):
                    store.get("missing-before-upgrade")
                with target.begin() as conn:
                    self.assertNotIn("operator_workflows", inspect(conn).get_table_names())
                    migration = importlib.import_module("codex_agent.storage.migrations.versions.0005_operator_workflows")
                    conn.execute(text("CREATE TABLE " + migration.TABLE_DDL[0] +
                                      " ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin"))
                for _ in range(2):
                    with target.connect() as conn:
                        cfg.attributes["connection"] = conn
                        command.upgrade(cfg, "head")
                with target.connect() as conn:
                    after = {table: [dict(r) for r in conn.execute(text(f"SELECT * FROM {table}")).mappings()]
                             for table in before}
                    self.assertEqual(before, after)
                self.assertEqual(job_store.get(job["id"])["status"], "queued")
                created = store.create(dict(session_id=session["id"], kind="repair", operator="legacy_operator",
                    requirement="fixture"), request_id="new-workflow")
                self.assertEqual(created["state"], "preparing")
        finally:
            target.dispose()
            with admin.begin() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS {name}"))
            admin.dispose()


if __name__ == "__main__":
    unittest.main()
