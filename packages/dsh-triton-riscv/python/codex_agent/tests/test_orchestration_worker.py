"""Synthetic proposal pipeline on disposable MySQL; never model or hardware results."""
from concurrent.futures import ThreadPoolExecutor
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from codex_agent.artifacts import ArtifactStore
from codex_agent.harness import HarnessRunResult
from codex_agent.messaging.broker import Broker
from codex_agent.messaging.relay import publish_once
from codex_agent.orchestration.coordinator import ProposalCoordinator
from codex_agent.orchestration.roles import operator_paths, sha256
from codex_agent.orchestration.runtime import RoleAgent
from codex_agent.orchestration.schema import tasks
from codex_agent.orchestration.service import WorkflowService
from codex_agent.orchestration.store import WorkflowStore
from codex_agent.platform.store import PlatformStore
from codex_agent.storage.schema import runs
from codex_agent.tasks.schema import artifacts, jobs, outbox
from codex_agent.tasks.store import Conflict, LeaseLost
from codex_agent.workers.handlers import execute_job
from codex_agent.workers.consumer import execute_delivery


CONTRACT = dict(formula="y=x*x", inputs=["x: contiguous tensor"], outputs=["y: same shape"],
    dtypes=["float32"], shapes=["one-dimensional, nonempty"], boundaries=["tail elements use mask"],
    tolerances="rtol=1e-4; atol=1e-4")
TESTS = "def test_fixture():\n    assert True  # synthetic routing fixture, not operator acceptance\n"
SOURCE = "def fixture_operator(x):\n    return x * x\n"


class ScriptedBackend:
    def __init__(self, change=None, decision="request_semantics"):
        self.calls, self.change, self.decision = [], change, decision

    def run(self, policy, prompt, *, session_id, on_event=None):
        request = json.loads(prompt.split("\n", 1)[1])
        self.calls.append(request)
        role, phase = request["role"], request["phase"]
        result = dict(status="completed", summary="Synthetic proposal only", evidence_refs=[], unknowns=[])
        if role == "main":
            result.update(decision=self.decision)
        elif role == "semantic":
            result.update(contract=dict(CONTRACT))
        elif phase == "planning_tests":
            path = operator_paths(request["operator"])[1]
            original = next((f for f in request["files"] if f["path"] == path), None)
            result.update(verdict="tests_proposed", test_candidate=dict(path=path,
                base_sha256=request["versions"]["tests_sha256"], content=original["content"] if original else TESTS))
        elif role == "implementer":
            result.update(candidate=dict(path=operator_paths(request["operator"])[0],
                base_sha256=request["versions"]["source_sha256"], content=SOURCE))
        else:
            result.update(verdict="acceptable")
        event = {"type": "fixture_usage", "input_tokens": 10, "output_tokens": 5}
        if on_event:
            on_event(event)
        if self.change:
            self.change(request, result)
        return HarnessRunResult(session_id, json.dumps(result), "completed", [event])


class WorkflowWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cfg = {"schemaVersion": 1, "repoRoot": str(self.root), "stateDir": str(self.root / "agent-results"),
                    "queue": {"enabled": True, "proposalWorkflows": True}}
        env = patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg)})
        env.start()
        self.addCleanup(env.stop)
        self.platform = PlatformStore(self.root)
        self.session = self.platform.create_session("synthetic workflow")
        self.service = WorkflowService(self.root)
        self.jobs, self.store = self.service.jobs, self.service.store
        self.request = dict(session_id=self.session["id"], kind="create", operator="fixture_operator",
                            requirement="Synthetic routing fixture: square the tensor under the supplied contract.")
        self.backend = ScriptedBackend()

    def submit(self, key="one", **kwargs):
        return self.service.submit(self.request, key, **kwargs)

    def claim(self, job):
        event = self.jobs.db.rows(outbox, outbox.c.job_id == job["id"], order=(outbox.c.generation.desc(),))[0]
        return self.jobs.claim(self.jobs.envelope(event), "agent")

    def execute(self, job):
        claimed = self.claim(job)
        execute_job(self.root, job["id"], claimed["owner"], proposal_agent=RoleAgent(backend=self.backend))
        return self.jobs.get(job["id"])

    def task(self, job):
        return self.store.get(job["payload"]["workflow_id"])

    def seed_repair(self):
        self.request["kind"] = "repair"
        for path, text in zip(operator_paths("fixture_operator"), ("def fixture_operator(x):\n    return x\n", TESTS)):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def test_create_full_worker_route_stops_at_approval_without_writes_or_validation(self):
        job = self.submit()
        result = self.execute(job)
        self.assertEqual(result["status"], "succeeded", result)
        task = self.task(job)
        self.assertEqual(task["state"], "waiting_approval")
        self.assertEqual([r["role"] for r in self.backend.calls], ["main", "semantic", "reviewer", "implementer", "reviewer"])
        self.assertEqual(result["result"]["validation"], "not_started")
        self.assertFalse(result["result"]["files_applied"])
        self.assertFalse((self.root / "python").exists())
        self.assertEqual(self.jobs.db.rows(jobs, jobs.c.kind.in_(["memory", "validation", "index"])), [])
        self.assertEqual(len(self.jobs.db.rows(artifacts)), 6)
        planning = self.backend.calls[2]
        self.assertFalse(planning["files"])
        self.assertEqual(self.backend.calls[3]["files"][0]["content"], TESTS)
        review = self.backend.calls[4]
        self.assertCountEqual(review["files"], [
            dict(path=operator_paths("fixture_operator")[1], content=TESTS, sha256=sha256(TESTS)),
            dict(path=operator_paths("fixture_operator")[0], content_ref="/candidate/content", sha256=sha256(SOURCE))])
        self.assertEqual(review["candidate"]["content"], SOURCE)
        self.assertIsNone(review["candidate"]["base_sha256"])
        self.assertIsNone(review["review_baseline"])
        self.assertIsNone(planning["candidate"])

    def test_confirmed_contract_skips_semantic_role(self):
        self.backend.decision = "plan_tests"
        result = self.execute(self.submit(contract=CONTRACT))
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(len(self.backend.calls), 4)
        self.assertNotIn("semantic", [r["role"] for r in self.backend.calls])

    def test_blocked_role_exposes_missing_conditions_not_optimistic_summary(self):
        for role in ("main", "semantic"):
            with self.subTest(role=role):
                def missing(request, result):
                    if request["role"] == role:
                        result.update(summary="Ready for the next step", unknowns=["Confirm dtype and tolerance"])
                self.backend.change = missing
                result = self.execute(self.submit(key="unknown-" + role))
                self.assertEqual(result["result"]["workflow_state"], "blocked")
                self.assertIn("Confirm dtype and tolerance", result["result"]["reason"])
                self.assertNotIn("Ready for the next step", result["result"]["reason"])

    def test_repair_keeps_disk_and_fixed_acceptance_tests(self):
        self.seed_repair()
        before = {p: (self.root / p).read_bytes() for p in operator_paths("fixture_operator")}
        result = self.execute(self.submit())
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(before, {p: (self.root / p).read_bytes() for p in before})
        review = self.backend.calls[-1]
        self.assertEqual(review["candidate"]["content"], SOURCE)
        self.assertNotEqual(review["candidate"]["content"], before[operator_paths("fixture_operator")[0]].decode())
        self.assertEqual(review["review_baseline"]["content"], before[operator_paths("fixture_operator")[0]].decode())
        self.assertEqual(review["candidate"]["base_sha256"], review["review_baseline"]["sha256"])
        self.assertNotEqual(review["candidate"]["base_sha256"], review["versions"]["source_sha256"])

    def test_repair_cannot_replace_tests(self):
        self.seed_repair()
        def weaken(request, result):
            if request["phase"] == "planning_tests":
                result["test_candidate"]["content"] = "pass\n"
        self.backend.change = weaken
        job = self.submit()
        result = self.execute(job)
        self.assertEqual(self.task(job)["state"], "blocked")
        self.assertIn("acceptance tests", result["result"]["reason"])
        self.assertEqual(len(self.backend.calls), 3)

    def test_duplicate_submission_has_one_workflow_run_job_and_outbox(self):
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: WorkflowService(self.root).submit(self.request, "same"), range(8)))
        self.assertEqual(len({r["id"] for r in results}), 1)
        for table in (tasks, runs, jobs, outbox):
            self.assertEqual(len(self.jobs.db.rows(table)), 1)

    def test_submission_failure_rolls_back_workflow_and_run(self):
        with patch.object(self.jobs, "submit", side_effect=RuntimeError("queue failed")), self.assertRaises(RuntimeError):
            self.submit()
        for table in (tasks, runs, jobs, outbox):
            self.assertFalse(self.jobs.db.rows(table))

    def test_duplicate_key_rejects_changed_evidence(self):
        self.submit()
        with self.assertRaises(Conflict):
            self.submit(evidence=[dict(id="changed", kind="log", source="synthetic", text="changed log")])

    def test_create_collision_and_repair_missing_files_are_rejected(self):
        self.request["kind"] = "repair"
        with self.assertRaises(ValueError):
            self.submit()
        self.seed_repair()
        self.request["kind"] = "create"
        with self.assertRaises(Conflict):
            self.submit()

    def test_caller_cannot_forge_versions_or_reserved_evidence_ids(self):
        self.request["versions"] = {"source_sha256": "0" * 64}
        with self.assertRaises(Conflict):
            self.submit()
        self.request.pop("versions")
        with self.assertRaises(ValueError):
            self.submit(evidence=[dict(id="workflow_contract", kind="contract", source="synthetic", text="fake")])

    def test_default_worker_does_not_instantiate_a_model_adapter(self):
        job = self.submit()
        claimed = self.claim(job)
        self.cfg["queue"].pop("proposalWorkflows")
        with patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg)}), \
                patch("codex_agent.orchestration.runtime.RoleAgent", side_effect=AssertionError("API forbidden")) as model:
            execute_job(self.root, job["id"], claimed["owner"])
            with self.assertRaises(PermissionError):
                self.submit("disabled")
        model.assert_not_called()
        self.assertEqual(self.jobs.get(job["id"])["status"], "failed")
        self.assertEqual(self.jobs.get(job["id"])["effects_started"], 0)
        self.assertFalse(self.task(job)["steps"])

    def test_enabled_worker_constructs_role_adapter_not_ordinary_chat_loop(self):
        job = self.submit()
        claimed = self.claim(job)
        with patch("codex_agent.harness.HarnessSettings.from_env", return_value=object()) as settings, \
                patch("codex_agent.orchestration.runtime.RoleAgent", return_value=RoleAgent(backend=self.backend)) as adapter, \
                patch("codex_agent.harness.HarnessAgent", side_effect=AssertionError("ordinary loop forbidden")):
            execute_job(self.root, job["id"], claimed["owner"])
        self.assertEqual(adapter.call_count, 1)
        self.assertEqual(settings.call_count, 1)
        self.assertEqual(self.jobs.get(job["id"])["status"], "succeeded")

    def test_missing_requirements_stop_with_specific_question(self):
        def ask(request, result):
            if request["role"] == "main":
                result.update(decision="ask_user", questions=["Which dtype is required?"])
        self.backend.change = ask
        result = self.execute(self.submit())
        self.assertEqual(result["result"]["workflow_state"], "blocked")
        self.assertIn("Which dtype", result["result"]["reason"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_incomplete_contract_is_not_filled_in_by_program(self):
        def incomplete(request, result):
            if request["role"] == "semantic":
                result["contract"]["tolerances"] = None
        self.backend.change = incomplete
        result = self.execute(self.submit())
        self.assertEqual(result["result"]["workflow_state"], "blocked")
        self.assertEqual(len(self.backend.calls), 2)

    def test_invalid_model_output_is_saved_but_not_accepted(self):
        self.backend.change = lambda request, result: result.update(approved=True)
        job = self.submit()
        result = self.execute(job)
        self.assertEqual(result["result"]["workflow_state"], "blocked")
        record = self.jobs.db.rows(artifacts, artifacts.c.name != "result.json")[0]
        audit = json.loads(ArtifactStore(self.root).read(record))
        self.assertEqual(audit["outcome"], "invalid_output")
        self.assertIn('"approved": true', audit["raw"]["final_response"])
        self.assertIn("Output JSON schema", audit["system_prompt"])
        self.assertIn("TASK_DATA_JSON", audit["task_prompt"])
        self.assertTrue(audit["raw"]["events"])

    def test_review_disagreement_blocks_instead_of_claiming_success(self):
        def reject(request, result):
            if request["phase"] == "reviewing":
                result.update(verdict="changes_requested", findings=["Tail mask is missing"])
        self.backend.change = reject
        result = self.execute(self.submit())
        self.assertEqual(result["result"]["workflow_state"], "blocked")
        self.assertIn("Tail mask", result["result"]["reason"])

    def test_disk_changes_while_role_runs_invalidate_candidate(self):
        self.seed_repair()
        def external_edit(request, result):
            (self.root / operator_paths("fixture_operator")[0]).write_text("# user changed file\n")
        self.backend.change = external_edit
        result = self.execute(self.submit())
        self.assertEqual(result["result"]["workflow_state"], "blocked")
        self.assertIn("changed", result["result"]["reason"])

    def test_unknown_model_outcome_is_not_automatically_repeated(self):
        def timeout(request, result):
            raise TimeoutError("synthetic timeout")
        self.backend.change = timeout
        job = self.submit()
        result = self.execute(job)
        self.assertEqual(result["status"], "needs-reconciliation")
        self.assertEqual(self.task(job)["recovery_action"], "inspect_unfinished_steps")
        self.assertIsNone(self.claim(job))
        record = self.jobs.db.rows(artifacts)[0]
        audit = json.loads(ArtifactStore(self.root).read(record))
        self.assertEqual(audit["outcome"], "unknown")
        self.assertTrue(audit["notifications"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_restart_after_saved_implementation_does_not_call_it_again(self):
        job = self.submit()
        claimed = self.claim(job)
        agent = RoleAgent(backend=self.backend)
        coordinator = ProposalCoordinator(self.jobs, claimed, claimed["owner"], agent=agent)
        advance = coordinator.store.advance
        def crash(identifier, target, *args, **kwargs):
            if target == "reviewing":
                raise RuntimeError("crash after durable implementation")
            return advance(identifier, target, *args, **kwargs)
        with patch.object(coordinator.store, "advance", side_effect=crash), self.assertRaises(RuntimeError):
            coordinator.run()
        self.assertEqual(self.task(job)["steps"][-1]["status"], "completed")
        restored = ProposalCoordinator(self.jobs, claimed, claimed["owner"], agent=agent).run()
        self.assertEqual(restored["workflow_state"], "waiting_approval")
        self.assertEqual(len(self.backend.calls), 5)
        again = ProposalCoordinator(self.jobs, claimed, claimed["owner"], agent=agent).run()
        self.assertEqual(again, restored)
        self.assertEqual(len(self.backend.calls), 5)

    def test_expired_worker_cannot_store_role_result(self):
        def expire(request, result):
            with self.jobs.db.transaction() as conn:
                self.jobs.db.update(conn, jobs, {"lease_until": 1}, jobs.c.id == job["id"])
        job = self.submit()
        self.backend.change = expire
        with self.assertRaises(LeaseLost):
            self.execute(job)
        self.assertFalse(self.jobs.db.rows(artifacts))
        self.assertEqual(self.task(job)["steps"][-1]["status"], "running")
        self.jobs.recover()
        self.assertEqual(self.jobs.get(job["id"])["status"], "needs-reconciliation")

    def test_foreign_task_binding_rejected_before_role_call(self):
        job = self.submit()
        other = self.submit("second")
        claimed = self.claim(job)
        payload = dict(job["payload"], workflow_id=other["payload"]["workflow_id"])
        claimed["payload_json"] = json.dumps(payload)
        with self.assertRaises(Conflict):
            ProposalCoordinator(self.jobs, claimed, claimed["owner"], agent=RoleAgent(backend=self.backend)).run()
        self.assertFalse(self.backend.calls)

    @unittest.skipUnless(os.environ.get("TRITON_TEST_AMQP_URL"), "temporary RabbitMQ required")
    def test_real_broker_child_worker_and_harness_with_loopback_only_model(self):
        backend, received = self.backend, []
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(body)
                prompt = next(s for s in strings(body["input"]) if s.startswith("TASK_DATA_JSON"))
                response = backend.run("fixture", prompt, session_id="fixture").final_response
                item = {"type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": response, "annotations": []}]}
                events = [
                    {"type": "response.created", "response": {"id": "resp_fixture", "status": "in_progress", "output": []}},
                    {"type": "response.output_item.added", "output_index": 0,
                     "item": {**item, "status": "in_progress", "content": []}},
                    {"type": "response.content_part.added", "item_id": item["id"], "output_index": 0,
                     "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
                    {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0,
                     "content_index": 0, "delta": response},
                    {"type": "response.output_text.done", "item_id": item["id"], "output_index": 0,
                     "content_index": 0, "text": response},
                    {"type": "response.content_part.done", "item_id": item["id"], "output_index": 0,
                     "content_index": 0, "part": item["content"][0]},
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response.completed", "response": {"id": "resp_fixture", "status": "completed", "output": [item],
                     "usage": {"input_tokens": 32, "output_tokens": 16, "total_tokens": 48,
                               "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}},
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        job = self.submit()
        async def run():
            async with Broker(self.jobs) as broker:
                await publish_once(self.jobs, broker)
                delivery = await broker.queues["agent"].get(timeout=5)
                await execute_delivery(self.jobs, delivery, "agent")
                self.assertTrue(delivery.processed)
                event = self.jobs.db.rows(outbox, outbox.c.job_id == job["id"])[0]
                await broker.publish(event)
                duplicate = await broker.queues["agent"].get(timeout=5)
                await execute_delivery(self.jobs, duplicate, "agent")
        try:
            with patch.dict(os.environ, {"TRITON_AMQP_URL": os.environ["TRITON_TEST_AMQP_URL"],
                "ISRC_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "DSH_API_KEY_ENV": "ROLE_FIXTURE_KEY",
                "ROLE_FIXTURE_KEY": "synthetic-local-only", "DSH_MODEL": "fixture-model", "DSH_PROVIDER": "fixture",
                "DSH_REQUEST_TIMEOUT_SECONDS": "15", "NO_PROXY": "127.0.0.1,localhost"}):
                asyncio.run(run())
            result = self.jobs.get(job["id"])
            self.assertEqual(result["status"], "succeeded", result)
            self.assertEqual(self.task(job)["state"], "waiting_approval")
            self.assertEqual(result["attempt"], 1)
            self.assertEqual(len(received), 5)
            for body in received:
                self.assertEqual(body["model"], "fixture-model")
                self.assertFalse(body.get("tools"))
            self.assertFalse((self.root / "python").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
