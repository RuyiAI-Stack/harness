"""Use a disposable MySQL schema; RabbitMQ cases require TRITON_TEST_AMQP_URL."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from codex_agent.artifacts import ArtifactStore
from codex_agent.harness import HarnessAgent, HarnessSettings
from codex_agent.messaging.broker import Broker
from codex_agent.messaging.relay import publish_once
from codex_agent.platform.api import create_app
from codex_agent.platform.store import PlatformStore
from codex_agent.tasks.schema import jobs, outbox, attempts, inbox
from codex_agent.tasks.service import TaskService
from codex_agent.tasks.store import JobStore, Conflict, LeaseLost, now_ms
from codex_agent.tests.test_platform_service import FakeHarnessBackend
from codex_agent.workers.consumer import execute_delivery
from codex_agent.workers.handlers import execute_job


class Fixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cfg = {"schemaVersion": 1, "repoRoot": str(self.root), "stateDir": str(self.root / "agent-results"), "queue": {"enabled": True},
                    "permissions": {"validation": True}}
        env = patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(self.cfg),
            "TRITON_AMQP_URL": os.environ.get("TRITON_TEST_AMQP_URL", "amqp://guest:guest@127.0.0.1:1/")})
        env.start()
        self.addCleanup(env.stop)
        self.store = JobStore(self.root)

    def submit(self, key="one", kind="validation"):
        return self.store.submit(kind, {"fixture": True}, key)

    def event(self, job):
        return self.store.db.rows(outbox, outbox.c.job_id == job["id"], order=(outbox.c.generation.desc(),))[0]

    def claim(self, job):
        return self.store.claim(self.store.envelope(self.event(job)), job["kind"])

    def expire(self, job):
        with self.store.db.transaction() as conn:
            self.store.db.update(conn, jobs, {"lease_until": 1}, jobs.c.id == job["id"])


class TaskQueueTests(Fixture, unittest.TestCase):
    def test_secrets_are_removed_from_generated_test_environment(self):
        from codex_agent.process_control import validation_environment
        with patch.dict(os.environ, {"TRITON_AMQP_URL": "amqp://private"}):
            self.assertNotIn("TRITON_AMQP_URL", validation_environment())

    def test_queue_capacity_rejects_without_partial_write(self):
        self.store.config = self.store.config.model_copy(update={"maxPending": 1})
        self.submit()
        with self.assertRaises(Conflict):
            self.submit("second")
        self.assertEqual(len(self.store.db.rows(jobs)), 1)
        self.assertEqual(len(self.store.db.rows(outbox)), 1)

    def test_wrong_execution_host_defers_without_running(self):
        job = self.submit()
        with patch("codex_agent.tasks.store.socket.gethostname", return_value="different-machine"):
            self.assertIsNone(self.claim(job))
        self.assertEqual(self.store.get(job["id"])["attempt"], 0)
        self.assertEqual(self.store.get(job["id"])["generation"], 2)
        self.assertEqual(self.store.get(job["id"])["result"]["reason"], "worker host mismatch")

    def test_validation_uses_existing_domain_guard_and_enqueues_memory(self):
        from codex_agent.operator_lifecycle import validate_operator_target, decide_validation_plan
        from codex_agent.validate_operator import OperatorValidationResult
        from codex_agent.tests.test_operator_lifecycle import ORIGINAL_SOURCE, TEST_SOURCE
        folder = self.root / "python/examples/flaggems"
        folder.mkdir(parents=True)
        (folder / "demo.py").write_text(ORIGINAL_SOURCE)
        (folder / "test_demo.py").write_text(TEST_SOURCE)
        plan = validate_operator_target(self.root, "demo", source_env=False)
        decide_validation_plan(self.root, plan.run_id, approve=True, reviewer="fixture")
        job = TaskService(self.root).submit_validation("demo", plan.run_id, "validate")
        claim = self.claim(job)
        log = self.root / "synthetic-validation.log"
        log.write_text("Synthetic hardware boundary fixture; not a RISC-V execution.\n1 passed\n")
        fake = OperatorValidationResult(operator="demo", implementation_file="python/examples/flaggems/demo.py",
            test_files=["python/examples/flaggems/test_demo.py"], command=plan.command, dry_run=False,
            exit_code=0, status="passed", failure_stage=None, likely_reason=None, error_excerpt=[],
            duration_seconds=.1, log_path=str(log))
        with patch("codex_agent.operator_lifecycle.run_operator", return_value=fake), \
             patch("codex_agent.operator_lifecycle.remember_validation") as inline_memory:
            execute_job(self.root, job["id"], claim["owner"])
        inline_memory.assert_not_called()
        finished = self.store.get(job["id"])
        self.assertEqual(finished["status"], "succeeded", finished)
        self.assertEqual(finished["result"]["memory_write"]["status"], "deferred")
        followup = self.store.db.rows(jobs, jobs.c.kind == "memory")
        self.assertEqual(len(followup), 1)
        self.assertEqual(len(self.store.detail(job["id"])["artifacts"]), 3)
        memory = self.claim(self.store.public(followup[0]))
        execute_job(self.root, memory["id"], memory["owner"])
        self.assertEqual(self.store.get(memory["id"])["status"], "succeeded")
        self.assertEqual(self.store.detail(job["id"])["followups"][0]["status"], "succeeded")

    def test_task_and_outbox_rollback_together(self):
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.store.db.transaction() as conn:
                self.store.submit("agent", {}, "rollback", connection=conn)
                raise RuntimeError("rollback")
        self.assertEqual(self.store.db.rows(jobs), [])
        self.assertEqual(self.store.db.rows(outbox), [])

    def test_concurrent_idempotency_and_conflicting_input(self):
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(lambda _: self.submit(), range(20)))
        self.assertEqual(len({r["id"] for r in results}), 1)
        self.assertEqual(len(self.store.db.rows(outbox)), 1)
        with self.assertRaises(Conflict):
            self.store.submit("validation", {"different": True}, "one")

    def test_invalid_kind_payload_and_key(self):
        for kind, payload, key in [("shell", {}, "k"), ("agent", {}, ""), ("agent", {"x": "x" * 40000}, "k")]:
            with self.assertRaises(ValueError):
                self.store.submit(kind, payload, key)

    def test_duplicate_delivery_claims_one_attempt(self):
        job = self.submit()
        with ThreadPoolExecutor(6) as pool:
            claims = list(pool.map(lambda _: self.claim(job), range(12)))
        self.assertEqual(sum(c is not None for c in claims), 1)
        self.assertEqual(len(self.store.db.rows(attempts)), 1)
        claim = next(c for c in claims if c)
        self.store.finish(job["id"], claim["owner"], "succeeded", {"fixture": "done"})
        self.assertIsNone(self.claim(job))

    def test_envelope_is_reference_not_authority(self):
        job = self.submit()
        envelope = self.store.envelope(self.event(job))
        self.assertNotIn("payload", envelope)
        for bad in [{**envelope, "job_id": "wrong"}, {**envelope, "command": "touch unsafe"},
                    {**envelope, "workspace_id": "other"}, {**envelope, "version": True}]:
            with self.assertRaises((ValueError, KeyError)):
                self.store.claim(bad, "validation")
        self.assertEqual(self.store.get(job["id"])["attempt"], 0)

    def test_other_workspace_cannot_read_task(self):
        job = self.submit()
        with self.assertRaises(KeyError):
            JobStore(self.root / "other").get(job["id"])

    def test_mutations_are_fifo_and_memory_is_independent(self):
        first, second = self.submit("first"), self.submit("second")
        self.assertIsNone(self.claim(second))
        running = self.claim(first)
        self.assertIsNone(self.claim(second))
        memory = self.submit("memory", "memory")
        self.assertIsNotNone(self.claim(memory))
        self.store.finish(first["id"], running["owner"], "succeeded", {})
        self.assertIsNotNone(self.claim(second))

    def test_no_effects_recovery_requeues_but_old_owner_cannot_write(self):
        job = self.submit()
        old = self.claim(job)
        self.expire(job)
        self.assertEqual(self.store.recover()[0]["status"], "queued")
        fresh = self.claim(job)
        self.assertNotEqual(old["owner"], fresh["owner"])
        with self.assertRaises(LeaseLost):
            self.store.finish(job["id"], old["owner"], "succeeded", {})

    def test_effectful_crash_blocks_replay_and_requires_explicit_reconciliation(self):
        job = self.submit()
        claim = self.claim(job)
        self.store.begin_effects(job["id"], claim["owner"])
        self.expire(job)
        self.assertEqual(self.store.recover()[0]["status"], "needs-reconciliation")
        self.assertIsNone(self.claim(job))
        with self.assertRaises(ValueError):
            self.store.reconcile(job["id"], "")
        result = self.store.reconcile(job["id"], "fixture external process checked and stopped")
        self.assertEqual(result["status"], "failed")
        self.assertIn("operator-attested", result["result"]["verification"])

    def test_retry_budget_eventually_dead(self):
        job = self.submit()
        for _ in range(3):
            self.claim(job)
            self.expire(job)
            self.store.recover()
        self.assertEqual(self.store.get(job["id"])["status"], "dead")
        self.assertEqual(len(self.store.db.rows(attempts)), 3)

    def test_cancel_queued_never_executes_and_running_is_only_requested(self):
        job = self.submit()
        self.assertEqual(self.store.cancel(job["id"])["status"], "cancelled")
        self.assertIsNone(self.claim(job))
        other = self.submit("other")
        claim = self.claim(other)
        self.assertEqual(self.store.cancel(other["id"])["status"], "running")
        with self.assertRaises(LeaseLost):
            self.store.begin_effects(other["id"], claim["owner"])

    def test_outbox_concurrent_publishers_and_expired_publisher(self):
        self.submit()
        with ThreadPoolExecutor(4) as pool:
            claims = list(pool.map(lambda _: self.store.claim_publication(), range(8)))
        self.assertEqual(sum(c is not None for c in claims), 1)
        event = next(c for c in claims if c)
        with self.store.db.transaction() as conn:
            self.store.db.update(conn, outbox, {"lease_until": 1}, outbox.c.id == event["id"])
        newer = self.store.claim_publication()
        with self.assertRaises(LeaseLost):
            self.store.publication_result(event)
        self.store.publication_result(newer)
        self.assertEqual(self.store.db.rows(outbox)[0]["status"], "published")

    def test_artifacts_bound_to_attempt_and_checksum(self):
        job = self.submit()
        claim = self.claim(job)
        storage = ArtifactStore(self.root)
        identifier = storage.put(self.store, job["id"], claim["owner"], "result.json", b'{"ok":true}')
        record = self.store.detail(job["id"])["artifacts"][0]
        self.assertEqual(record["id"], identifier)
        self.assertEqual(storage.read(record), b'{"ok":true}')
        with self.assertRaises(ValueError):
            storage.put(self.store, job["id"], claim["owner"], "../bad", b"no")
        (storage.folder / record["storage_key"]).write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            storage.read(record)

    def test_finish_and_memory_followup_atomic(self):
        job = self.submit()
        claim = self.claim(job)
        with patch.object(self.store, "submit", side_effect=RuntimeError("db unavailable")):
            with self.assertRaises(RuntimeError):
                self.store.finish(job["id"], claim["owner"], "succeeded", {}, followup={"receipt": "fixture"})
        self.assertEqual(self.store.get(job["id"])["status"], "running")
        self.store.finish(job["id"], claim["owner"], "succeeded", {}, followup={"receipt": "fixture"})
        self.assertEqual(len(self.store.db.rows(jobs)), 2)
        self.assertEqual(len(self.store.db.rows(outbox)), 2)

    def test_validation_needs_real_approval_and_rechecks_changed_inputs(self):
        from codex_agent.operator_lifecycle import validate_operator_target, decide_validation_plan
        from codex_agent.tests.test_operator_lifecycle import ORIGINAL_SOURCE, TEST_SOURCE
        folder = self.root / "python/examples/flaggems"
        folder.mkdir(parents=True)
        implementation = folder / "demo.py"
        implementation.write_text(ORIGINAL_SOURCE)
        (folder / "test_demo.py").write_text(TEST_SOURCE)
        plan = validate_operator_target(self.root, "demo", source_env=False)
        service = TaskService(self.root)
        with self.assertRaises(PermissionError):
            service.submit_validation("demo", plan.run_id, "validate")
        decide_validation_plan(self.root, plan.run_id, approve=True, reviewer="fixture")
        job = service.submit_validation("demo", plan.run_id, "validate")
        claim = self.claim(job)
        implementation.write_text(ORIGINAL_SOURCE + "\n# changed after approval\n")
        with patch("codex_agent.operator_lifecycle.validate_operator_target") as execute:
            execute_job(self.root, job["id"], claim["owner"])
        execute.assert_not_called()
        self.assertEqual(self.store.get(job["id"])["status"], "failed")

    def test_http_submission_is_durable_and_duplicate_does_not_add_message(self):
        agent = HarnessAgent(HarnessSettings.from_env(self.root, {}), FakeHarnessBackend())
        app = create_app(self.root, harness_agent=agent)
        with TestClient(app) as client:
            session = client.post("/api/sessions", json={"title": "fixture"}).json()
            url = f'/api/sessions/{session["id"]}/messages'
            payload = {"content": "inspect demo", "request_id": "request-1"}
            response = client.post(url, json=payload)
            self.assertEqual(response.status_code, 202, response.text)
            first = response.json()
            self.assertEqual(first["run"]["status"], "queued")
            self.assertEqual(client.post(url, json=payload).json()["job"]["id"], first["job"]["id"])
            self.assertEqual(len(app.state.store.list_messages(session["id"])), 1)
            self.assertEqual(client.post(url, json={**payload, "content": "different"}).status_code, 409)
            self.assertEqual(client.post(url, json={"content": "no key"}).status_code, 400)
            job = first["job"]
            claim = self.claim(job)
            with patch("codex_agent.harness.HarnessAgent", return_value=agent):
                execute_job(self.root, job["id"], claim["owner"])
            detail = client.get('/api/jobs/' + job["id"]).json()
            self.assertEqual(detail["status"], "succeeded", detail)
            run = client.get('/api/runs/' + first["run"]["id"]).json()
            self.assertEqual(run["status"], "completed")
            self.assertGreater(len(client.get('/api/runs/' + run["id"] + '/events').json()), 1)

    def test_completed_run_recovers_without_second_model_call(self):
        platform = PlatformStore(self.root)
        session = platform.create_session()
        from types import SimpleNamespace
        submitted = TaskService(self.root).submit_turn(SimpleNamespace(store=platform), session["id"], "fixture", "once")
        job = submitted["job"]
        claim = self.claim(job)
        self.store.begin_effects(job["id"], claim["owner"])
        platform.update_run(submitted["run"]["id"], status="completed", result={"fixture": "committed"})
        self.expire(job)
        self.assertEqual(self.store.recover()[0]["status"], "succeeded")
        self.assertEqual(self.store.get(job["id"])["result"], {"fixture": "committed"})

    def test_queued_cancellation_emits_terminal_event_for_sse(self):
        from types import SimpleNamespace
        platform = PlatformStore(self.root)
        session = platform.create_session()
        submitted = TaskService(self.root).submit_turn(SimpleNamespace(store=platform), session["id"], "fixture", "once")
        self.store.cancel(submitted["job"]["id"])
        events = platform.list_events(submitted["run"]["id"])
        self.assertEqual(events[-1]["event_type"], "cancelled")
        self.assertEqual(events[-1]["payload"]["job_id"], submitted["job"]["id"])


@unittest.skipUnless(os.environ.get("TRITON_TEST_AMQP_URL"), "temporary RabbitMQ required")
class RabbitMQTests(Fixture, unittest.IsolatedAsyncioTestCase):
    async def test_separate_relay_and_worker_cli_processes(self):
        job = self.store.submit("memory", {"receipt_path": "missing.json", "sha256": "0" * 64}, "cli")
        for role in ("relay", "memory"):
            result = await asyncio.to_thread(subprocess.run,
                [sys.executable, "-I", "-m", "codex_agent.workers", role, "--workspace", str(self.root), "--once"],
                capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
        detail = self.store.detail(job["id"])
        self.assertEqual(detail["deliveries"][0]["status"], "published")
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(detail["result"]["error_type"], "ValueError")

    async def test_timeout_stops_real_child_and_does_not_claim_success(self):
        job = self.submit()
        original = asyncio.create_subprocess_exec
        children = []
        async def controlled_child(*_args, **kwargs):
            child = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
            children.append(child)
            return child
        # Only shorten this fixture's wall deadline, not production configuration.
        self.store.config = self.store.config.model_copy(update={"jobTimeoutSeconds": .1})
        async with Broker(self.store) as broker:
            await publish_once(self.store, broker)
            delivery = await broker.queues["validation"].get(timeout=5)
            with patch("codex_agent.workers.consumer.asyncio.create_subprocess_exec", side_effect=controlled_child):
                await execute_delivery(self.store, delivery, "validation")
            self.assertTrue(delivery.processed)
        self.assertIsNotNone(children[0].returncode)
        self.assertEqual(self.store.get(job["id"])["status"], "failed")

    @unittest.skipUnless(os.environ.get("TRITON_TEST_RABBITMQ_CTL"), "private broker control required")
    async def test_broker_restart_preserves_confirmed_message(self):
        job = self.submit()
        async with Broker(self.store) as broker:
            await publish_once(self.store, broker)
        env = dict(os.environ)
        env["PATH"] = str(Path(env["ERL_ROOTDIR"]) / "bin") + ":" + env["PATH"]
        ctl = env["TRITON_TEST_RABBITMQ_CTL"]
        stopped = await asyncio.to_thread(subprocess.run, [ctl, "stop_app"], env=env, capture_output=True, timeout=30)
        self.assertEqual(stopped.returncode, 0, stopped.stderr.decode())
        try:
            with self.assertRaises(RuntimeError):
                async with Broker(self.store):
                    self.fail("stopped broker must not connect")
        finally:
            started = await asyncio.to_thread(subprocess.run, [ctl, "start_app"], env=env, capture_output=True, timeout=30)
            self.assertEqual(started.returncode, 0, started.stderr.decode())
        async with Broker(self.store) as broker:
            delivery = await broker.queues["validation"].get(timeout=5)
            self.assertEqual(json.loads(delivery.body)["job_id"], job["id"])
            await delivery.ack()

    async def test_confirm_before_database_checkpoint_can_republish_safely(self):
        job = self.submit()
        async with Broker(self.store) as broker:
            event = self.store.claim_publication()
            await broker.publish(event)
            with self.store.db.transaction() as conn:
                self.store.db.update(conn, outbox, {"lease_until": 1}, outbox.c.id == event["id"])
            await publish_once(self.store, broker)
            deliveries = [await broker.queues["validation"].get(timeout=5) for _ in range(2)]
            claims = [self.store.claim(json.loads(item.body), "validation") for item in deliveries]
            self.assertEqual(sum(c is not None for c in claims), 1)
            for delivery in deliveries:
                await delivery.ack()
        self.assertEqual(self.event(job)["attempts"], 2)

    async def test_backlog_delivers_all_100_distinct_jobs(self):
        started = time.monotonic()
        for number in range(100):
            self.submit(str(number), "memory")
        async with Broker(self.store) as broker:
            for _ in range(100):
                await publish_once(self.store, broker)
            seen = set()
            for _ in range(100):
                message = await broker.queues["memory"].get(timeout=5)
                envelope = json.loads(message.body)
                claim = self.store.claim(envelope, "memory")
                self.store.finish(claim["id"], claim["owner"], "succeeded", {"synthetic_queue_fixture": True})
                seen.add(claim["id"])
                await message.ack()
            self.assertEqual(len(seen), 100)
        print(json.dumps({"fixture": "queue-only-no-model-no-operator", "submitted": 100, "completed": len(seen),
                          "seconds": round(time.monotonic() - started, 3)}), flush=True)

    async def test_confirmed_publish_delivery_and_duplicate(self):
        job = self.submit()
        async with Broker(self.store) as broker:
            await publish_once(self.store, broker)
            self.assertEqual(self.event(job)["status"], "published")
            delivery = await broker.queues["validation"].get(timeout=5)
            claim = self.store.claim(json.loads(delivery.body), "validation")
            self.store.finish(job["id"], claim["owner"], "succeeded", {"fixture": True})
            await delivery.ack()
            await broker.publish(self.event(job))
            duplicate = await broker.queues["validation"].get(timeout=5)
            await execute_delivery(self.store, duplicate, "validation")
        self.assertEqual(len(self.store.db.rows(attempts)), 1)

    async def test_unroutable_message_is_not_marked_published(self):
        job = self.submit()
        async with Broker(self.store) as broker:
            await broker.queues["validation"].unbind(broker.exchange, routing_key="validation")
            await publish_once(self.store, broker)
        event = self.event(job)
        self.assertEqual(event["status"], "pending")
        self.assertTrue(event["last_error"])
        self.assertIsNone(event["published_ms"])

    async def test_connection_failure_keeps_outbox_for_retry(self):
        self.submit()
        async with Broker(self.store) as broker:
            await broker.connection.close()
            await publish_once(self.store, broker)
        self.assertEqual(self.store.db.rows(outbox)[0]["status"], "pending")

    async def test_poison_message_goes_to_dead_letter_without_execution(self):
        import aio_pika
        async with Broker(self.store) as broker:
            await broker.exchange.publish(aio_pika.Message(b'{"command":"not allowed"}'), routing_key="agent")
            delivery = await broker.queues["agent"].get(timeout=5)
            await execute_delivery(self.store, delivery, "agent")
            dead = await broker.channel.get_queue(broker.prefix + ".dead")
            message = await dead.get(timeout=5)
            self.assertIn(b"not allowed", message.body)
            await message.ack()
        self.assertEqual(self.store.db.rows(attempts), [])

    async def test_child_process_records_invalid_receipt_failure_before_ack(self):
        job = self.store.submit("memory", {"receipt_path": "missing.json", "sha256": "0" * 64}, "invalid")
        async with Broker(self.store) as broker:
            await publish_once(self.store, broker)
            delivery = await broker.queues["memory"].get(timeout=5)
            await execute_delivery(self.store, delivery, "memory")
            self.assertTrue(delivery.processed)
        detail = self.store.detail(job["id"])
        self.assertEqual(detail["status"], "failed", detail)
        self.assertEqual(len(detail["attempts"]), 1)
        self.assertEqual(self.store.db.rows(inbox)[0]["status"], "completed")

    async def test_unacked_delivery_redelivered_after_channel_close(self):
        job = self.submit()
        async with Broker(self.store) as broker:
            await publish_once(self.store, broker)
            first = await broker.queues["validation"].get(timeout=5)
            first_id = first.message_id
        async with Broker(self.store) as broker:
            again = await broker.queues["validation"].get(timeout=5)
            self.assertEqual(again.message_id, first_id)
            self.assertTrue(again.redelivered)
            await again.ack()
        self.assertEqual(self.store.get(job["id"])["status"], "queued")


if __name__ == "__main__":
    unittest.main()
