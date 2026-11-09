"""Vitest scenario driver using the existing workflow fixture, never live models."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import json
import sys
from threading import Barrier, Lock
from unittest.mock import patch

from fastapi.testclient import TestClient
from dsh_multi_agent.usage import RunUsage

from codex_agent import operator_development as development
from codex_agent.artifacts import ArtifactStore
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.delegation import Plans
from codex_agent.orchestration.lifecycle import WorkflowLifecycle
from codex_agent.orchestration.roles import operator_paths
from codex_agent.orchestration.runtime import RoleAgent
from codex_agent.storage.schema import runs
from codex_agent.tasks.schema import artifacts, jobs, outbox
from codex_agent.tasks.store import Conflict, JobStore, LeaseLost
from codex_agent.tests.test_operator_development import valid_spec
from codex_agent.tests.test_orchestration_integration import WorkflowIntegrationTests
from codex_agent.workers.handlers import execute_job


REQUEST = {
    "reason": "Inspect independent source and acceptance boundaries before implementation",
    "tasks": [{"goal": "Inspect the elementwise square contract and dtype boundaries"},
              {"goal": "Inspect independent tail masking and acceptance test boundaries"}],
    "context": "Synthetic read-only research; no hardware or model calls are permitted.",
}


def sdk_events(input_tokens, output_tokens, reasoning_tokens):
    usage = {"inputTokens": input_tokens, "outputTokens": output_tokens,
             "cacheReadTokens": 0, "cacheWriteTokens": 0, "reasoningTokens": reasoning_tokens,
             "totalTokens": input_tokens + output_tokens}
    return [
        {"seq": 1, "type": "step/start", "data": {"turn": 1, "step": 1}},
        {"seq": 2, "type": "assistant/message", "data": {"turn": 1, "step": 1, "usage": usage}},
        {"seq": 3, "type": "step/end", "data": {"turn": 1, "step": 1}},
    ]


class Scenario:
    def __init__(self, stack, options):
        self.options = options
        self.f = f = WorkflowIntegrationTests(
            "test_interactive_text_extracts_spec_and_publishes_without_writes")
        stack.callback(f.doCleanups)
        f.setUp()
        f.request["auto_publish"] = True
        f.backend.decision = "request_semantics"
        f.backend.change = self.reply
        settings = HarnessSettings.from_env(f.root, {
            "ISRC_API_KEY": "synthetic-not-a-real-key", "DSH_MODEL": "fixture-model",
            "ISRC_BASE_URL": "http://127.0.0.1:1/v1",
        })
        self.agent = RoleAgent(backend=f.backend)
        self.agent.settings = settings
        if options.get("usage"):
            original_run = f.backend.run
            def run(*args, **kwargs):
                events = sdk_events(100, 20, 5)
                for event in events:
                    kwargs["on_event"](event)
                return replace(original_run(*args, **kwargs), events=events)
            f.backend.run = run
        self.clones, self.executions, self.expected_results = [], [], []
        self.cancel_job_id = None
        self.lock = Lock()
        self.real_execute = Plans.execute
        stack.enter_context(patch.object(Plans, "execute", autospec=True, side_effect=self.execute_plan))
        for target in ("codex_agent.harness.replica.run_replica",
                       "codex_agent.orchestration.runtime.ProposalHarnessBackend.run",
                       "codex_agent.operator_lifecycle.run_operator"):
            stack.enter_context(patch(target, side_effect=AssertionError("Live execution forbidden")))
        self.publication = {"checks": [], "preparations": []}
        if options.get("publication_probe"):
            self.probe_publication(stack)

    def probe_publication(self, stack):
        original_execute = WorkflowLifecycle.execute
        original_prepare = development.prepare_operator_development
        context = {}

        def publish(lifecycle, task_id, action, *, expected_revision):
            if action == "publish" and lifecycle.publication_owner:
                job_id, owner = lifecycle.publication_owner
                context.update(job_id=job_id, owner=owner, task_id=task_id)
                task = self.f.store.get(task_id)
                counts = self.counts()

                def rejected(publication_owner):
                    try:
                        original_execute(WorkflowLifecycle(self.f.root, publication_owner=publication_owner),
                                         task_id, "publish", expected_revision=expected_revision)
                    except (Conflict, LeaseLost) as error:
                        return type(error).__name__
                    return None

                # Probe before the legitimate publisher takes its workflow lock.
                response = self.f.client.post("/api/operator-workflows/" + task_id + "/publish",
                                              json={"expected_revision": expected_revision})
                wrong_owner = rejected((job_id, "not-the-publication-owner"))
                wrong_job = rejected(("another-job", owner))
                self.publication["checks"].append({
                    "job_id": job_id, "job_status": self.f.jobs.get(job_id)["status"],
                    "step_statuses": [step["status"] for step in task["steps"]],
                    "step_job_statuses": [self.f.jobs.get(step["job_id"])["status"] for step in task["steps"]],
                    "manual_status": response.status_code, "manual_body": response.json(),
                    "wrong_owner": wrong_owner, "wrong_job": wrong_job,
                    "task_unchanged": self.f.store.get(task_id) == task,
                    "counts_unchanged": self.counts() == counts,
                })
            return original_execute(lifecycle, task_id, action, expected_revision=expected_revision)

        def prepare(*args, **kwargs):
            before = self.f.jobs.get(context["job_id"])
            heartbeat = self.f.jobs.heartbeat(context["job_id"], context["owner"])
            after = self.f.jobs.get(context["job_id"])
            task = self.f.store.get(context["task_id"])
            self.publication["preparations"].append({
                "job_id": before["id"], "before_status": before["status"], "after_status": after["status"],
                "heartbeat_cancelled": heartbeat, "lease_renewed": after["lease_until"] >= before["lease_until"],
                "workflow_state": task["state"], "operation": task["result"]["lifecycle"]["operation"],
            })
            return original_prepare(*args, **kwargs)

        stack.enter_context(patch.object(WorkflowLifecycle, "execute", autospec=True, side_effect=publish))
        stack.enter_context(patch.object(development, "prepare_operator_development", side_effect=prepare))

    def reply(self, request, result):
        self.f.reply(request, result)
        if request["role"] == "main":
            mode = self.options.get("request", "present")
            if mode != "absent":
                result["parallel_request"] = None if mode == "null" else REQUEST
        if request["role"] == "semantic":
            result["specification"] = valid_spec()

    def clone(self, task, stopped):
        with self.lock:
            self.clones.append(task["task_id"])
        failure = self.options.get("failure")
        if failure == "all" or (failure == "partial" and task["task_id"] == "child-2"):
            raise RuntimeError("private-provider-error-must-not-leak")
        response = "synthetic-clone-" + task["task_id"] + ": read-only findings, not a test pass."
        if self.options.get("long_result"):
            response += "x" * 9000 + "raw-tail-" + task["task_id"]
        result = {"status": "completed", "response": response,
                  "usage": {"input_tokens": 11, "output_tokens": 7},
                  "usage_complete": True, "events": [{"type": "synthetic_raw_event"}]}
        if self.options.get("usage") == "sdk":
            meter = RunUsage("fixture-" + task["task_id"], provider="fixture", model="fixture-model")
            result["events"] = sdk_events(11, 7, 3)
            for event in result["events"]:
                meter.observe(event)
            result["usage"] = meter.finish("completed")
        with self.lock:
            self.expected_results.append({**result, "task_id": task["task_id"]})
        return result

    def execute_plan(self, plans, identifier, session_id, settings, runner=None, *, stop=None):
        self.executions.append({"plan_id": identifier, "session_id": session_id,
                                "has_stop": callable(stop)})
        if self.cancel_job_id:
            self.f.jobs.cancel(self.cancel_job_id)
        if self.options.get("failure") == "execution":
            raise OSError("private-provider-error-must-not-leak")
        return self.real_execute(plans, identifier, session_id, settings, self.clone, stop=stop)

    def execute(self, job):
        claimed = self.f.claim(job)
        if claimed is None:
            raise RuntimeError("Fixture attempted to execute an unclaimable job")
        execute_job(self.f.root, job["id"], claimed["owner"], proposal_agent=self.agent)
        return self.f.jobs.get(job["id"])

    def start(self, key="one"):
        job = self.f.submit(key=key)
        self.execute(job)
        return job, self.f.lifecycle.view(job["payload"]["workflow_id"])

    def counts(self):
        return {name: len(self.f.jobs.db.rows(table))
                for name, table in (("jobs", jobs), ("outbox", outbox), ("runs", runs))}

    def snapshot(self, task_id):
        view = self.f.lifecycle.view(task_id)
        return {"view": view, "counts": self.counts(),
                "roles": [c["role"] for c in self.f.backend.calls],
                "phases": [c["phase"] for c in self.f.backend.calls],
                "clones": sorted(self.clones), "executions": list(self.executions),
                "files_exist": [(self.f.root / p).exists() for p in operator_paths("square_new")]}

    @staticmethod
    def body(view, approved=True):
        plan = view.get("parallel_plan")
        if plan is None:
            raise RuntimeError("No pending parallel plan: " + json.dumps(view))
        return {"expected_revision": view["revision"], "plan_id": plan["plan_id"],
                "digest": plan["digest"], "approved": approved}

    def decide(self, task_id, body):
        # Independent clients exercise host locking under simultaneous HTTP requests.
        with TestClient(self.f.client.app) as client:
            response = client.post("/api/operator-workflows/" + task_id + "/parallel-decision", json=body)
        return {"status": response.status_code, "body": response.json()}

    def audits(self, task_id):
        rows = {row["id"]: row for row in self.f.jobs.db.rows(artifacts)}
        return [json.loads(ArtifactStore(self.f.root).read(rows[step["result"]["artifact_id"]]))
                for step in self.f.store.get(task_id)["steps"]]

    def usage(self, task_id):
        response = self.f.client.get("/api/operator-workflows/" + task_id + "/usage")
        if response.status_code != 200:
            raise RuntimeError("Workflow usage endpoint failed: " + response.text)
        return response.json()

    def run(self):
        original, pending = self.start()
        task_id = pending["id"]
        result = {"initial": self.snapshot(task_id), "original_id": original["id"],
                  "original_status": self.f.jobs.get(original["id"])["status"]}
        if self.options.get("publication_probe"):
            result["publication"] = self.publication
        if self.options.get("usage"):
            result["initial_usage"] = self.usage(task_id)
        if pending.get("parallel_plan"):
            result["initial_raw_scope"] = Plans(self.f.root).load(
                pending["parallel_plan"]["plan_id"], "workflow-" + task_id)["session_id"]
        scenario = self.options.get("scenario", "resume")
        if scenario in {"default", "pause"}:
            result["plan_files"] = len(list((self.f.root / "agent-results/parallel-plans").glob("*.json")))
            result["audits"] = self.audits(task_id)
            return result
        body = self.body(pending, self.options.get("approved", True))
        if scenario == "invalid_decision":
            _, foreign = self.start("foreign-workflow")
            foreign_body = self.body(foreign)
            before = self.snapshot(task_id)
            result["rejected"] = [self.decide(task_id, changed) for changed in (
                {**body, "expected_revision": body["expected_revision"] - 1},
                {**body, "digest": "0" * 64},
                {**body, "plan_id": foreign_body["plan_id"], "digest": foreign_body["digest"]})]
            result.update(before=before, after=self.snapshot(task_id),
                          foreign=self.f.lifecycle.view(foreign["id"]),
                          foreign_raw_scope=Plans(self.f.root).load(
                              foreign_body["plan_id"], "workflow-" + foreign["id"])["session_id"])
            return result
        if scenario == "source_before_decision":
            self.change_source()
            result["decision"] = self.decide(task_id, body)
            result["after"] = self.snapshot(task_id)
            return result
        if scenario == "decision_rollback":
            plans = Plans(self.f.root)
            saved = plans.path(body["plan_id"]).read_bytes()
            with patch.object(JobStore, "submit", autospec=True,
                              side_effect=Conflict("synthetic continuation submit failure")) as submit:
                response = self.decide(task_id, body)
            result["rollback"] = {
                "response": response, "submit_calls": submit.call_count,
                "after": self.snapshot(task_id),
                "raw_plan": plans.load(body["plan_id"], "workflow-" + task_id),
                "plan_bytes_unchanged": plans.path(body["plan_id"]).read_bytes() == saved,
            }
        if scenario == "concurrent":
            barrier = Barrier(4)
            def decide(_):
                barrier.wait(timeout=10)
                return self.decide(task_id, body)
            with ThreadPoolExecutor(max_workers=4) as pool:
                decisions = list(pool.map(decide, range(4)))
        else:
            decisions = [self.decide(task_id, body), self.decide(task_id, body)]
        result["decisions"] = decisions
        result["queued"] = self.snapshot(task_id)
        if scenario == "decision_rollback":
            result["queued_raw_plan"] = Plans(self.f.root).load(body["plan_id"], "workflow-" + task_id)
        if self.options.get("usage"):
            result["queued_usage"] = self.usage(task_id)
        if any(d["status"] != 200 for d in decisions):
            return result
        continuation = self.f.jobs.get(decisions[0]["body"]["job"]["id"])
        result["continuation"] = continuation
        if scenario == "source_after_decision":
            self.change_source()
        if scenario == "model_changed":
            self.agent.settings = replace(self.agent.settings, **self.options["settings_change"])
        if scenario == "cancel":
            result["execution_error"] = None
            if self.options["at"] == "queued":
                self.f.jobs.cancel(continuation["id"])
                result["cancelled_claim"] = self.f.claim(continuation)
            else:
                self.cancel_job_id = continuation["id"]
                try:
                    self.execute(continuation)
                except LeaseLost as error:
                    result["execution_error"] = type(error).__name__
        else:
            result["execution"] = self.execute(continuation)
        result["final"] = self.snapshot(task_id)
        if self.options.get("usage"):
            result["usage"] = self.usage(task_id)
        result["calls"] = self.f.backend.calls
        result["audits"] = self.audits(task_id)
        plan = Plans(self.f.root)
        path = plan.path(body["plan_id"])
        saved = path.read_bytes()
        result["raw_plan"] = plan.load(body["plan_id"], "workflow-" + task_id)
        result["expected_results"] = sorted(self.expected_results, key=lambda r: r["task_id"])
        result["replay"] = self.decide(task_id, body)
        result["opposite_decision"] = self.decide(task_id, {**body, "approved": not body["approved"]})
        result["redelivery_claims"] = [self.f.claim(original), self.f.claim(continuation)]
        result["after_replay"] = self.snapshot(task_id)
        if self.options.get("usage"):
            result["usage_after_replay"] = self.usage(task_id)
        result["raw_preserved_after_replay"] = path.read_bytes() == saved
        result["old_job_status_after_resume"] = self.f.jobs.get(original["id"])["status"]
        if scenario == "source_after_decision":
            result["source_content"] = (self.f.root / operator_paths("square_new")[0]).read_text()
        return result

    def change_source(self):
        path = self.f.root / operator_paths("square_new")[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# external source edit while approval was pending\n")


if __name__ == "__main__":
    with ExitStack() as stack:
        result = Scenario(stack, json.load(sys.stdin)).run()
    print(json.dumps(result))
