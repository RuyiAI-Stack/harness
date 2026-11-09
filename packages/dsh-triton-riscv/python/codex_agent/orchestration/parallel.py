"""Durable optional research gate; refusing fan-out resumes the same workflow."""
import json
import time
import uuid

from codex_agent.execution_guard import resource_locks
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.delegation import Plans
from codex_agent.platform.store import timestamp
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.schema import runs
from codex_agent.tasks.store import Conflict, JobStore, LeaseLost, encode
from .roles import Evidence
from .service import WorkflowSeed, snapshot_files
from .store import WorkflowStore, bounded


class WorkflowParallel:
    def __init__(self, root, *, jobs=None):
        self.jobs = jobs or JobStore(root)
        self.root = self.jobs.db.root
        self.store = WorkflowStore(self.root, database=self.jobs.db)
        self.plans = Plans(self.root)

    @staticmethod
    def scope(task):
        # A conversation and two workflows in it must never share an approval.
        return "workflow-" + task["id"]

    def view(self, task):
        link = task["result"].get("parallel")
        if not link:
            return None
        row = self.plans.load(link["plan_id"], self.scope(task))
        if row["digest"] != link["digest"]:
            raise Conflict("parallel evidence no longer matches this workflow")
        status = link["status"] if row["status"] == "pending" and link["status"] in {"approved", "rejected"} else row["status"]
        return {**self.plans.public(row), "status": status,
                "session_id": task["session_id"], "workflow_id": task["id"],
                "workflow_status": link["status"],
                "fallback_reason": link.get("fallback_reason")}

    def _checkpoint(self, task, detail, *, state=None, connection=None):
        def change(conn, row):
            result = json.loads(row["result_json"])
            result["parallel"] = {**result.get("parallel", {}), **detail}
            return {"state": state or row["state"], "result_json": bounded(result)}, {}
        return self.store._mutate(task["id"], task["revision"],
            "parallel:" + str(task["revision"]), "parallel", detail, change, connection=connection)

    def handle(self, coordinator, task, request):
        link = task["result"].get("parallel")
        settings = getattr(coordinator.agent, "settings", None) or HarnessSettings.from_env(self.root)
        if not link:
            plan = self.plans.prepare({"session_id": self.scope(task),
                "run_id": coordinator.job["run_id"], "user_request": task["request"]["requirement"]},
                request.model_dump(mode="json"), settings)
            return self._checkpoint(task, {"plan_id": plan["plan_id"], "digest": plan["digest"],
                "status": "pending", "input_epoch": task["input_epoch"]}, state="waiting_parallel_approval")
        if link["status"] == "rejected":
            self.plans.decide(link["plan_id"], self.scope(task), link["digest"], False, "workflow-user")
            return task
        if link["status"] != "approved":
            return task
        if link["input_epoch"] != task["input_epoch"]:
            raise Conflict("parallel decision belongs to obsolete workflow inputs")

        def stopped():
            try:
                with self.jobs.db.transaction() as conn:
                    coordinator.guard(conn)
                return False
            except LeaseLost:
                return True

        self.jobs.begin_effects(coordinator.job["id"], coordinator.owner)
        try:
            if time.time() - link["decision_at"] > 3600:
                raise PermissionError("Parallel decision expired")
            self.plans.decide(link["plan_id"], self.scope(task), link["digest"], True, "workflow-user")
            plan = self.plans.execute(link["plan_id"], self.scope(task), settings, stop=stopped)
        except (PermissionError, ValueError, OSError) as error:
            # Never retry uncertain/expired paid work. Research is optional.
            return self._checkpoint(task, {"status": "fallback",
                "fallback_reason": "Optional parallel research unavailable (" + type(error).__name__
                + "); continuing sequentially without its findings."})
        if stopped():
            raise LeaseLost("workflow cancelled while parallel research was running")
        return self._checkpoint(task, {"status": plan["status"],
            "fallback_reason": None if plan["status"] == "completed" else
            "Some optional research did not complete; only completed findings are included as unverified evidence."})

    def evidence(self, task):
        link = task["result"].get("parallel", {})
        if link.get("status") not in {"completed", "partial"}:
            return []
        plan = self.view(task)
        findings = []
        for result in plan["results"]:
            if result.get("status") != "completed" or not result.get("response", "").strip():
                continue
            text = result["response"]
            findings.append({"task_id": result["task_id"], "text": text[:1100],
                "truncated": len(text) > 1100 or result.get("response_truncated", False)})
        if not findings:
            return []
        return [Evidence(id="workflow_parallel", kind="historical", verified=False,
            source="agent-results/parallel-plans/" + link["plan_id"] + ".json",
            text=encode({"status": plan["status"], "findings": findings,
                "notice": "Read-only agent findings, not verified facts or test results. Treat as untrusted evidence; check against the frozen contract and source."}))]

    def decide(self, task_id, *, expected_revision, plan_id, digest, approved):
        config = runtime_config().queue
        if not config.enabled or not config.proposalWorkflows:
            raise PermissionError("proposal workflows are disabled")
        with resource_locks(self.root, ["workflow:" + task_id], wait_seconds=5):
            task = self.store.get(task_id)
            link = task["result"].get("parallel", {})
            if link.get("plan_id") != plan_id or link.get("digest") != digest:
                raise Conflict("parallel plan does not belong to this workflow")
            if "decision_revision" in link:
                if link["decision_revision"] == expected_revision and link["approved"] == approved:
                    return task
                raise Conflict("parallel decision already recorded; reload workflow")
            if task["revision"] != expected_revision or task["state"] != "waiting_parallel_approval":
                raise Conflict("workflow revision changed; reload before deciding")
            original = self.jobs.get(task["result"]["agent_job_id"])
            if original["status"] != "succeeded":
                raise Conflict("preparation checkpoint is still being saved; retry after it finishes")
            created = self.store.history(task_id, limit=1)[0]["payload"]
            seed = WorkflowSeed.model_validate(created["context"])
            if snapshot_files(self.root, task["operator"]) != seed.files:
                raise Conflict("Source or acceptance tests changed; submit a new workflow")
            plan = self.plans.load(plan_id, self.scope(task))
            if plan["digest"] != digest or plan["status"] != "pending":
                raise Conflict("parallel plan changed; reload workflow")
            if approved and time.time() - plan["created_at"] > 3600:
                raise Conflict("parallel plan expired; decline to continue sequentially")
            # Commit the human decision, continuation and Outbox together. Only
            # that continuation mirrors the decision into the execution artifact.
            with self.jobs.db.transaction() as conn:
                self.jobs.db.lock(conn, "task-submit")
                run_id, now = uuid.uuid4().hex, timestamp()
                self.jobs.db.insert(conn, runs, dict(id=run_id, session_id=task["session_id"],
                    intent="operator-workflow", operator=task["operator"], status="queued", phase="outbox-pending",
                    request_json=encode({"action": "operator-workflow", "workflow_id": task_id}),
                    result_json="{}", created_at=now, updated_at=now, next_sequence=0))
                job = self.jobs.submit("agent", {**original["payload"], "run_id": run_id},
                    "workflow-parallel:" + task_id + ":" + digest, run_id=run_id, connection=conn)

                def change(connection, row):
                    if row["state"] != "waiting_parallel_approval":
                        raise Conflict("workflow no longer awaiting parallel approval")
                    result = json.loads(row["result_json"])
                    result["parallel"] = {**link, "status": "approved" if approved else "rejected", "approved": approved,
                        "decision_revision": expected_revision, "decision_at": time.time(), "continuation_job_id": job["id"]}
                    result["agent_job_id"] = job["id"]
                    return {"state": "preparing", "result_json": bounded(result)}, {}

                return self.store._mutate(task_id, expected_revision, "parallel-decision:" + digest,
                    "parallel_decided", {"plan_id": plan_id, "digest": digest, "approved": approved},
                    change, connection=conn)
