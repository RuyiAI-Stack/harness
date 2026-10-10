"""Reuse domain services; do not grant capabilities just because work arrived via AMQP."""
from dataclasses import replace
import hashlib
import json

from codex_agent.artifacts import ArtifactStore
from codex_agent.tasks.store import JobStore, LeaseLost, encode


def execute_job(root, identifier, owner):
    jobs = JobStore(root)
    with jobs.db.transaction() as conn:
        job = jobs.require_owner(conn, identifier, owner)
    payload = json.loads(job["payload_json"])
    artifacts = ArtifactStore(root)
    followup = None
    try:
        if job["kind"] == "agent":
            from codex_agent.harness import HarnessAgent, HarnessSettings
            from codex_agent.platform.executor import HarnessRunExecutor
            from codex_agent.platform.store import PlatformStore
            store = PlatformStore(root)
            store.db.task_guard = lambda conn: jobs.require_owner(conn, identifier, owner)
            settings = replace(HarnessSettings.from_env(root), managed_context=True)
            agent = HarnessAgent(settings)
            executor = HarnessRunExecutor(root, store, agent, workers=0)
            jobs.begin_effects(identifier, owner)
            try:
                executor._execute(payload["run_id"])
            finally:
                executor.close()
            run = store.get_run(payload["run_id"])
            result = run["result"]
            status = "succeeded" if run["status"] == "completed" else "failed"
        elif job["kind"] == "validation":
            from codex_agent.execution_guard import artifact, approval_digest
            from codex_agent.operator_lifecycle import validate_operator_target
            from codex_agent.runtime_config import remote_execution_identity
            from codex_agent.validation_evidence import capture_source_snapshot
            plan = artifact(root, "validation", payload["approved_run_id"])
            if (approval_digest(root, "validation", plan) != payload["approval_digest"]
                    or remote_execution_identity() != payload["remote_identity"]
                    or plan.get("approval", {}).get("status") != "approved"
                    or capture_source_snapshot(root, plan["implementation_file"], plan["test_files"]) != plan.get("source_snapshot")):
                raise PermissionError("queued validation input or target changed; request new approval")
            jobs.begin_effects(identifier, owner)
            from codex_agent.tasks.context import defer_memory
            from codex_agent.validation_evidence import audit_validation_receipt
            token = defer_memory.set(True)
            try:
                value = validate_operator_target(root, payload["operator"], execute=True,
                    approved_run_id=payload["approved_run_id"], source_env=payload["source_env"],
                    timeout_seconds=payload["timeout_seconds"])
            finally:
                defer_memory.reset(token)
            result = value.model_dump()
            status = "succeeded" if result["status"] == "passed" else "failed"
            receipt = (root / result["receipt_path"]).resolve()
            receipt.relative_to(root)
            audited = audit_validation_receipt(root, result)
            if audited.verdict in {"verified-passed", "verified-failed"}:
                followup = {"receipt_path": receipt.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
            artifacts.put(jobs, identifier, owner, "receipt.json", receipt.read_bytes())
            if result.get("log_path"):
                log = (root / result["log_path"]).resolve()
                log.relative_to(root)
                if log.is_file() and log.stat().st_size <= 16 * 1024 * 1024:
                    artifacts.put(jobs, identifier, owner, "validation.log", log.read_bytes())
        else:
            from codex_agent.diagnostic_memory import remember_validation
            from codex_agent.validation_evidence import audit_validation_receipt
            path = (root / payload["receipt_path"]).resolve()
            path.relative_to(root / "agent-results/operator-lifecycle/receipts")
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("receipt too large")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != payload["sha256"]:
                raise ValueError("receipt changed since submission")
            receipt = json.loads(raw)
            audited = audit_validation_receipt(root, receipt)
            if audited.verdict not in {"verified-passed", "verified-failed"}:
                raise ValueError("receipt is not verified execution evidence")
            jobs.begin_effects(identifier, owner)
            result = remember_validation(root, receipt)
            if result.get("status") == "unavailable":
                result = {"status": "unavailable", "reason": "memory storage unavailable; original validation unchanged"}
            status = "succeeded" if result["status"] in {"recorded", "deduplicated"} else "failed"
        artifacts.put(jobs, identifier, owner, "result.json", encode(result).encode())
        jobs.finish(identifier, owner, status, result, followup=followup)
    except LeaseLost:
        raise
    except Exception as error:
        # Never return an arbitrary SDK exception which could contain a credential URL.
        result = {"error_type": type(error).__name__, "reason": "Worker failed; inspect task artifacts and evidence"}
        if isinstance(error, PermissionError):
            result["reason"] = "Validation inputs or authorization changed; review a new plan. Unknown execution still requires reconciliation."
        current = jobs.get(identifier)
        status = "needs-reconciliation" if current["effects_started"] and job["kind"] != "memory" else "failed"
        jobs.finish(identifier, owner, status, result)
