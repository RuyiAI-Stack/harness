"""Explicit operator maintenance; never reruns tests or unlocks unknown jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from codex_agent.execution_guard import (
    ROOT, approval_digest, artifact, atomic_json, cancel_path, file_snapshot,
    inspect_execution, journal_path, resource_locks,
)
from codex_agent.remote_executor import RemoteValidationConfig, _job_rpc


def _operation(root, run_id):
    plan = artifact(root, "validation", run_id)
    if plan.get("status") != "planned":
        raise ValueError("use the original approved plan ID, not a result ID")
    journal = inspect_execution(root, "validation", run_id)
    if journal and (journal.get("kind") != "validation" or journal.get("id") != run_id):
        raise ValueError("execution journal identity mismatch")
    return plan, journal


def _target(job):
    configured = RemoteValidationConfig.from_env()
    if (configured is None or job.get("host") != configured.host or
            job.get("repository") != configured.repository):
        raise PermissionError("host configuration differs from the recorded remote task")
    return configured


def validation_status(root: Path, run_id: str, *, remote: bool = False) -> dict:
    """Report facts separately: a finished command is not automatically a passed test."""
    root = root.resolve()
    plan, journal = _operation(root, run_id)
    value = {"run_id": run_id, "operator": plan["operator"],
             "approval": plan.get("approval", {}).get("status", "pending"),
             "execution_state": journal["state"] if journal else "not-started",
             "test_status": None, "remote_state": None, "observation": "local-record-only"}
    if not journal:
        return value
    job = journal.get("execution", {}).get("remote_job")
    if journal["state"] == "completed":
        _verified_local_evidence(root, run_id, plan, journal, require_log=False)
        value["test_status"] = journal["result"]["status"]
        value["result_run_id"] = journal["result"]["run_id"]
    if job:
        value["cancellation_request"] = None
        cancellation_path = journal_path(root, "validation", run_id).with_suffix(".remote-cancel.json")
        if cancellation_path.is_file():
            cancellation = json.loads(cancellation_path.read_text())
            if (not isinstance(cancellation, dict) or cancellation.get("run_id") != run_id
                    or cancellation.get("job_id") != job["job_id"]):
                raise ValueError("cancellation evidence identity mismatch")
            value["cancellation_request"] = cancellation
        value.update(job_id=job["job_id"], remote_state=job.get("observed_state"),
                     observed_at=job.get("observed_at"))
        if job.get("terminal"):
            value["remote_state"] = "completed"
            value["observed_at"] = job["terminal"].get("finished_at", job.get("observed_at"))
            value["admission"] = job["terminal"].get("admission")
        if remote:
            config = _target(job)
            try:
                state = _job_rpc(config, job, "inspect")
                value.update(remote_state=state["state"], observation="live-remote", observed_at=time.time())
                value["cancellation"] = state.get("cancellation")
                if state["state"] == "completed" and journal["state"] != "completed":
                    value["next_action"] = ("remote-stop-confirmed; preserve evidence and ask host to reconcile the cancelled operation"
                        if state.get("cancellation", {}).get("confirmed") else
                        "remote-completed after local cancellation; preserve original result and ask host to reconcile"
                        if cancel_path(root, "validation", run_id).exists() else
                        "retrieve-original-approved-run; local receipt is not committed")
            except RuntimeError as error:
                value.update(remote_state="unknown", observation="remote-query-failed", detail=str(error))
    if journal["state"] in {"running", "unknown"}:
        if cancel_path(root, "validation", run_id).exists():
            value.setdefault("next_action", "cancel-requested; inspect remote evidence and ask host to reconcile; do not retry execution")
        else:
            value.setdefault("next_action", "recover-same-approved-run; never create a duplicate execution")
    return value


def _verified_local_evidence(root, run_id, plan, journal, *, require_log=True):
    if journal.get("state") != "completed":
        raise PermissionError("execution is not locally committed; preserve remote evidence")
    if plan.get("approval_digest") != approval_digest(root, "validation", plan):
        raise PermissionError("approved plan seal changed")
    result = journal.get("result", {})
    if plan.get("approval", {}).get("execution_run_id") != result.get("run_id") or not result.get("run_id"):
        raise PermissionError("plan and committed result are not linked")
    files = journal.get("result_files", {})
    required = [result.get("receipt_path")]
    if require_log or result.get("log_path") or result.get("status") == "passed":
        required.append(result.get("log_path"))
    if not all(required):
        raise PermissionError("receipt or log missing; preserve remote evidence")
    resolved = [(root / name).resolve() for name in required]
    for path in resolved:
        path.relative_to(root)
        if str(path) not in files or not files[str(path)]:
            raise PermissionError("evidence is not sealed by the committed journal")
    if files != file_snapshot(root, list(files)):
        raise PermissionError("local receipt or log changed; preserve remote evidence")
    receipt = json.loads(resolved[0].read_text())
    if receipt.get("approved_run_id") != run_id or receipt.get("run_id") != result["run_id"]:
        raise PermissionError("receipt belongs to a different approved run")
    return hashlib.sha256(resolved[1].read_bytes()).hexdigest() if len(resolved) > 1 else None


def cleanup_collected_validation(root: Path, run_id: str) -> dict:
    """Explicit CLI action only; exact ack protocol, no age-based recursive deletion."""
    root = root.resolve()
    with resource_locks(root, [f"artifact:validation:{run_id}"]):
        plan, journal = _operation(root, run_id)
        log_digest = _verified_local_evidence(root, run_id, plan, journal or {})
        job = journal.get("execution", {}).get("remote_job")
        if (not job or job.get("log_sha256") != log_digest or
                job.get("terminal", {}).get("log_sha256") != log_digest):
            raise PermissionError("collected log and remote job do not match")
        config = _target(job)
        report = {"run_id": run_id, "job_id": job["job_id"], "log_sha256": log_digest,
                  "checked_at": time.time()}
        try:
            response = _job_rpc(config, job, "ack", log_digest)
            if response.get("state") != "removed" or response.get("job_id") != job["job_id"]:
                raise RuntimeError("remote cleanup did not confirm the exact job")
            report["state"] = "removed"
        except RuntimeError as error:
            # A lost ack may already have removed the stage. Never infer deletion
            # from a timeout, nor mutate the committed test result in response.
            report.update(state="unknown", detail=str(error))
        atomic_json(root / ROOT / "cleanup" / f"{job['job_id']}.json", report)
        return report


def capacity_maintenance(root: Path, run_id: str, *, release_token: str | None = None) -> dict:
    """Host CLI only: inspect by default, release only an explicitly selected snapshot."""
    root = root.resolve()
    plan, journal = _operation(root, run_id)
    if plan.get("approval_digest") != approval_digest(root, "validation", plan):
        raise PermissionError("approved plan seal changed")
    job = (journal or {}).get("execution", {}).get("remote_job")
    if not job:
        raise ValueError("no recorded remote job; cannot select a capacity reservation")
    config = _target(job)
    if release_token is not None and not re.fullmatch(r"[a-f0-9]{64}", release_token):
        raise ValueError("use the exact release token returned by capacity inspection")
    action = "capacity-inspect" if release_token is None else "capacity-release"
    report = {"run_id": run_id, "job_id": job["job_id"], "host": config.host,
              "action": action, "checked_at": time.time()}
    try:
        response = _job_rpc(config, job, action, release_token)
        if response.get("job_id") != job["job_id"] or response.get("request_digest") != job["request_digest"]:
            raise RuntimeError("capacity response belongs to a different task")
        if release_token is not None and (response.get("state") not in {"released", "already-released"} or
                                          response.get("release_token") != release_token):
            raise RuntimeError("release not confirmed for the selected reservation")
        report["capacity"] = response
        report["state"] = response["state"]
    except RuntimeError as error:
        report.update(state="unknown", detail=str(error),
                      next_action="inspect original reservation; never delete locks or rerun the test to free capacity")
    if release_token is not None:
        # Separate maintenance evidence, never replace the validation journal/receipt.
        atomic_json(root / ROOT / "capacity-release" / f"{release_token}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--remote", action="store_true", help="query existing job only; never start it")
    mode.add_argument("--cleanup-collected", action="store_true", help="ack only sealed, locally committed evidence")
    mode.add_argument("--capacity", action="store_true", help="read-only reservation and shutdown evidence inspection")
    mode.add_argument("--release-capacity", metavar="TOKEN", help="explicitly release the exact inspected reservation")
    args = parser.parse_args()
    result = (capacity_maintenance(args.repo, args.run_id, release_token=args.release_capacity)
              if args.capacity or args.release_capacity else
              cleanup_collected_validation(args.repo, args.run_id) if args.cleanup_collected else
              validation_status(args.repo, args.run_id, remote=args.remote))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("state") == "unknown" else 0


if __name__ == "__main__":
    raise SystemExit(main())
