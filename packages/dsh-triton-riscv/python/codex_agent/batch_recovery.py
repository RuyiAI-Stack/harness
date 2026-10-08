"""Durable ordered batch cursor; child execution remains owned by the validation guard."""
import time

from codex_agent.execution_guard import (
    approval_digest, cancel_path, digest, file_snapshot, inspect_execution,
    execution_details, record_execution_detail,
)

VERSION = 1


def enabled(job):
    return job.get("kind") == "operator-batch" and job.get("batch_recovery_version") == VERSION


def check_plan(root, job, item):
    from codex_agent.operator_lifecycle import _load_receipt
    plan = _load_receipt(root, item["plan"]["run_id"])
    if (plan.get("run_id") != item["plan"]["run_id"] or plan.get("operator") != item["id"]
            or plan.get("execution_target") != "remote" or plan.get("status") != "planned"
            or approval_digest(root, "validation", plan) != item.get("plan_digest")):
        raise PermissionError("Batch child plan changed or belongs to another target; preserve evidence")
    # Lifecycle plans intentionally omit approval until the first host decision.
    approval = plan.setdefault("approval", {"status": "pending_approval"})
    if approval.get("status") == "approved":
        if (approval.get("reviewer") != job["approval"]["reviewer"]
                or approval.get("note") != f"Approved as part of {job['job_id']}"
                or plan.get("approval_digest") != item["plan_digest"]):
            raise PermissionError("Batch child approval is not owned by this batch")
    elif approval.get("status") != "pending_approval":
        raise PermissionError("Batch child approval was rejected or is invalid")
    return plan


def validate_recovery(root, job, journal):
    """Read-only preflight of ALL children before any item may execute or collect."""
    if not enabled(job) or journal.get("kind") != "job" or journal.get("id") != job["job_id"]:
        raise PermissionError("This batch has no supported recovery protocol")
    if cancel_path(root, "job", job["job_id"]).exists():
        raise PermissionError("Batch was cancelled; host reconciliation is required, not automatic resume")
    checkpoint = journal.get("execution", {}).get("batch_progress")
    if checkpoint is None:
        if journal.get("effects_started"):
            raise PermissionError("Batch progress is missing after effects began; do not restart")
        checkpoint = {"version": VERSION, "job_id": job["job_id"],
                      "deadline_at": journal["started_at"] + 900,
                      "items": [{"id": item["id"], "run_id": item["plan"]["run_id"], "phase": "pending"}
                                for item in job["items"]]}
    if (checkpoint.get("version") != VERSION or checkpoint.get("job_id") != job["job_id"]
            or checkpoint.get("deadline_at") != journal["started_at"] + 900
            or len(checkpoint.get("items", [])) != len(job["items"])):
        raise PermissionError("Batch checkpoint identity, size or original deadline changed")
    if time.time() >= checkpoint["deadline_at"]:
        raise PermissionError("Original batch wall-clock budget expired; inspect existing jobs, do not restart")
    tail = False
    for item, entry in zip(job["items"], checkpoint["items"]):
        identifier = item["plan"]["run_id"]
        if entry.get("id") != item["id"] or entry.get("run_id") != identifier:
            raise PermissionError("Batch checkpoint child identity mismatch")
        phase = entry.get("phase")
        if phase not in {"pending", "active", "completed"} or (tail and phase != "pending"):
            raise PermissionError("Batch checkpoint order is inconsistent")
        tail = tail or phase != "completed"
        plan = check_plan(root, job, item)
        child = inspect_execution(root, "validation", identifier)
        if cancel_path(root, "validation", identifier).exists():
            raise PermissionError("Batch child was cancelled; host inspection required")
        if phase == "pending":
            if child is not None or plan["approval"]["status"] != "pending_approval":
                raise PermissionError("Pending batch item has execution/approval evidence; do not dispatch")
            continue
        if (not child or child.get("kind") != "validation" or child.get("id") != identifier
                or plan["approval"]["status"] != "approved"):
            raise PermissionError("Active batch child has no matching durable journal; do not guess or restart")
        if child["state"] == "completed":
            result = child.get("result", {})
            if result.get("status") == "cancelled" or result.get("failure_stage") == "cancellation":
                raise PermissionError("Batch child cancellation requires host inspection; do not continue pending items")
            if (plan["approval"].get("execution_run_id") != result.get("run_id") or not result.get("run_id")
                    or result.get("operator") != item["id"] or result.get("execution_target") != "remote"
                    or not result.get("receipt_path")):
                raise PermissionError("Batch child result does not match its approved plan")
            paths = child.get("result_files", {})
            receipt_path = str((root / result["receipt_path"]).resolve())
            if (not paths or not paths.get(receipt_path)
                    or paths != file_snapshot(root, list(paths))):
                raise PermissionError("Batch child evidence changed or is missing; no later items executed")
            if phase == "completed" and entry.get("result_digest") != digest(result):
                raise PermissionError("Batch completed-result checkpoint changed")
        elif (phase != "active" or child["state"] not in {"running", "unknown"}
              or not child.get("execution", {}).get("remote_job")
              or plan["approval"].get("execution_run_id")):
            raise PermissionError("Batch child outcome has no recoverable remote handle; host inspection required")
    return checkpoint


def initialize(root, job):
    journal = inspect_execution(root, "job", job["job_id"])
    checkpoint = validate_recovery(root, job, journal)
    record_execution_detail(batch_progress=checkpoint)
    return checkpoint


def mark(index, phase, result=None):
    checkpoint = execution_details()["batch_progress"]
    entry = checkpoint["items"][index]
    entry["phase"] = phase
    if result is not None:
        entry["result_digest"] = digest(result)
    record_execution_detail(batch_progress=checkpoint)


def owns_child_claim(job, journal, owner):
    """Only the active, verified child may hand its unfinished file claim back."""
    if not enabled(job) or owner.get("kind") != "validation":
        return False
    entries = journal.get("execution", {}).get("batch_progress", {}).get("items", [])
    return any(entry.get("run_id") == owner.get("id") and entry.get("phase") == "active" for entry in entries)
