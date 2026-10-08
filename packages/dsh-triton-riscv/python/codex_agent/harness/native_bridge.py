"""Trusted local Harness adapter. Never exported to the model as an MCP tool."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from codex_agent import operator_development as development
from codex_agent import operator_lifecycle as lifecycle
from codex_agent.memory_api import retrieve_operator_memory
from codex_agent.paths import repository_root
from codex_agent.runtime_config import permission_enabled
from codex_agent import project_tools
from codex_agent.execution_guard import atomic_json, approval_digest, cancel_path, inspect_execution, journal_path, resource_locks, file_snapshot, _files


def _artifact(root: Path, kind: str, identifier: str) -> dict[str, Any]:
    development._safe_id(identifier, "artifact_id")
    if kind == "development":
        return development._load_proposal(root, identifier)[1]
    if kind == "repair":
        return lifecycle._load_proposal(root, identifier)[1]
    if kind == "validation":
        return lifecycle._load_receipt(root, identifier)
    if kind == "job":
        return project_tools.load_job(root, identifier)
    raise ValueError("unsupported approval kind")


def _fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _expected_sources(record: dict, kind: str) -> dict:
    if kind == "development":
        return record["expected_fingerprints"]
    if kind == "repair":
        return {record["implementation_file"]: record["source_sha256"], **record["test_sha256"]}
    if kind == "validation":
        return record["source_snapshot"]
    return {name: value for item in record["items"] for name, value in item["snapshot"].items()}


def review_artifact(root: Path, kind: str, identifier: str, session_id: str) -> dict[str, Any]:
    if not permission_enabled(kind):
        raise PermissionError("Host must explicitly enable this operation in permissions")
    record = _artifact(root, kind, identifier)
    approval = record.get("approval", {}) if kind in {"validation", "job"} else record
    status = approval.get("status", "pending_approval")
    if status in {"approved", "applied"}:
        reviewer = approval.get("reviewer") if kind in {"validation", "job"} else record.get("review", {}).get("reviewer")
        if reviewer != f"native-harness:{session_id}":
            raise PermissionError("Artifact was approved by another host/session")
        if record.get("approval_digest") != approval_digest(root, kind, record):
            raise PermissionError("Approved content changed; create a new proposal/plan")
        execution = inspect_execution(root, kind, identifier)
        if execution:
            if execution["state"] != "completed":
                if kind == "job" and execution["state"] in {"running", "unknown"}:
                    from codex_agent.batch_recovery import enabled, validate_recovery
                    if enabled(record):
                        validate_recovery(root, record, execution)
                        if file_snapshot(root, _files(record, kind)) != execution.get("files_before"):
                            raise PermissionError("Sources changed during batch disconnect; host inspection required")
                        return {"fingerprint": _fingerprint(record), "status": "approved",
                                "replay": True, "recovery": True}
                remote_job = execution.get("execution", {}).get("remote_job")
                if (kind == "validation" and execution["state"] in {"running", "unknown"}
                        and isinstance(remote_job, dict) and remote_job
                        and not approval.get("execution_run_id")):
                    if file_snapshot(root, _files(record, kind)) != execution.get("files_before"):
                        raise PermissionError("Sources changed during disconnect; recovery needs host inspection")
                    # The business guard rechecks the sealed request, target and
                    # job evidence; this permits collection, never a new dispatch.
                    return {"fingerprint": _fingerprint(record), "status": "approved",
                            "replay": True, "recovery": True}
                raise PermissionError(f"Execution outcome is {execution['state']}; inspect {journal_path(root, kind, identifier)} and remote processes before authorizing a new operation")
            # The business guard still checks arguments and current source hashes.
            return {"fingerprint": _fingerprint(record), "status": "approved", "replay": True}
        if status == "applied":
            raise PermissionError("Legacy applied artifact has no replay journal; inspect existing files")
    elif status != "pending_approval":
        raise PermissionError(f"Artifact cannot be approved in state {status}")
    if kind == "validation" and (record.get("status") != "planned" or approval.get("execution_run_id")):
        raise PermissionError("Validation plan has already been consumed")
    if kind == "job":
        if record["status"] != "planned":
            raise PermissionError("Validation job has already been consumed")
    observed = file_snapshot(root, _files(record, kind))
    expected = _expected_sources(record, kind)
    changes = [{"path": name, "previous_sha256": expected.get(name), "current_sha256": value}
               for name, value in observed.items() if expected.get(name) != value]
    contract = (development._load_request(root, record["development_id"])[1]["specification"]
                if kind == "development" else None)
    fingerprint = _fingerprint({"artifact": record, "sources": observed, "contract": contract})
    details = {
        "kind": kind, "id": identifier, "operator": record.get("operator"),
        "implementation_file": record.get("implementation_file"),
        "test_files": record.get("test_files") or [record.get("test_file")],
        "command": record.get("command"), "execution_target": record.get("execution_target"),
        "timeout_seconds": record.get("timeout_seconds"), "wall_clock_budget_seconds": 900,
        "rationale": record.get("rationale"), "diff": record.get("diff"),
        "source_snapshot": observed,
        "contract": contract,
    }
    if kind == "job":
        details["items"] = [{"id": item["id"], "command": item.get("command") or item.get("plan", {}).get("command")}
                            for item in record["items"]]
        details["timeout_seconds_per_target"] = record["timeout_seconds"]
    if changes:
        details = {"action": "replan_from_latest_sources", "kind": kind, "id": identifier,
                   "operator": record.get("operator"), "changed_files": changes,
                   "message": "文件已被其他任务或编辑器修改。是否基于当前版本重新规划？这次确认不会应用旧补丁，也不会执行测试；新的执行计划或补丁仍需单独确认。"}
    reason = json.dumps(details, ensure_ascii=False, indent=2)
    if len(reason.encode()) > 120_000:
        raise ValueError("Proposal is too large for inline review; split it into smaller proposals")
    return {"fingerprint": fingerprint, "reason": reason, "status": status,
            "source_change": {"files": changes, "observed": observed} if changes else None}


def dispatch(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    if request.get("action") in {"review", "decide", "refresh", "reconcile"}:
        key = f"artifact:{request['kind']}:{request['id']}"
        # No lock is held while the native dialog waits for the user. Recheck
        # sources under the same locks when their decision comes back.
        with resource_locks(root, [key], wait_seconds=30):
            record = _artifact(root, request["kind"], request["id"])
            keys = ["file:" + str((root / name).resolve()) for name in _files(record, request["kind"])]
            with resource_locks(root, keys, wait_seconds=30):
                return _dispatch(root, request)
    return _dispatch(root, request)


def _dispatch(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action")
    if action in {"cancel", "reconcile"}:
        kind, identifier, session_id = request["kind"], request["id"], request["session_id"]
        record = _artifact(root, kind, identifier)
        review = record.get("approval", {}) if kind in {"validation", "job"} else record.get("review", {})
        if review.get("reviewer") != f"native-harness:{session_id}":
            raise PermissionError("Only the approving host/session can cancel this operation")
        if action == "reconcile":
            if request.get("confirmed_stopped") is not True or not str(request.get("note", "")).strip():
                raise PermissionError("Recovery requires explicit host confirmation that remote/local processes stopped and files were inspected, plus an audit note")
            journal = inspect_execution(root, kind, identifier)
            if not journal or journal["state"] not in {"running", "unknown"}:
                raise PermissionError("Only an unresolved execution can be reconciled")
            with resource_locks(root, ["file:" + str((root / name).resolve()) for name in journal["files_before"]]):
                journal.update(state="abandoned", reconciliation={"reviewer": review["reviewer"], "note": request["note"]})
                atomic_json(journal_path(root, kind, identifier), journal)
            return {"status": "abandoned", "next_action": "Preserve this journal. Create and approve a NEW plan; the old operation can never execute again."}
        atomic_json(cancel_path(root, kind, identifier), {"session_id": session_id, "requested": True})
        from codex_agent.remote_control import request_remote_stop
        return request_remote_stop(root, kind, identifier)
    if action == "memory":
        result = retrieve_operator_memory(root, **request["query"])
        return result.model_dump(mode="json")
    if action not in {"review", "decide", "refresh"}:
        raise ValueError("unsupported host action")
    kind, identifier, session_id = request["kind"], request["id"], request["session_id"]
    development._safe_id(session_id, "session_id")
    review = review_artifact(root, kind, identifier, session_id)
    if action == "review":
        return review
    if review["fingerprint"] != request.get("fingerprint"):
        return {"status": "review_changed", "next_action": "Review the latest sources and request approval again; nothing was applied."}
    if request.get("outcome") not in {"allowed-once", "rejected"}:
        raise PermissionError("Only a settled native human approval can record a decision")
    if action == "refresh":
        if request["outcome"] != "allowed-once" or not review.get("source_change"):
            raise PermissionError("Refreshing requires explicit approval of the changed source snapshot")
        return _refresh_plan(root, kind, identifier, session_id, review["source_change"])
    if review.get("source_change"):
        return {"status": "review_changed", "next_action": "Old approval cannot authorize changed sources."}
    if review["status"] == "approved":
        if request["outcome"] != "allowed-once":
            raise PermissionError("Previously approved action was not reauthorized")
        return {"status": "approved"}
    decide = {
        "development": development.decide_operator_development_proposal,
        "repair": lifecycle.decide_repair_proposal,
        "validation": lifecycle.decide_validation_plan,
        "job": project_tools.decide_validation_job,
    }[kind]
    return decide(root, identifier, approve=request["outcome"] == "allowed-once",
                  reviewer=f"native-harness:{session_id}", note="Native Harness approval service decision")


def _refresh_plan(root: Path, kind: str, identifier: str, session_id: str, change: dict) -> dict:
    """Adopt an explicitly reviewed baseline; never rebase or reuse an old patch."""
    record = _artifact(root, kind, identifier)
    if kind == "job":
        result = project_tools.prepare_validation_job(root, [item["id"] for item in record["items"]],
            kind=record["kind"], source_env=record["source_env"], timeout_seconds=record["timeout_seconds"])
        path = project_tools._path(root, identifier)
    else:
        result = lifecycle.validate_operator_target(root, record["operator"], execute=False,
            source_env=record.get("source_env", True), timeout_seconds=record.get("timeout_seconds", 900)).model_dump(mode="json")
        if kind == "validation":
            path = lifecycle._artifact_dir(root, "receipts") / f"{identifier}.json"
        elif kind == "repair":
            path = lifecycle._load_proposal(root, identifier)[0]
        else:
            path = development._load_proposal(root, identifier)[0]
    # If re-planning fails (e.g. the other task left incomplete files), leave
    # the old proposal untouched but still source-locked. No silent fallback.
    if kind in {"validation", "job"}:
        record.setdefault("approval", {})["status"] = "superseded"
    else:
        record["status"] = "superseded"
    record["superseded_by"] = result.get("run_id") or result.get("job_id")
    record["source_refresh"] = {"reviewer": f"native-harness:{session_id}", **change}
    atomic_json(path, record)
    next_action = "The user accepted the new baseline only. Approve and run this NEW validation plan before diagnosing/re-proposing any repair. The old patch was NOT applied."
    # Native MCP validates the intercepted return against the original tool's
    # declared result schema, even though its old execution body was skipped.
    if kind == "development":
        tool_result = development.ApplyDevelopmentResult(
            proposal_id=identifier, development_id=record["development_id"], operator=record["operator"],
            status="replanned", message=next_action, followup_plan=result).model_dump(mode="json")
    elif kind == "repair":
        tool_result = lifecycle.ApplyRepairResult(
            proposal_id=identifier, operator=record["operator"], implementation_file=record["implementation_file"],
            status="replanned", message=next_action, followup_plan=result).model_dump(mode="json")
    else:
        tool_result = result
    return {"status": "replanned", "previous_id": identifier, "plan": result,
            "next_action": next_action, "tool_result": tool_result}


def main() -> int:
    try:
        raw = sys.stdin.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("host request exceeds limit")
        result = dispatch(repository_root(), json.loads(raw))
    except (ValueError, PermissionError, KeyError, FileNotFoundError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
