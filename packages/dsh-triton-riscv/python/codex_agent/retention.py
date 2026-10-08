"""Read-only retention preview for known, guarded remote validation artifacts.

This is not a garbage collector. Original evidence and unknown artifacts remain
protected. No report authorizes deletion; the existing explicit cleanup command
must revalidate evidence when invoked by the host operator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import stat
import time

from codex_agent.execution_guard import ROOT, digest, journal_path
from codex_agent.operator_lifecycle import ARTIFACT_ROOT
from codex_agent.remote_executor import _job_rpc
from codex_agent.remote_maintenance import _target, _verified_local_evidence

METADATA_LIMIT = 2 * 1024 * 1024
EVIDENCE_LIMIT = 16 * 1024 * 1024
JOURNAL_NAME = re.compile(r"[a-f0-9]{64}\.json")


@dataclass(frozen=True)
class RetentionPolicy:
    success_days: int = 7
    failure_days: int = 30

    def __post_init__(self):
        for value in (self.success_days, self.failure_days):
            if type(value) is not int or not 0 <= value <= 3650:
                raise ValueError("retention days must be integers in 0..3650")


def _safe_path(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    parts = path.relative_to(root).parts
    if ".." in parts:
        raise ValueError("artifact path escapes repository")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink artifact requires manual inspection")
    return path


def _regular(root, path, maximum):
    path = _safe_path(root, path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError("artifact is not a bounded regular file; preserve for inspection")
    return path


def _json(root, path):
    path = _regular(root, path, METADATA_LIMIT)
    with path.open("rb") as stream:
        content = stream.read(METADATA_LIMIT + 1)
    if len(content) > METADATA_LIMIT:
        raise ValueError("metadata grew beyond inspection budget")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value


def _timestamp(value, now):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= now:
        raise ValueError("completion time missing, invalid or in the future")
    return value


def _item(root, path, policy, now, remote_query):
    item = {"journal": str(path.relative_to(root)), "state": "protected",
            "reason": "not-inspected", "local_action": "retain", "remote_action": "retain",
            "observation": "local-only", "remote_bytes_reclaimable": None}
    try:
        journal = _json(root, path)
        kind, run_id = journal.get("kind"), journal.get("id")
        item.update(kind=kind, run_id=run_id)
        if kind != "validation":
            return {**item, "reason": "outside-validation-retention-scope"}
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("invalid original plan ID")
        if path != journal_path(root, kind, run_id):
            raise ValueError("journal filename and task identity differ")
        if journal.get("state") != "completed":
            return {**item, "reason": "execution-not-durably-completed"}
        plan = _json(root, root / ARTIFACT_ROOT / "receipts" / f"{run_id}.json")
        if plan.get("status") != "planned" or plan.get("run_id") != run_id:
            raise ValueError("not the original approved plan")
        files = journal.get("result_files")
        if not isinstance(files, dict) or not files or len(files) > 16:
            raise ValueError("missing evidence manifest")
        for name in files:
            _regular(root, name, EVIDENCE_LIMIT)
        checksum = _verified_local_evidence(root, run_id, plan, journal)
        job = journal.get("execution", {}).get("remote_job")
        if not isinstance(job, dict):
            return {**item, "reason": "no-remote-job; retain-local-evidence"}
        if (not isinstance(job.get("job_id"), str) or not re.fullmatch(r"[a-f0-9]{32}", job["job_id"]) or
                job.get("stage") != f"/tmp/triton-riscv-agent/{job['job_id']}" or
                not isinstance(job.get("request_digest"), str) or not re.fullmatch(r"[a-f0-9]{64}", job["request_digest"])):
            raise ValueError("invalid recorded remote job identity")
        terminal = job.get("terminal", {})
        if (terminal.get("state") != "completed" or terminal.get("job_id") != job["job_id"] or
                terminal.get("request_digest") != job["request_digest"] or
                terminal.get("log_sha256") != checksum or job.get("log_sha256") != checksum):
            raise ValueError("local receipt and remote terminal are not linked")
        result = journal["result"]
        receipt = _json(root, result["receipt_path"])
        if receipt.get("status") != result.get("status") or receipt.get("exit_code") != result.get("exit_code"):
            raise ValueError("receipt and journal outcomes differ")
        status = result.get("status")
        if (status not in {"passed", "failed", "cancelled"} or
                type(result.get("exit_code")) is not int or type(terminal.get("exit_code")) is not int or
                result["exit_code"] != terminal.get("exit_code") or
                (status == "passed") != (result["exit_code"] == 0)):
            raise ValueError("unknown or contradictory test outcome")
        end = max(_timestamp(journal.get("completed_at"), now), _timestamp(terminal.get("finished_at"), now))
        days = policy.success_days if status == "passed" else policy.failure_days
        expires = end + days * 86400
        item.update(operator=plan.get("operator"), test_status=status, host=job.get("host"),
                    job_id=job["job_id"], remote_stage=job["stage"], completed_at=end,
                    retain_days=days, retain_until=expires,
                    snapshot_sha256=digest({"plan": plan, "journal": journal}),
                    local_evidence=sorted(files),
                    verified_local_bytes=sum((root / name).stat().st_size for name in files))
        cleanup_path = root / ROOT / "cleanup" / f"{job['job_id']}.json"
        if cleanup_path.exists() or cleanup_path.is_symlink():
            cleanup = _json(root, cleanup_path)
            if (cleanup.get("run_id") != run_id or cleanup.get("job_id") != job["job_id"] or
                    cleanup.get("log_sha256") != checksum):
                raise ValueError("cleanup receipt identity mismatch")
            if cleanup.get("state") == "removed":
                if _timestamp(cleanup.get("checked_at"), now) < end:
                    raise ValueError("cleanup confirmation predates completion")
                return {**item, "state": "previously-cleaned", "reason": "prior-cleanup-confirmed; not-a-live-observation"}
        if now < expires:
            return {**item, "state": "retained", "reason": "retention-window-not-expired"}
        item.update(state="local-candidate", reason="local-evidence-verified; remote-check-required",
                    remote_action="review-only")
        if remote_query is None:
            return item
        response = remote_query(_target(job), job)
        item["observation"] = "live-remote-read-only"
        if response.get("job_id") != job["job_id"] or response.get("request_digest") != job["request_digest"]:
            raise ValueError("remote preview identity mismatch")
        item["remote_evidence"] = response
        if response.get("state") != "eligible":
            return {**item, "state": "protected", "remote_action": "retain",
                    "reason": "remote-not-ready: " + str(response.get("reason", "unknown"))}
        if (response.get("log_sha256") != checksum or response.get("terminal_sha256") != digest(terminal)):
            raise ValueError("remote terminal changed since local collection")
        # A preview cannot replace final checks. Never reuse a stale report as approval.
        return {**item, "state": "reviewable", "reason": "age-and-evidence-checked; explicit-cleanup-still-required",
                "next_action": "review then run existing remote_maintenance --cleanup-collected with original run_id"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        return {**item, "state": "protected", "remote_action": "retain", "reason": str(error)}


def preview_retention(root: Path, *, policy=None, remote=False, limit=100, after=None, now=None, remote_limit=10):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("repository does not exist")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("page limit must be in 1..1000")
    if type(remote_limit) is not int or not 1 <= remote_limit <= 20:
        raise ValueError("remote observation limit must be in 1..20")
    if after is not None and not JOURNAL_NAME.fullmatch(after):
        raise ValueError("invalid journal cursor")
    policy = policy or RetentionPolicy()
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now) or now <= 0:
        raise ValueError("invalid observation time")
    folder = _safe_path(root, root / ROOT / "operations")
    if folder.exists() and not folder.is_dir():
        raise ValueError("operations path is not a directory")
    paths = sorted((p for p in folder.glob("*.json") if JOURNAL_NAME.fullmatch(p.name)
                    and (after is None or p.name > after)), key=lambda p: p.name)
    selected = paths[:limit]
    calls = 0
    def query(config, job):
        nonlocal calls
        if calls >= remote_limit:
            raise RuntimeError("remote-observation-budget-exhausted; inspect smaller pages instead")
        calls += 1
        return _job_rpc(config, job, "cleanup-preview")
    items = [_item(root, p, policy, now, query if remote else None) for p in selected]
    return {"schema_version": 1, "mode": "preview-only", "created_at": now, "repo_root": str(root),
            "policy": {**asdict(policy), "scope": "residual-remote-validation-copies-only",
                       "local_originals": "retain-indefinitely", "automatic_cleanup_changed": False},
            "network_requested": bool(remote), "remote_checks_attempted": calls, "remote_check_limit": remote_limit,
            "items": items, "counts": dict(Counter(i["state"] for i in items)),
            "page": {"limit": limit, "scanned": len(selected), "has_more": len(paths) > limit,
                     "next_after": selected[-1].name if len(paths) > limit else None},
            "excluded_and_preserved": ["untracked-or-legacy-artifacts", "local-logs-and-receipts", "approvals-and-patches",
                                       "rag-and-session-databases", "capacity-and-file-locks", "other-repositories"],
            "warning": "No files deleted. Candidates are not deletion permission. Remote size is unknown, not zero."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--remote", action="store_true", help="explicit read-only checks of mature local candidates")
    parser.add_argument("--success-days", type=int, default=7)
    parser.add_argument("--failure-days", type=int, default=30)
    parser.add_argument("--limit", type=int, help="page size; defaults to 100 locally or 10 with --remote")
    parser.add_argument("--remote-limit", type=int, default=10, help="at most 20 read-only SSH checks; default 10")
    parser.add_argument("--after", help="next_after from the previous page")
    parser.add_argument("--output", type=Path, help="optional NEW report file, never overwrite an existing file")
    args = parser.parse_args()
    report = preview_retention(args.repo, policy=RetentionPolicy(args.success_days, args.failure_days),
                               remote=args.remote, limit=args.limit if args.limit is not None else (10 if args.remote else 100),
                               after=args.after, remote_limit=args.remote_limit)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
