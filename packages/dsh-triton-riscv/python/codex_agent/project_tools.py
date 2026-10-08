"""Native, reviewable batch/project validation over existing discovery and runners."""
from __future__ import annotations
from codex_agent.runtime_config import permission_enabled, runtime_config

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Literal

from codex_agent import operator_lifecycle as lifecycle
from codex_agent.discover import discover
from codex_agent.discover_operators import discover_operators
from codex_agent.failure_diagnosis import diagnose_log
from codex_agent.run_validation import run_target
from codex_agent.execution_guard import atomic_json, approval_digest, begin_effects, guarded_execution, inspect_execution, locked_decision
from codex_agent import batch_recovery
from codex_agent.process_control import ExecutionCancelled, execution_budget


def inspect_project(root: Path, *, kind: str = "operator", offset: int = 0,
                    limit: int = 25, contains: str = "") -> dict:
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative; limit must be 1..100")
    if kind == "operator":
        result = discover_operators(root)
        entries = [{"id": item["name"], "implementation_file": item["implementation_file"],
                    "test_files": item["test_files"]} for item in result["operators"]]
    elif kind in {"project", "pytest", "lit", "build"}:
        result = discover(root)
        entries = [item for item in result["targets"] if kind == "project" or item["kind"] == kind]
    else:
        raise ValueError("unsupported inventory kind")
    entries = [item for item in entries if contains.lower() in item["id"].lower()]
    return {"status": "discovered", "kind": kind, "summary": result["summary"],
            "total_matches": len(entries), "offset": offset, "items": entries[offset:offset+limit],
            "next_offset": offset+limit if offset+limit < len(entries) else None,
            "note": "Static inventory, not evidence of tests passing."}


def _path(root: Path, job_id: str) -> Path:
    lifecycle._safe_id(job_id, "job_id")
    return lifecycle._artifact_dir(root, "jobs") / f"{job_id}.json"


def load_job(root: Path, job_id: str) -> dict:
    return lifecycle._read_json(_path(root, job_id))


def get_validation_job(root: Path, job_id: str) -> dict:
    """Read persisted progress, never dispatch or infer a test pass from a cursor."""
    job = load_job(root, job_id)
    if not batch_recovery.enabled(job):
        return job
    journal = inspect_execution(root, "job", job_id)
    recovery = {"supported": True, "observation": "local-record-only",
                "execution_state": journal["state"] if journal else "not-started",
                "same_job_id": job_id}
    if journal:
        recovery["checkpoint"] = journal.get("execution", {}).get("batch_progress")
        if journal["state"] in {"running", "unknown"}:
            try:
                batch_recovery.validate_recovery(root, job, journal)
                check_job_sources(root, job)
                recovery["next_action"] = "resume-same-job; executor still checks locks, arguments and host environment"
            except (OSError, ValueError, KeyError, TypeError) as error:
                recovery.update(next_action="host-inspection-required", reason=str(error))
    return {**job, "recovery": recovery}


def _save(root: Path, job: dict) -> None:
    atomic_json(_path(root, job["job_id"]), job)


@contextmanager
def _lock(root: Path, job_id: str):
    with _path(root, job_id).with_suffix(".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PermissionError("validation job is already executing") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _snapshot(root: Path, files: list[str]) -> dict:
    result = {}
    for name in sorted(set(files)):
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        if not path.is_file():
            raise ValueError(f"source file unavailable: {name}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _project_command(root: Path, target: dict, source_env: bool) -> tuple[str, list[str]]:
    # The model selects a discovered ID, never supplies a shell command.
    kind, path = target["kind"], target["path"]
    files = [path, *target.get("source_files", [])]
    if kind == "pytest":
        argv = ["python" if source_env else sys.executable, "-m", "pytest", "-q", path, "-s"]
    elif kind == "lit":
        argv = ["llvm-lit", "-sv", path]
    elif kind == "build":
        build = os.environ.get("TRITON_RISCV_BUILD_DIR", "")
        if not build or not Path(build).is_absolute():
            raise ValueError("host must set absolute TRITON_RISCV_BUILD_DIR for build targets")
        argv = ["cmake", "--build", str(Path(build).resolve()), "--target", target["name"]]
    else:
        raise ValueError("unsupported project target")
    if any("\x00" in arg or "\n" in arg for arg in argv):
        raise ValueError("unsafe target argument")
    return shlex.join(argv), files


def prepare_validation_job(root: Path, targets: list[str], *,
                           kind: Literal["operator-batch", "project"] = "operator-batch",
                           source_env: bool = True, timeout_seconds: int = 300) -> dict:
    if not targets or len(targets) > 20 or len(set(targets)) != len(targets):
        raise ValueError("choose 1..20 distinct targets; paginate larger regressions")
    if not 1 <= timeout_seconds <= 3600 or timeout_seconds * len(targets) > 900:
        raise ValueError("per-target timeout must be 1..3600; job total must not exceed 900 seconds")
    if kind not in {"operator-batch", "project"}:
        raise ValueError("unsupported validation job kind")
    items = []
    if kind == "operator-batch":
        for name in targets:
            plan = lifecycle.validate_operator_target(root, name, source_env=source_env,
                                                       timeout_seconds=timeout_seconds)
            items.append({"id": name, "plan": plan.model_dump(mode="json"),
                          "plan_digest": approval_digest(root, "validation", lifecycle._load_receipt(root, plan.run_id)),
                          "snapshot": lifecycle._load_receipt(root, plan.run_id)["source_snapshot"]})
    else:
        if runtime_config().remote.required:
            raise PermissionError("Project checks are local to the host; use operator-batch for guarded SSH validation, or run the host on RISC-V")
        inventory = {item["id"]: item for item in discover(root)["targets"]}
        for identifier in targets:
            if identifier not in inventory:
                raise ValueError(f"unknown project target: {identifier}")
            target = inventory[identifier]
            command, files = _project_command(root, target, source_env)
            if source_env:
                files.append("scripts/triton-riscv-env.sh")
            items.append({"id": identifier, "target": {**target, "command": command},
                          "snapshot": _snapshot(root, files),
                          "command": ("source scripts/triton-riscv-env.sh && " if source_env else "") + command})
    job = {"job_id": lifecycle._new_id("job"), "kind": kind, "status": "planned",
           "approval": {"status": "pending_approval"}, "source_env": source_env,
           "timeout_seconds": timeout_seconds, "items": items, "results": [],
           "next_action": "Call execute_validation_job to request native approval for these exact commands."}
    if kind == "operator-batch" and all(item["plan"]["execution_target"] == "remote" for item in items):
        job["batch_recovery_version"] = batch_recovery.VERSION
    _save(root, job)
    return {**job, "receipt_path": str(_path(root, job["job_id"]))}


def check_job_sources(root: Path, job: dict) -> None:
    for item in job["items"]:
        if job["kind"] == "operator-batch":
            plan = item["plan"]
            current = lifecycle.capture_source_snapshot(root, plan["implementation_file"], plan["test_files"])
        else:
            current = _snapshot(root, list(item["snapshot"]))
        if current != item["snapshot"]:
            raise PermissionError(f"source changed after planning: {item['id']}; create a new job")


@locked_decision("job")
def decide_validation_job(root: Path, job_id: str, *, approve: bool,
                          reviewer: str, note: str = "") -> dict:
    with _lock(root, job_id):
        job = load_job(root, job_id)
        if job["status"] != "planned" or job["approval"]["status"] != "pending_approval":
            raise PermissionError("job already decided or consumed")
        check_job_sources(root, job)
        job["approval"] = {"status": "approved" if approve else "rejected", "reviewer": reviewer, "note": note}
        job["approval_digest"] = approval_digest(root, "job", job)
        _save(root, job)
        return {"job_id": job_id, "status": job["approval"]["status"]}


@guarded_execution("job", "job_id")
def execute_validation_job(root: Path, job_id: str) -> dict:
    if not permission_enabled("validation"):
        raise PermissionError("host must enable TRITON_RISCV_ALLOW_VALIDATION=1")
    with _lock(root, job_id):
        job = load_job(root, job_id)
        if batch_recovery.enabled(job):
            return _execute_remote_batch(root, job)
        if job["status"] != "planned" or job["approval"]["status"] != "approved":
            raise PermissionError("job is unapproved, running or consumed; inspect results before creating a new job")
        check_job_sources(root, job)
        if job["kind"] == "project" and runtime_config().remote.required:
            raise PermissionError("remote-only host cannot execute local project checks")
        begin_effects()
        job["status"] = "running"
        _save(root, job)
        try:
            for item in job["items"]:
                try:
                    check_job_sources(root, {**job, "items": [item]})
                    if job["kind"] == "operator-batch":
                        plan = item["plan"]
                        lifecycle.decide_validation_plan(root, plan["run_id"], approve=True,
                            reviewer=job["approval"]["reviewer"], note=f"Approved as part of {job_id}")
                        result = lifecycle.validate_operator_target(root, item["id"], execute=True,
                            approved_run_id=plan["run_id"], source_env=job["source_env"],
                            timeout_seconds=job["timeout_seconds"]).model_dump(mode="json")
                    else:
                        result = asdict(run_target(item["target"], repo_root=root,
                            results_dir=lifecycle._artifact_dir(root, "jobs") / job_id,
                            source_env=job["source_env"], dry_run=False, timeout_seconds=job["timeout_seconds"]))
                        result["status"] = "passed" if result["exit_code"] == 0 else "failed"
                        result["execution_target"] = "local"
                        result["diagnosis"] = diagnose_log(Path(result["log_path"]).read_text(),
                            result["exit_code"], validation_command=result["command"])
                        check_job_sources(root, {**job, "items": [item]})
                    job["results"].append({"id": item["id"], **result})
                except (OSError, ValueError, PermissionError, RuntimeError) as error:
                    job["results"].append({"id": item["id"], "status": "error", "error": str(error)})
                _save(root, job)
            job["status"] = "passed" if all(r["status"] == "passed" for r in job["results"]) else "failed"
        except BaseException:
            job["status"] = "interrupted"
            _save(root, job)
            raise
        job["next_action"] = "Inspect per-target receipts; a failure is not automatically a repaired result. Repeated calls return this committed outcome without rerunning tests."
        _save(root, job)
        return {**job, "receipt_path": str(_path(root, job_id))}


def _execute_remote_batch(root: Path, job: dict) -> dict:
    """Called under the parent guard/locks. Resume children, never replace them."""
    if job["approval"]["status"] != "approved":
        raise PermissionError("batch must be approved")
    check_job_sources(root, job)
    checkpoint = batch_recovery.initialize(root, job)
    begin_effects()
    job.update(status="running", results=[])
    _save(root, job)
    try:
        with execution_budget(checkpoint["deadline_at"] - time.time()):
            for index, item in enumerate(job["items"]):
                check_job_sources(root, {**job, "items": [item]})
                plan = batch_recovery.check_plan(root, job, item)
                if checkpoint["items"][index]["phase"] == "pending":
                    # Intent is durable before child approval/dispatch. An absent
                    # child journal after this point stays unknown, never reruns.
                    batch_recovery.mark(index, "active")
                    lifecycle.decide_validation_plan(root, plan["run_id"], approve=True,
                        reviewer=job["approval"]["reviewer"], note=f"Approved as part of {job['job_id']}")
                result = lifecycle.validate_operator_target(root, item["id"], execute=True,
                    approved_run_id=plan["run_id"], source_env=job["source_env"],
                    timeout_seconds=job["timeout_seconds"]).model_dump(mode="json")
                # The child guard has committed receipt/log hashes before this.
                batch_recovery.mark(index, "completed", result)
                job["results"].append({"id": item["id"], **result})
                _save(root, job)
                if result.get("status") == "cancelled" or result.get("failure_stage") == "cancellation":
                    raise ExecutionCancelled("Batch child was cancelled; inspect evidence before continuing")
    except BaseException:
        job.update(status="interrupted", next_action=
            "Batch paused: preserve this job ID. Resume execute_validation_job with the SAME ID; unknown outcomes are not test failures. Cancelled, expired or inconsistent jobs require host inspection.")
        _save(root, job)
        raise
    job["status"] = "passed" if all(item["status"] == "passed" for item in job["results"]) else "failed"
    job["next_action"] = "All child receipts are committed. Repeated calls replay this outcome, not the tests."
    _save(root, job)
    return {**job, "receipt_path": str(_path(root, job["job_id"]))}


def evaluate_plugin() -> dict:
    """Offline fixture evaluation; does not open user memory or call models."""
    from codex_agent.evaluate_diagnosis import DEFAULT_CASES, evaluate_cases, load_cases
    from codex_agent.evaluate_memory_retrieval import DEFAULT_FIXTURE, evaluate_fixture, load_fixture
    return {"status": "evaluated", "scope": "offline bundled fixtures, not business success rate",
            "diagnosis": evaluate_cases(load_cases(DEFAULT_CASES)),
            "retrieval": evaluate_fixture(load_fixture(DEFAULT_FIXTURE))}
