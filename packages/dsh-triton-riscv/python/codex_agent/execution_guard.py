"""Durable, fail-closed guards for plugin-owned side effects (not an OS sandbox)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import get_type_hints
from codex_agent.process_control import execution_budget, remaining
from codex_agent.runtime_config import permission_enabled, remote_execution_identity, runtime_config


ROOT = Path("agent-results/execution-guard")
_held = threading.local()
_MUTABLE = {"status", "approval", "review", "approval_digest", "applied_at",
            "applied_fingerprints", "applied_source_sha256", "results", "next_action"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".guard-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def resource_locks(root: Path, resources, *, wait_seconds: float = 0):
    """Sorted advisory locks; callers may opt into bounded, cancellable waiting."""
    if not 0 <= wait_seconds <= 30:
        raise ValueError("lock wait must be in 0..30 seconds")
    deadline = time.monotonic() + wait_seconds
    root = root.resolve()
    folder = root / ROOT / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    if getattr(_held, "pid", None) != os.getpid():
        _held.pid, _held.keys = os.getpid(), set()
    acquired = []
    try:
        for resource in sorted(set(resources)):
            key = str(root) + "\0" + resource
            if key in _held.keys:
                continue
            handle = (folder / (digest(key) + ".lock")).open("a")
            try:
                while True:
                    remaining(900)
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise PermissionError(f"Resource busy: {resource}; waiting budget exhausted; no operation executed") from None
                        time.sleep(min(left, 0.05))
            except BaseException:
                handle.close()
                raise
            acquired.append((key, handle))
            _held.keys.add(key)
        yield
    finally:
        for key, handle in reversed(acquired):
            _held.keys.remove(key)
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def artifact(root: Path, kind: str, identifier: str) -> dict:
    from codex_agent import operator_development, operator_lifecycle, project_tools
    operator_development._safe_id(identifier, "artifact_id")
    if kind == "development":
        return operator_development._load_proposal(root, identifier)[1]
    if kind == "repair":
        return operator_lifecycle._load_proposal(root, identifier)[1]
    if kind == "validation":
        return operator_lifecycle._load_receipt(root, identifier)
    if kind == "job":
        return project_tools.load_job(root, identifier)
    raise ValueError("unsupported operation kind")


def approval_digest(root: Path, kind: str, record: dict) -> str:
    content = {key: value for key, value in record.items() if key not in _MUTABLE}
    if kind == "development":
        from codex_agent.operator_development import _load_request
        content["contract"] = _load_request(root, record["development_id"])[1]["specification"]
    return digest({"kind": kind, "content": content})


def _files(record: dict, kind: str) -> list[str]:
    if kind == "development":
        return [record[key] for key in ("implementation_file", "test_file", "task_file")]
    if kind == "repair":
        return [record["implementation_file"], *record["test_sha256"]]
    if kind == "validation":
        return [record["implementation_file"], *record["test_files"]]
    return sorted({name for item in record["items"] for name in item["snapshot"]})


def file_snapshot(root: Path, files: list[str]) -> dict:
    result = {}
    for name in files:
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def _result_files(root: Path, result: dict) -> list[str]:
    names = []
    for key in ("receipt_path", "log_path", "patch_path"):
        if result.get(key):
            path = (root / result[key]).resolve()
            path.relative_to(root)
            names.append(str(path))
    for item in result.get("results", []):
        names.extend(_result_files(root, item))
    return names


def journal_path(root: Path, kind: str, identifier: str) -> Path:
    # Never use a caller-supplied identifier as a filesystem path.
    return root.resolve() / ROOT / "operations" / (digest([kind, identifier]) + ".json")


def inspect_execution(root: Path, kind: str, identifier: str) -> dict | None:
    path = journal_path(root, kind, identifier)
    return json.loads(path.read_text()) if path.is_file() else None


def cancel_path(root: Path, kind: str, identifier: str) -> Path:
    return journal_path(root, kind, identifier).with_suffix(".cancel")


def begin_effects() -> None:
    remaining(900)
    operation = getattr(_held, "operation", None)
    if operation:
        path, journal = operation
        journal["effects_started"] = True
        atomic_json(path, journal)


def record_execution_detail(**details) -> None:
    operation = getattr(_held, "operation", None)
    if operation:
        path, journal = operation
        journal.setdefault("execution", {}).update(details)
        atomic_json(path, journal)


def execution_details() -> dict:
    operation = getattr(_held, "operation", None)
    return dict(operation[1].get("execution", {})) if operation else {}


def execution_kind() -> str | None:
    operation = getattr(_held, "operation", None)
    return operation[1]["kind"] if operation else None


def locked_decision(kind: str):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            root, identifier = list(bound.arguments.values())[:2]
            with resource_locks(root, [f"artifact:{kind}:{identifier}"]):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def guarded_execution(kind: str, id_parameter: str):
    """Replay committed outcomes or collect original remote jobs, never repeat unknown effects."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            root = Path(next(iter(values.values()))).resolve()
            identifier = values[id_parameter]
            if kind == "validation" and not values["execute"]:
                return function(*args, **kwargs)
            if not identifier:
                # Legacy CLI callers retain their explicit non-approved execution path.
                # They have no stable business key and therefore no replay guarantee.
                return function(*args, **kwargs)
            with resource_locks(root, [f"artifact:{kind}:{identifier}"]):
                record = artifact(root, kind, identifier)
                approved = record.get("approval", record).get("status")
                if approved == "applied" and not permission_enabled(kind):
                    raise PermissionError("Host must enable this operation in permissions even for replay")
                if approved not in {"approved", "applied"} or not permission_enabled(kind):
                    return function(*args, **kwargs)
                seal = approval_digest(root, kind, record)
                if record.get("approval_digest") != seal:
                    raise PermissionError("Approved content changed or predates approval sealing; create a new proposal/plan")
                files = _files(record, kind)
                keys = ["file:" + str((root / name).resolve()) for name in files]
                with resource_locks(root, keys):
                    request = {key: value for key, value in values.items()
                               if key not in {next(iter(values)), id_parameter}}
                    execution_context = {"request": request, "approval": seal,
                                         "remote": remote_execution_identity()}
                    if kind in {"validation", "job"} and runtime_config().remote.host:
                        from codex_agent.linux_sandbox import launcher_digest
                        from codex_agent.remote_executor import remote_protocol_digest
                        execution_context["sandbox_launcher_sha256"] = launcher_digest()
                        execution_context["remote_protocol_sha256"] = remote_protocol_digest()
                    fingerprint = digest(execution_context)
                    path = journal_path(root, kind, identifier)
                    previous = inspect_execution(root, kind, identifier)
                    if previous:
                        if previous["fingerprint"] != fingerprint:
                            raise PermissionError("Idempotency key reused with different arguments or environment")
                        recovery_job = previous.get("execution", {}).get("remote_job")
                        recoverable = (kind == "validation" and previous["state"] in {"running", "unknown"}
                                       and isinstance(recovery_job, dict) and bool(recovery_job)
                                       and not record.get("approval", {}).get("execution_run_id"))
                        batch_recovery = False
                        if kind == "job" and previous["state"] in {"running", "unknown"}:
                            from codex_agent.batch_recovery import enabled, validate_recovery
                            if enabled(record):
                                validate_recovery(root, record, previous)
                                batch_recovery = recoverable = True
                        if previous["state"] != "completed" and not recoverable:
                            raise PermissionError(f"Execution outcome is {previous['state']}; do not replay. Inspect {path} and remote logs/processes before authorizing a NEW plan")
                        if recoverable:
                            if previous["files_before"] != file_snapshot(root, files):
                                raise PermissionError("Files changed during disconnect; preserve remote evidence for manual reconciliation")
                        else:
                            if previous["files_after"] != file_snapshot(root, files):
                                raise PermissionError("Files changed since completion; cached result is not valid for this version")
                            if previous.get("result_files", {}) != file_snapshot(root, list(previous.get("result_files", {}))):
                                raise PermissionError("Stored receipt, log or patch changed; cached evidence cannot be replayed")
                            result_type = get_type_hints(function)["return"]
                            return (result_type.model_validate(previous["result"])
                                    if hasattr(result_type, "model_validate") else previous["result"])
                    if record.get("status") == "applied" or record.get("approval", {}).get("execution_run_id"):
                        raise PermissionError("Legacy consumed artifact has no replay journal; inspect its receipt, do not rerun")
                    claims = [root / ROOT / "claims" / (digest(key) + ".json") for key in keys]
                    parent = getattr(_held, "operation", None)
                    inherited = []
                    for claim in claims:
                        if not claim.is_file():
                            continue
                        owner = json.loads(claim.read_text())
                        if previous and owner == {"kind": kind, "id": identifier}:
                            continue
                        if parent and owner == {"kind": parent[1]["kind"], "id": parent[1]["id"]}:
                            inherited.append((claim, owner))
                            continue
                        if kind == "job" and previous and batch_recovery:
                            from codex_agent.batch_recovery import owns_child_claim
                            if owns_child_claim(record, previous, owner):
                                continue
                        old = inspect_execution(root, owner["kind"], owner["id"])
                        if old and old["state"] not in {"completed", "rejected", "abandoned"}:
                            raise PermissionError(f"An unresolved execution owns these files: {owner['kind']}:{owner['id']}; reconcile it before creating another operation")
                    journal = previous or {"kind": kind, "id": identifier, "state": "running",
                               "fingerprint": fingerprint, "started_at": time.time(),
                               "pid": os.getpid(), "files_before": file_snapshot(root, files)}
                    if previous:
                        journal.update(state="running", recovery_attempts=journal.get("recovery_attempts", 0) + 1)
                    atomic_json(path, journal)
                    for claim in claims:
                        atomic_json(claim, {"kind": kind, "id": identifier})
                    _held.operation = (path, journal)
                    try:
                        with execution_budget(900, cancel_path(root, kind, identifier)):
                            result = function(*args, **kwargs)
                        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
                        journal.update(state="completed", result=payload,
                                       result_files=file_snapshot(root, _result_files(root, payload)),
                                       files_after=file_snapshot(root, files), completed_at=time.time())
                        atomic_json(path, journal)
                        jobs = journal.get("execution", {}).get("remote_jobs", {})
                        if jobs:
                            from codex_agent.remote_executor import acknowledge_remote_job
                            for remote_job in jobs.values():
                                acknowledge_remote_job(remote_job)
                    except BaseException as error:
                        # A write, remote job, or receipt may already have happened.
                        journal.update(state="unknown" if journal.get("effects_started") else "rejected",
                                       error_type=type(error).__name__)
                        atomic_json(path, journal)
                        raise
                    finally:
                        _held.operation = parent
                        if journal["state"] in {"completed", "rejected"}:
                            for claim, owner in inherited:
                                atomic_json(claim, owner)
                    return result
        return wrapped
    return decorate
