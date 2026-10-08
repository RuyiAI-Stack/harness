"""Trusted, stdlib-only remote supervisor. Never imports candidate code.

One durable dispatch per job. Losing SSH cannot restart it; an absent/corrupt
terminal record is unknown, never proof that the test failed or may be repeated.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

MAX_LOG_BYTES = 16 * 1024 * 1024


def trusted_module(name):
    # -I deliberately excludes the script directory from sys.path. Load only the
    # trusted sibling whose checksum was verified with the other staged inputs.
    if __package__:
        from importlib import import_module
        return import_module("codex_agent." + name)
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cancelled(stage, expected):
    marker = stage / "cancel.json"
    if not marker.exists():
        return False
    value = json.loads(marker.read_text())
    if value.get("request_digest") != expected or value.get("job_id") != stage.name:
        raise ValueError("cancellation identity mismatch")
    return True


def request_cancel(stage, expected):
    request = load_request(stage, expected)
    with (stage / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (stage / "result.json").exists():
            return inspect_job(stage, expected)
        atomic_json(stage / "cancel.json", {"job_id": stage.name, "request_digest": expected,
                                            "requested_at": time.time()})
        if not (stage / "dispatch.json").exists():
            return save_cancelled(stage, request, expected, time.monotonic(), before_start=True)
    result = inspect_job(stage, expected)
    return {**result, "cancel_requested": True}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def load_request(stage, expected):
    if stage.parent != Path("/tmp/triton-riscv-agent") or not re.fullmatch("[a-f0-9]{32}", stage.name):
        raise ValueError("invalid job directory")
    if stage.is_symlink() or stage.parent.is_symlink():
        raise ValueError("symlink job directory")
    stat = stage.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError("job directory must be private and owned by the executing account")
    request = json.loads((stage / "request.json").read_text())
    if digest(request) != expected or request["job_id"] != stage.name or request["version"] != 1:
        raise ValueError("job identity or request digest mismatch")
    if not 1 <= request["timeout_seconds"] <= 3600:
        raise ValueError("invalid job timeout")
    return request


def inspect_job(stage, expected, *, include_log=False):
    request = load_request(stage, expected)
    result_path = stage / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("request_digest") != expected or result.get("job_id") != request["job_id"]:
            raise ValueError("terminal record belongs to a different request")
        if result.get("state") != "completed" or type(result.get("exit_code")) is not int:
            raise ValueError("invalid terminal record")
        log = stage / "output.log"
        if log.stat().st_size != result["log_bytes"] or file_digest(log) != result["log_sha256"]:
            raise ValueError("remote log does not match terminal record")
        if include_log:
            if result["log_bytes"] > MAX_LOG_BYTES:
                raise ValueError("remote log exceeds retrieval limit; preserve job for manual inspection")
            result["log_base64"] = base64.b64encode(log.read_bytes()).decode("ascii")
        return result
    with trusted_module("remote_capacity")._open_lock(stage / "worker.lock") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = "accepted"
            phase_path = stage / "worker-state.json"
            if phase_path.exists():
                phase = json.loads(phase_path.read_text())
                if phase.get("request_digest") != expected or phase.get("state") not in {"queued", "running"}:
                    raise ValueError("invalid worker phase")
                state = phase["state"]
        else:
            if (stage / "worker-claimed").exists():
                state = "unknown"
            elif (stage / "dispatch.json").exists():
                dispatch = json.loads((stage / "dispatch.json").read_text())
                state = "accepted" if time.time() - dispatch["created_at"] < 30 else "unknown"
            else:
                state = "not-started"
    return {"job_id": request["job_id"], "request_digest": expected, "state": state}


def start_job(stage, expected):
    load_request(stage, expected)
    with (stage / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (stage / "result.json").exists():
            return inspect_job(stage, expected)
        if not (stage / "dispatch.json").exists():
            # Write before spawning. A crash in this narrow gap stays unknown;
            # repeat start requests must not risk executing twice.
            atomic_json(stage / "dispatch.json", {"request_digest": expected, "created_at": time.time()})
            with (stage / "supervisor.log").open("ab") as output:
                subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "worker",
                                  str(stage), expected], stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    return inspect_job(stage, expected)


def run_worker(stage, expected):
    request = load_request(stage, expected)
    with trusted_module("remote_capacity")._open_lock(stage / "worker.lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (stage / "worker-claimed").open("x") as claim:
            claim.write(expected)
            claim.flush()
            os.fsync(claim.fileno())
        for relative, checksum in request["input_sha256"].items():
            path = stage / relative
            if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()):
                raise ValueError("job input escaped stage")
            if file_digest(path) != checksum:
                raise ValueError("staged input changed before execution")
        started = time.monotonic()
        capacity = trusted_module("remote_capacity")
        atomic_json(stage / "worker-state.json", {"state": "queued", "request_digest": expected})
        try:
            with capacity.server_pool().acquire(request["job_id"], request_digest=expected,
                                               cancelled=lambda: cancelled(stage, expected)) as admission:
                if cancelled(stage, expected):
                    return save_cancelled(stage, request, expected, started, before_start=True, admission=admission)
                admission["started_at"] = time.time()
                atomic_json(stage / "worker-state.json", {"state": "running", "request_digest": expected})
                with (stage / "validation.sh").open("rb") as script, (stage / "output.log").open("wb") as output:
                    argv = ["timeout", "--signal=TERM", "--kill-after=10s", str(request["timeout_seconds"] + 45),
                            "bash", "-s", "--", request["repository"], str(stage / "execution"),
                            str(request["timeout_seconds"]), str(len(request["files"])),
                            *request["files"], *request["command"]]
                    outcome = trusted_module("remote_process").run_controlled(
                        argv, stdin=script, stdout=output, cancelled=lambda: cancelled(stage, expected))
                    if outcome["cancel_applied"]:
                        output.write(b"\nVALIDATION_CANCELLED=supervisor confirmed its descendants stopped\n")
                    output.flush()
                    os.fsync(output.fileno())
                cancellation = {"requested": outcome["cancel_applied"],
                                "confirmed": outcome["cancel_applied"] and outcome["descendants_stopped"],
                                "supported": outcome["cancellation_supported"]}
                result = save_result(stage, request, expected,
                                     130 if outcome["cancel_applied"] else outcome["returncode"], started,
                                     {"state": "admitted", **admission}, cancellation=cancellation,
                                     shutdown={"version": 1, "descendants_stopped": outcome["descendants_stopped"],
                                               "method": "linux-subreaper-pidfd"})
        except capacity.AdmissionCancelled:
            result = save_cancelled(stage, request, expected, started, before_start=True)
        except capacity.CapacityUnavailable as error:
            with (stage / "output.log").open("wb") as output:
                output.write(f"CAPACITY_ERROR={error}\n".encode())
                output.flush()
                os.fsync(output.fileno())
            result = save_result(stage, request, expected, 81, started,
                                 {"state": "not-admitted", "limit": capacity.MAX_ACTIVE_JOBS,
                                  "queue_seconds": round(time.monotonic() - started, 3)})
    return result


def save_cancelled(stage, request, expected, started, *, before_start, admission=None):
    with (stage / "output.log").open("wb") as log:
        log.write(b"VALIDATION_CANCELLED=cancelled before command startup\n")
        log.flush()
        os.fsync(log.fileno())
    return save_result(stage, request, expected, 130, started, {"state": "cancelled-before-start", **(admission or {})},
                       cancellation={"requested": True, "confirmed": True, "before_start": before_start},
                       shutdown={"version": 1, "descendants_stopped": True, "method": "not-started"})


def save_result(stage, request, expected, exit_code, started, admission, *, cancellation=None, shutdown=None):
    log = stage / "output.log"
    result = {"version": 1, "job_id": request["job_id"], "request_digest": expected,
              "state": "completed", "exit_code": exit_code,
              "duration_seconds": round(time.monotonic() - started, 3),
              "log_bytes": log.stat().st_size, "log_sha256": file_digest(log),
              "finished_at": time.time(), "admission": admission}
    if cancellation is not None:
        result["cancellation"] = cancellation
    if shutdown is not None:
        result["shutdown"] = shutdown
    atomic_json(stage / "result.json", result)
    return result


def capacity_evidence(stage, expected, slot, reservation):
    capacity = trusted_module("remote_capacity")
    # Do not create missing worker locks, nor infer death from PID/mtime alone.
    with capacity._open_lock(stage / "worker.lock", create=False) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (stage / "result.json").is_file():
            raise ValueError("no terminal evidence; descendants may still be running")
        result = inspect_job(stage, expected)
        admission = result.get("admission", {})
        shutdown = result.get("shutdown", {})
        if (admission.get("slot") != slot or admission.get("reservation_id") != reservation.get("reservation_id") or
                shutdown.get("version") != 1 or shutdown.get("descendants_stopped") is not True or
                shutdown.get("method") not in {"linux-subreaper-pidfd", "not-started"}):
            raise ValueError("missing or mismatched descendant shutdown proof")
        return {"terminal_sha256": file_digest(stage / "result.json"), "log_sha256": result["log_sha256"],
                "shutdown": shutdown, "exit_code": result["exit_code"]}


def capacity_operation(stage, expected, token=None):
    load_request(stage, expected)
    capacity = trusted_module("remote_capacity")
    def verify(slot, reservation):
        return capacity_evidence(stage, expected, slot, reservation)
    result = (capacity.inspect_job_capacity(capacity.server_root(), stage.name, expected, verify) if token is None else
              capacity.release_job_capacity(capacity.server_root(), stage.name, expected, token, verify))
    return {**result, "job_id": stage.name, "request_digest": expected}


def acknowledge(stage, expected, log_digest):
    result = inspect_job(stage, expected)
    if result["state"] != "completed" or result.get("log_sha256") != log_digest:
        raise ValueError("only a collected terminal result can be acknowledged")
    capacity = trusted_module("remote_capacity")
    for row in capacity.inspect_existing(capacity.server_root()):
        if (row["reservation"] or {}).get("job_id") == stage.name:
            raise ValueError("preserve terminal evidence until this capacity reservation is released")
    # Local caller may acknowledge only after its receipt/journal commit.
    shutil.rmtree(stage)
    return {"state": "removed", "job_id": stage.name}


def preview_cleanup(stage, expected):
    """No creation/deletion, even for missing terminal evidence or worker locks."""
    load_request(stage, expected)
    report = {"job_id": stage.name, "request_digest": expected, "state": "protected"}
    if not (stage / "result.json").is_file():
        return {**report, "reason": "terminal-evidence-missing"}
    result = inspect_job(stage, expected)
    capacity = trusted_module("remote_capacity")
    for row in capacity.inspect_existing(capacity.server_root()):
        if (row["reservation"] or {}).get("job_id") == stage.name:
            return {**report, "reason": "capacity-still-reserved"}
    shutdown = result.get("shutdown", {})
    if shutdown.get("version") != 1 or shutdown.get("descendants_stopped") is not True:
        return {**report, "reason": "shutdown-proof-missing-or-legacy"}
    method = shutdown.get("method")
    if method == "not-started":
        if result.get("cancellation", {}).get("before_start") is not True:
            return {**report, "reason": "pre-start-proof-mismatch"}
    elif method != "linux-subreaper-pidfd":
        return {**report, "reason": "unsupported-shutdown-proof"}
    try:
        with capacity._open_lock(stage / "worker.lock", create=False) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except FileNotFoundError:
        if method != "not-started" or (stage / "dispatch.json").exists() or (stage / "worker-claimed").exists():
            return {**report, "reason": "worker-state-missing"}
    except BlockingIOError:
        return {**report, "reason": "worker-still-active"}
    return {**report, "state": "eligible", "log_sha256": result["log_sha256"],
            "terminal_sha256": digest(result), "finished_at": result["finished_at"],
            "log_bytes": result["log_bytes"], "exit_code": result["exit_code"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "inspect", "collect", "worker", "ack", "cancel",
                                          "capacity-inspect", "capacity-release", "cleanup-preview"])
    parser.add_argument("stage", type=Path)
    parser.add_argument("digest")
    parser.add_argument("log_digest", nargs="?")
    args = parser.parse_args()
    try:
        if args.action == "start":
            result = start_job(args.stage, args.digest)
        elif args.action == "worker":
            result = run_worker(args.stage, args.digest)
        elif args.action == "cancel":
            result = request_cancel(args.stage, args.digest)
        elif args.action == "capacity-inspect":
            result = capacity_operation(args.stage, args.digest)
        elif args.action == "capacity-release":
            if not args.log_digest:
                raise ValueError("release token required")
            result = capacity_operation(args.stage, args.digest, args.log_digest)
        elif args.action == "ack":
            result = acknowledge(args.stage, args.digest, args.log_digest)
        elif args.action == "cleanup-preview":
            result = preview_cleanup(args.stage, args.digest)
        else:
            result = inspect_job(args.stage, args.digest, include_log=args.action == "collect")
        print(json.dumps(result), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"state": "unknown", "error": str(error)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
