"""Guarded SSH execution for Triton-RISCV validation."""

from __future__ import annotations

import os
import base64
import hashlib
import json
import re
import shlex
import subprocess
import time
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from pydantic import BaseModel, Field
from codex_agent.failure_diagnosis import diagnose_log
from codex_agent.process_control import run_bounded, remaining
from codex_agent.linux_sandbox import POLICY as SANDBOX_POLICY, launcher_digest
from codex_agent.remote_job import digest, MAX_LOG_BYTES
from codex_agent.remote_capacity import MAX_ACTIVE_JOBS, QUEUE_TIMEOUT_SECONDS
from codex_agent.remote_retry import observe_with_retry, TransportUnavailable
from codex_agent import resource_probe

from codex_agent.validate_operator import (
    OperatorValidationResult,
    classify_log,
    extract_error_excerpt,
    slugify,
)


REMOTE_HOST_RE = re.compile(r"[A-Za-z0-9_.-]+")
REMOTE_ROOT_RE = re.compile(r"/[A-Za-z0-9_./-]+")
ALLOWED_SYNC_ROOT = PurePosixPath("python/examples/flaggems")
PREFLIGHT_TIMEOUT_SECONDS = 90
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


class RemotePreflightResult(BaseModel):
    """Structured result of checking the configured RISC-V validation host."""

    configured: bool
    status: str
    host: str | None = None
    repository: str | None = None
    architecture: str | None = None
    triton_version: str | None = None
    python_path: str | None = None
    triton_shared_opt: str | None = None
    buddy_opt: str | None = None
    exit_code: int | None = None
    duration_seconds: float = 0.0
    error_excerpt: list[str] = Field(default_factory=list)
    isolation_support: str = "not-probed"
    resource_capabilities: dict = Field(default_factory=resource_probe.unknown)


@dataclass(frozen=True)
class RemoteValidationConfig:
    """Host-owned remote target; model-facing tools cannot override it."""

    host: str
    repository: str

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "RemoteValidationConfig | None":
        env = os.environ if environ is None else environ
        from codex_agent.runtime_config import runtime_config
        settings = runtime_config(env).remote
        host, repository = settings.host, settings.repository
        if not host and not repository:
            return None
        if not host or not repository:
            raise ValueError("RISCV_HOST and RISCV_REPO must be configured together")
        if not REMOTE_HOST_RE.fullmatch(host):
            raise ValueError("RISCV_HOST contains unsupported characters")
        if not REMOTE_ROOT_RE.fullmatch(repository):
            raise ValueError("RISCV_REPO must be an absolute path without spaces")
        normalized = PurePosixPath(repository)
        if ".." in normalized.parts:
            raise ValueError("RISCV_REPO cannot contain parent traversal")
        return cls(host=host, repository=normalized.as_posix())


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _run_ssh_script(
    config: RemoteValidationConfig,
    script: str,
    arguments: list[str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    remote_argv = ["bash", "-s", "--", *arguments]
    seconds = remaining(timeout_seconds)
    if script == VALIDATION_SCRIPT:
        # Bound the remote process independently of the SSH connection. This
        # also covers environment activation and snapshot preparation.
        remote_argv = ["timeout", "--signal=TERM", "--kill-after=10s", str(max(1, int(seconds) - 15)), *remote_argv]
    return run_bounded(
        ["ssh", *SSH_OPTIONS, config.host,
         shlex.join(remote_argv)],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=seconds,
        check=False,
    )


PREFLIGHT_SCRIPT = r"""
set -eo pipefail
root=$1
cd "$root" || { echo "PREFLIGHT_ERROR=repository-not-found"; exit 70; }
test -f .venv/bin/activate || { echo "PREFLIGHT_ERROR=venv-not-found"; exit 71; }
test -f scripts/triton-riscv-env.sh || { echo "PREFLIGHT_ERROR=environment-script-not-found"; exit 72; }
. .venv/bin/activate
. scripts/triton-riscv-env.sh
architecture=$(uname -m)
printf 'ARCHITECTURE=%s\n' "$architecture"
test "$architecture" = riscv64 || { echo "PREFLIGHT_ERROR=not-riscv64"; exit 73; }
printf 'PYTHON_PATH=%s\n' "$(command -v python)"
version=$(python -c 'import triton; import pytest; print(triton.__version__)') || { echo "PREFLIGHT_ERROR=python-dependency-import-failed"; exit 75; }
shared=$(command -v triton-shared-opt) || { echo "PREFLIGHT_ERROR=triton-shared-opt-not-found"; exit 76; }
buddy=$(command -v buddy-opt) || { echo "PREFLIGHT_ERROR=buddy-opt-not-found"; exit 77; }
printf 'TRITON_VERSION=%s\n' "$version"
printf 'TRITON_SHARED_OPT=%s\n' "$shared"
printf 'BUDDY_OPT=%s\n' "$buddy"
command -v timeout >/dev/null || { echo "PREFLIGHT_ERROR=timeout-not-found"; exit 74; }
for tool in unshare mount chroot setpriv; do
  command -v "$tool" >/dev/null || { echo "PREFLIGHT_ERROR=isolation-tool-missing:$tool"; exit 80; }
done
timeout --kill-after=3s 10s unshare --user --map-root-user --mount --net --ipc --pid --fork --kill-child=KILL --mount-proc true || { echo "PREFLIGHT_ERROR=rootless-namespaces-unavailable"; exit 80; }
printf 'ISOLATION_SUPPORT=rootless-namespaces\n'
"""
PREFLIGHT_SCRIPT += ("\npython -I - <<'TRITON_RESOURCE_PROBE'\n" +
                     Path(resource_probe.__file__).read_text(encoding="utf-8") +
                     "\nTRITON_RESOURCE_PROBE\n")


def _fields(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "ARCHITECTURE",
            "PYTHON_PATH",
            "TRITON_VERSION",
            "TRITON_SHARED_OPT",
            "BUDDY_OPT",
            "ISOLATION_SUPPORT",
        }:
            result[key] = value.strip()
    return result


def check_remote_environment(
    config: RemoteValidationConfig | None = None,
) -> RemotePreflightResult:
    """Check architecture and required compiler tools on the configured host."""

    selected = config or RemoteValidationConfig.from_env()
    if selected is None:
        return RemotePreflightResult(
            configured=False,
            status="blocked",
            error_excerpt=["RISCV_HOST and RISCV_REPO are not configured"],
        )

    policy_failure = _quota_policy_failure(selected)
    if policy_failure:
        return policy_failure

    start = time.monotonic()
    try:
        completed = _run_ssh_script(
            selected,
            PREFLIGHT_SCRIPT,
            [selected.repository],
            PREFLIGHT_TIMEOUT_SECONDS,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = _decode_timeout_output(error.stdout)
        output += "\nPREFLIGHT_ERROR=ssh-timeout\n"
        exit_code = 124
    except OSError as error:
        output = f"PREFLIGHT_ERROR={error}"
        exit_code = 127
    values = _fields(output)
    required = ("PYTHON_PATH", "TRITON_VERSION", "TRITON_SHARED_OPT", "BUDDY_OPT")
    if exit_code == 0 and (values.get("ARCHITECTURE") != "riscv64" or
                           any(not values.get(key) for key in required)):
        output += "\nPREFLIGHT_ERROR=incomplete-or-invalid-environment-evidence\n"
        exit_code = 78
    status = "passed" if exit_code == 0 else "failed"
    return RemotePreflightResult(
        configured=True,
        status=status,
        host=selected.host,
        repository=selected.repository,
        architecture=values.get("ARCHITECTURE"),
        triton_version=values.get("TRITON_VERSION"),
        python_path=values.get("PYTHON_PATH"),
        triton_shared_opt=values.get("TRITON_SHARED_OPT"),
        buddy_opt=values.get("BUDDY_OPT"),
        exit_code=exit_code,
        duration_seconds=round(time.monotonic() - start, 3),
        error_excerpt=extract_error_excerpt(output) if exit_code else [],
        isolation_support=values.get("ISOLATION_SUPPORT", "not-probed"),
        resource_capabilities=resource_probe.decode(output),
    )


def _quota_policy_failure(config):
    error = resource_probe.quota_policy_error()
    if error:
        return RemotePreflightResult(
            configured=True, status="blocked", host=config.host,
            repository=config.repository, exit_code=82, error_excerpt=[error],
            resource_capabilities=resource_probe.unknown("new task blocked by resource policy"),
        )
    return None


def _validated_relative_file(repo_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("remote validation file escaped the repository")
    if pure.parent != ALLOWED_SYNC_ROOT:
        raise ValueError("remote validation can sync only FlagGems source and tests")
    local = (repo_root.resolve() / Path(*pure.parts)).resolve()
    local.relative_to(repo_root.resolve())
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


def _validated_test_nodes(operator: dict) -> list[str]:
    test_files = set(operator.get("test_files", []))
    nodes = list(operator.get("test_nodes", []))
    if not nodes:
        raise ValueError("operator has no test nodes")
    for node in nodes:
        test_file = node.split("::", 1)[0]
        if test_file not in test_files or "\n" in node or "\x00" in node:
            raise ValueError("operator contains an unsafe test node")
    return nodes


def remote_validation_command(
    operator: dict,
    config: RemoteValidationConfig,
) -> tuple[list[str], str]:
    """Build the fixed pytest argv and a reviewable remote display command."""

    command = ["python", "-m", "pytest", "-q", *_validated_test_nodes(operator), "-s"]
    display = (f"ssh {config.host} -- [{SANDBOX_POLICY}; launcher={launcher_digest()[:16]}; "
               f"durable-job-v1={remote_protocol_digest()[:16]}; capacity={MAX_ACTIVE_JOBS}; queue={QUEUE_TIMEOUT_SECONDS}s; "
               f"readonly snapshot of {config.repository}; network disabled; fresh per-run cache] {shlex.join(command)}")
    return command, display


VALIDATION_SCRIPT = r"""
set -eo pipefail
root=$1
stage=$2
test_timeout=$3
file_count=$4
shift 4
files=()
for ((index = 0; index < file_count; index++)); do
  files+=("$1")
  shift
done
command=("$@")
case "$stage" in /tmp/triton-riscv-agent/*) ;; *) exit 79 ;; esac
test -d "$stage/payload"
: > "$stage/started"
child=""
cleanup() {
  if test -n "$child"; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  rm -rf -- "$stage"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
cd "$root" || exit 70
. .venv/bin/activate
. scripts/triton-riscv-env.sh
work="$stage/workspace"
# Copy the small example tree, never overwrite the shared checkout or follow
# source symlinks. The installed backend/toolchain remains read-only input.
python - "$root" "$stage" "${files[@]}" <<'PY'
import os
import shutil
import sys
from pathlib import Path
root, stage = map(Path, sys.argv[1:3])
work = stage / "workspace"
budget = 100 * 1024 * 1024
copied = 0
def copy_file(src, dst):
    global copied
    src = Path(src)
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"unsupported snapshot source: {src}")
    copied += src.stat().st_size
    if copied > budget:
        raise ValueError("remote source snapshot exceeds 100 MiB")
    return shutil.copy2(src, dst)
source = root / "python/examples/flaggems"
for directory, dirs, names in os.walk(source):
    dirs[:] = [name for name in dirs if name not in {"__pycache__", ".pytest_cache"}]
    if Path(directory).is_symlink() or any((Path(directory) / name).is_symlink() for name in dirs):
        raise ValueError("symlink directories are not allowed in the snapshot")
shutil.copytree(source, work / "python/examples/flaggems", copy_function=copy_file,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
for relative in ("conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg",
                 "python/__init__.py", "python/conftest.py",
                 "python/examples/__init__.py", "python/examples/conftest.py"):
    src = root / relative
    if src.exists():
        copy_file(src, work / relative)
for index, relative in enumerate(sys.argv[3:]):
    path = Path(relative)
    if path.parent.as_posix() != "python/examples/flaggems":
        raise ValueError("invalid snapshot destination")
    copy_file(stage / "payload" / str(index), work / path)
print(f"REMOTE_WORKSPACE={work}", flush=True)
print(f"REMOTE_SNAPSHOT_BYTES={copied}", flush=True)
PY
mkdir -p "$stage/tmp" "$stage/cache" "$stage/dump" "$stage/override"
export TMPDIR="$stage/tmp" PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$stage/cache" TRITON_DUMP_DIR="$stage/dump"
export TRITON_OVERRIDE_DIR="$stage/override" TRITON_SHARED_DUMP_PATH="$stage/dump/shared"
export PYTHONPATH="$work:$root:${PYTHONPATH:-}"
cd "$work"
test -f "$stage/sandbox.py" || { echo "SANDBOX_ERROR=trusted launcher missing"; exit 79; }
timeout --signal=TERM --kill-after=10s "${test_timeout}s" python -I "$stage/sandbox.py" --repo "$root" --stage "$stage" -- "${command[@]}" &
child=$!
status=0
wait "$child" || status=$?
child=""
exit "$status"
"""


def _cleanup_stage(config: RemoteValidationConfig, stage: str) -> None:
    try:
        _run_ssh_script(
            config,
            'case "$1" in /tmp/triton-riscv-agent/*) '
            'test -f "$1/started" || rm -rf -- "$1" ;; esac\n',
            [stage],
            30,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def remote_protocol_digest() -> str:
    return digest({"supervisor": hashlib.sha256(Path(__file__).with_name("remote_job.py").read_bytes()).hexdigest(),
                   "capacity": hashlib.sha256(Path(__file__).with_name("remote_capacity.py").read_bytes()).hexdigest(),
                   "process_control": hashlib.sha256(Path(__file__).with_name("remote_process.py").read_bytes()).hexdigest(),
                   "validation_script": VALIDATION_SCRIPT})


def _job_rpc(config, job, action, log_digest=None):
    stage = job["stage"]
    if (not re.fullmatch(r"/tmp/triton-riscv-agent/[a-f0-9]{32}", stage) or
            Path(stage).name != job.get("job_id") or
            not re.fullmatch(r"[a-f0-9]{64}", job.get("request_digest", ""))):
        raise ValueError("invalid recovery stage")
    args = [stage, action, job["request_digest"]]
    if log_digest:
        args.append(log_digest)
    def request():
        result = _run_ssh_script(config, 'python3 -I "$1/remote_job.py" "$2" "$1" "$3" "${@:4}"\n', args, 25)
        if result.returncode == 255:
            raise TransportUnavailable("SSH transport unavailable")
        if result.returncode != 0:
            raise RuntimeError(f"remote status request failed: {result.stdout[:1000]}")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("remote status is not an object")
        if action != "ack" and (value.get("job_id") != job["job_id"] or
                                value.get("request_digest") != job["request_digest"]):
            raise ValueError("remote status identity mismatch")
        return value
    def retry(number, error_type):
        events = list(job.get("transport_retries", []))
        events.append({"action": action, "retry": number, "error_type": error_type, "at": time.time()})
        job["transport_retries"] = events[-32:]
        job["transport_retry_count"] = job.get("transport_retry_count", 0) + 1
        _record_job(job)
    try:
        return observe_with_retry(action, request, on_retry=retry)
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        if action.startswith("capacity-") or action == "cleanup-preview":
            raise RuntimeError(f"Maintenance not confirmed; inspect the original task, never rerun the test. {error}") from error
        raise RuntimeError(f"Remote outcome unknown; retry the SAME approved run to retrieve, never start a new test. {error}") from error


def acknowledge_remote_job(job):
    """Only called after durable local receipt/journal commit. Failure retains evidence."""
    if not job.get("log_sha256"):
        return
    try:
        _job_rpc(RemoteValidationConfig(job["host"], job["repository"]), job, "ack", job["log_sha256"])
    except (RuntimeError, ValueError):
        pass


def _record_job(job):
    from codex_agent.execution_guard import execution_details, execution_kind, record_execution_detail
    jobs = dict(execution_details().get("remote_jobs", {}))
    jobs[job["job_id"]] = job
    fields = {"remote_jobs": jobs, "remote_host": job["host"], "remote_stage": job["stage"]}
    if execution_kind() == "validation":
        fields["remote_job"] = job
    record_execution_detail(**fields)


def _collect_job(config, job, timeout_seconds):
    deadline = time.monotonic() + remaining(timeout_seconds + 60 + QUEUE_TIMEOUT_SECONDS)
    while time.monotonic() < deadline:
        value = _job_rpc(config, job, "inspect")
        if value["state"] == "completed":
            final = _job_rpc(config, job, "collect")
            raw = base64.b64decode(final["log_base64"], validate=True)
            if (len(raw) > MAX_LOG_BYTES or len(raw) != final["log_bytes"] or
                    hashlib.sha256(raw).hexdigest() != final["log_sha256"] or
                    type(final.get("exit_code")) is not int):
                raise RuntimeError("Remote outcome unknown; collected log/terminal evidence mismatch")
            return final, raw
        if value["state"] not in {"accepted", "queued", "running"}:
            raise RuntimeError(f"Remote outcome unknown ({value['state']}); no automatic restart")
        _record_job({**job, "observed_state": value["state"], "observed_at": time.time()})
        remaining(timeout_seconds + 60 + QUEUE_TIMEOUT_SECONDS)
        time.sleep(1)
    raise RuntimeError("Remote outcome unknown; observation timed out; original job keeps its server deadline")


def run_remote_operator(
    operator: dict,
    *,
    repo_root: Path,
    results_dir: Path,
    timeout_seconds: int,
    config: RemoteValidationConfig | None = None,
) -> tuple[OperatorValidationResult, RemotePreflightResult]:
    """Validate in a disposable source snapshot without changing the checkout."""

    selected = config or RemoteValidationConfig.from_env()
    if selected is None:
        raise ValueError("remote validation is not configured")
    from codex_agent.execution_guard import atomic_json, execution_details, execution_kind
    details = execution_details() if execution_kind() == "validation" else {}
    job = details.get("remote_job")
    if "remote_job" in details and (not isinstance(job, dict) or not all(
        key in job for key in ("host", "repository", "job_id", "stage", "request_digest", "preflight")
    )):
        raise PermissionError("Original remote job handle is incomplete; do not restart")
    if job and (job["host"] != selected.host or job["repository"] != selected.repository):
        raise PermissionError("Recovery target differs from the original remote task")
    # A stricter deployment policy blocks new dispatch, never observation/stop
    # of an already-running task whose original evidence must remain collectable.
    preflight = (RemotePreflightResult.model_validate(job["preflight"]) if job
                 else (_quota_policy_failure(selected) or check_remote_environment(selected)))
    if job and preflight.status != "passed":
        raise PermissionError("Original remote preflight evidence is invalid; do not restart")
    if preflight.status != "passed":
        output = "\n".join(preflight.error_excerpt)
        result = OperatorValidationResult(
            operator=operator["name"],
            implementation_file=operator["implementation_file"],
            test_files=operator["test_files"],
            command=f"ssh {selected.host} <remote-preflight>",
            dry_run=False,
            exit_code=preflight.exit_code,
            status="failed",
            failure_stage="environment",
            likely_reason="remote RISC-V environment preflight failed",
            error_excerpt=preflight.error_excerpt or [output],
            duration_seconds=preflight.duration_seconds,
            log_path=None,
            diagnosis={"failure_stage": "environment", "repairable": False,
                       "summary": "Remote execution prerequisites were not satisfied; no operator test ran.",
                       "recommended_actions": ["Review host configuration and resource capabilities; do not repair operator code."]},
        )
        return result, preflight

    relative_files = [operator["implementation_file"], *operator["test_files"]]
    relative_files = list(dict.fromkeys(relative_files))
    local_files = [
        _validated_relative_file(repo_root, relative)
        for relative in relative_files
    ]
    command, display_command = remote_validation_command(operator, selected)
    if not job:
        job_id = uuid.uuid4().hex
        stage = f"/tmp/triton-riscv-agent/{job_id}"
        create = _run_ssh_script(selected, 'umask 077; mkdir -p /tmp/triton-riscv-agent\n'
                                 'mkdir "$1" && mkdir -p "$1/execution/payload"\n', [stage], 30)
        if create.returncode != 0:
            raise RuntimeError(f"remote staging failed: {create.stdout.strip()}")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                script = folder / "validation.sh"
                script.write_text(VALIDATION_SCRIPT)
                uploads = [(Path(__file__).with_name("linux_sandbox.py"), "execution/sandbox.py"),
                           (Path(__file__).with_name("remote_job.py"), "remote_job.py"),
                           (Path(__file__).with_name("remote_capacity.py"), "remote_capacity.py"),
                           (Path(__file__).with_name("remote_process.py"), "remote_process.py"),
                           (script, "validation.sh")]
                uploads += [(path, f"execution/payload/{index}") for index, path in enumerate(local_files)]
                request = {"version": 1, "job_id": job_id, "repository": selected.repository,
                           "timeout_seconds": timeout_seconds, "files": relative_files, "command": command,
                           "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in uploads}}
                request_path = folder / "request.json"
                request_path.write_text(json.dumps(request))
                uploads.append((request_path, "request.json"))
                for path, name in uploads:
                    uploaded = run_bounded(["scp", *SSH_OPTIONS, str(path), f"{selected.host}:{stage}/{name}"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120, check=False)
                    if uploaded.returncode:
                        raise RuntimeError(f"sandbox launcher/job input transfer failed: {name}; validation not started")
        except BaseException:
            _cleanup_stage(selected, stage)
            raise
        job = {"job_id": job_id, "stage": stage, "host": selected.host, "repository": selected.repository,
               "request_digest": digest(request), "preflight": preflight.model_dump()}
        # Commit the handle BEFORE start. Lost start acknowledgments recover this
        # exact job; they cannot create a fresh job with a new UUID.
        atomic_json(results_dir / "remote-jobs" / f"{job_id}.json", job)
        _record_job(job)
        _job_rpc(selected, job, "start")
    stage = job["stage"]
    terminal, raw = _collect_job(selected, job, timeout_seconds)
    output = raw.decode("utf-8", "replace")
    exit_code = terminal["exit_code"]
    duration = terminal["duration_seconds"]

    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{timestamp}-{Path(stage).name[:8]}-{uuid.uuid4().hex[:8]}-remote-{slugify(operator['name'])}.log"
    with log_path.open("xb") as log:
        log.write(raw)
        log.flush()
        os.fsync(log.fileno())
    fd = os.open(log_dir, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    job = {**job, "log_sha256": terminal["log_sha256"], "terminal": terminal}
    job["terminal"].pop("log_base64", None)
    atomic_json(results_dir / "remote-jobs" / f"{job['job_id']}.json", job)
    _record_job(job)
    status, failure_stage, likely_reason = classify_log(output, exit_code)
    diagnosis = diagnose_log(output, exit_code, validation_command=display_command)
    not_admitted = terminal.get("admission", {}).get("state") == "not-admitted"
    was_cancelled = terminal.get("cancellation", {}).get("confirmed") is True
    if not_admitted:
        status, failure_stage = "failed", "capacity"
        likely_reason = "server validation capacity unavailable; test did not start"
        diagnosis = {**diagnosis, "failure_stage": "capacity", "summary": likely_reason,
                     "repairable": False, "recommended_actions": [
                         "Check active or fenced server slots before approving a new attempt; do not edit operator code."]}
    if exit_code == 124:
        failure_stage = "timeout"
        likely_reason = "remote validation command timed out"
    if exit_code == 79 or "SANDBOX_ERROR=" in output:
        status, failure_stage = "failed", "isolation"
        likely_reason = "isolated validation could not start; no fallback to unrestricted execution"
    if was_cancelled:
        status, failure_stage = "cancelled", "cancellation"
        likely_reason = "host cancellation confirmed; this is not an operator correctness failure"
        diagnosis = {**diagnosis, "failure_stage": "cancellation", "summary": likely_reason,
                     "repairable": False, "recommended_actions": ["Preserve cancellation evidence; do not repair operator source."]}
    return (
        OperatorValidationResult(
            operator=operator["name"],
            implementation_file=operator["implementation_file"],
            test_files=operator["test_files"],
            command=display_command,
            dry_run=False,
            exit_code=exit_code,
            status=status,
            failure_stage=failure_stage,
            likely_reason=likely_reason,
            error_excerpt=extract_error_excerpt(output) if status == "failed" else [],
            duration_seconds=round(duration, 3),
            log_path=log_path.as_posix(),
            fresh_compile=not (not_admitted or was_cancelled),
            execution_mode="native-riscv" if preflight.architecture == "riscv64" else "unknown",
            diagnosis=diagnosis,
            isolation={"policy": SANDBOX_POLICY, "launcher_sha256": launcher_digest(),
                       "enforcement": "required", "network": "private-disabled",
                       "source_access": "read-only", "writable": ["/tmp", "/dev/shm"],
                       "setup_failed": failure_stage == "isolation",
                       "resource_capabilities": preflight.resource_capabilities},
        ),
        preflight,
    )
