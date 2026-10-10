"""Shared wall-clock budget and process-group shutdown for bounded tool execution."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import signal
import subprocess
import time


_deadline = ContextVar("execution_deadline", default=None)
_cancel = ContextVar("execution_cancel", default=None)


class ExecutionCancelled(RuntimeError):
    pass


def validation_environment() -> dict[str, str]:
    """Do not hand model credentials or plugin approval switches to test code."""
    from codex_agent.runtime_config import CONFIG_ENV, runtime_config
    settings = runtime_config()
    credential = settings.memory.embedding.apiKeyEnv
    database_credential = settings.storage.urlEnv
    cache_credential = settings.cache.urlEnv
    queue_credential = settings.queue.urlEnv
    return {key: value for key, value in os.environ.items()
            if not key.startswith("TRITON_RISCV_ALLOW_")
            and not key.endswith(("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "PASSWORD"))
            and key not in {CONFIG_ENV, credential, database_credential, cache_credential, queue_credential,
                           "TRITON_AMQP_URL",
                           "TRITON_REDIS_URL", "TRITON_MYSQL_URL", "SSH_AUTH_SOCK", "GH_TOKEN", "GITHUB_TOKEN"}}


@contextmanager
def execution_budget(seconds: float, cancel_path: Path | None = None):
    previous = _deadline.get()
    deadline = time.monotonic() + seconds
    token = _deadline.set(min(deadline, previous) if previous is not None else deadline)
    cancel = _cancel.set((*(_cancel.get() or ()), *((cancel_path,) if cancel_path else ())))
    try:
        yield
    finally:
        _deadline.reset(token)
        _cancel.reset(cancel)


def remaining(seconds: float) -> float:
    for path in _cancel.get() or ():
        if path.exists():
            raise ExecutionCancelled("Host cancelled execution; inspect the journal before any retry")
    deadline = _deadline.get()
    value = seconds if deadline is None else min(seconds, deadline - time.monotonic())
    if value <= 0:
        raise subprocess.TimeoutExpired("operation wall-clock budget", seconds)
    return value


def run_bounded(args, *, timeout, check=False, input=None, **kwargs):
    """Like subprocess.run, but cancellation/timeout also kills child processes."""
    seconds = remaining(timeout)
    end = time.monotonic() + seconds
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    first = True
    try:
        while True:
            remaining(timeout)
            left = end - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired(args, seconds)
            try:
                out, err = process.communicate(input=input if first else None, timeout=min(left, 0.2))
                break
            except subprocess.TimeoutExpired:
                first = False
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            out = err = None
        finally:
            # The group may still have grandchildren after its leader exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        if isinstance(error, subprocess.TimeoutExpired):
            error.output, error.stderr = out, err
        raise
    result = subprocess.CompletedProcess(args, process.returncode, out, err)
    if check:
        result.check_returncode()
    return result
