"""One typed invocation, independent of roles, operators or a parent workflow."""
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Generic, TypeVar

from .backend import HarnessRunResult
from .usage import RunUsage

T = TypeVar("T")


class RunCancelled(Exception):
    """Host cancellation; callers must not interpret partial output as a result."""


@dataclass(frozen=True)
class TaskDefinition(Generic[T]):
    system_prompt: str
    prompt: str
    validate_result: Callable[[str], T]
    rejection_message: str = "output failed task validation"

    def __post_init__(self):
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("A task system prompt is required")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("A task prompt is required")
        if not callable(self.validate_result):
            raise TypeError("A result validator is required")


@dataclass(frozen=True)
class TaskRun(Generic[T]):
    session_id: str
    outcome: str
    result: T | None
    harness: HarnessRunResult
    error: str | None = None
    usage: dict | None = None


def normalize_result(task, session_id, raw):
    if raw.session_id != session_id or raw.finish_reason != "completed":
        return TaskRun(session_id, "incomplete", None, raw,
                       "Harness session or completion did not match this invocation")
    try:
        result = task.validate_result(raw.final_response)
    except (ValueError, TypeError) as error:
        return TaskRun(session_id, "invalid_output", None, raw,
                       type(error).__name__ + ": " + task.rejection_message)
    return TaskRun(session_id, "validated", result, raw)


class TaskRunner:
    def __init__(self, backend, *, session_prefix="task-"):
        self.backend = backend
        self.session_prefix = session_prefix

    def run(self, task, *, on_event=None, on_usage=None, on_started=None):
        session_id = self.session_prefix + uuid.uuid4().hex
        settings = getattr(self.backend, "settings", None)
        meter = RunUsage(session_id, provider=getattr(settings, "provider", None),
                         model=getattr(settings, "model", None),
                         prompt_characters=len(task.system_prompt) + len(task.prompt))

        def observe(event):
            meter.observe(event)
            if on_event is not None:
                on_event(event)

        try:
            if on_started is not None:
                on_started(session_id)
            raw = self.backend.run(task.system_prompt, task.prompt, session_id=session_id, on_event=observe)
            if raw.session_id == session_id:
                for event in raw.events:
                    meter.observe(event)
            else:
                meter.issues.add("foreign_result_session")
            result = normalize_result(task, session_id, raw)
        except Exception as error:
            if on_usage is not None:
                on_usage(meter.finish("cancelled" if isinstance(error, RunCancelled) else "error"))
            raise
        usage = meter.finish(result.outcome)
        if on_usage is not None:
            on_usage(usage)
        return replace(result, usage=usage)
