"""Explicit, persistent state machine for operator development workflows."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowPhase(str, Enum):
    INITIALIZED = "initialized"
    CONTEXT_READY = "context-ready"
    ENVIRONMENT_READY = "environment-ready"
    GENERATING = "generating"
    CONTRACT_AUDIT = "contract-audit"
    CONTRACT_READY = "contract-ready"
    VALIDATING = "validating"
    DIAGNOSING = "diagnosing"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SAFETY_BLOCKED = "safety-blocked"
    FAILED = "failed"


TERMINAL_PHASES = {
    WorkflowPhase.COMPLETED,
    WorkflowPhase.BLOCKED,
    WorkflowPhase.SAFETY_BLOCKED,
    WorkflowPhase.FAILED,
}

ALLOWED_TRANSITIONS = {
    WorkflowPhase.INITIALIZED: {
        WorkflowPhase.CONTEXT_READY,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.CONTEXT_READY: {
        WorkflowPhase.ENVIRONMENT_READY,
        WorkflowPhase.COMPLETED,
        WorkflowPhase.BLOCKED,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.ENVIRONMENT_READY: {
        WorkflowPhase.GENERATING,
        WorkflowPhase.VALIDATING,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.GENERATING: {
        WorkflowPhase.GENERATING,
        WorkflowPhase.CONTRACT_AUDIT,
        WorkflowPhase.SAFETY_BLOCKED,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.CONTRACT_AUDIT: {
        WorkflowPhase.GENERATING,
        WorkflowPhase.CONTRACT_READY,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.CONTRACT_READY: {
        WorkflowPhase.VALIDATING,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.VALIDATING: {
        WorkflowPhase.DIAGNOSING,
        WorkflowPhase.COMPLETED,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.DIAGNOSING: {
        WorkflowPhase.REPAIRING,
        WorkflowPhase.BLOCKED,
        WorkflowPhase.FAILED,
    },
    WorkflowPhase.REPAIRING: {
        WorkflowPhase.REPAIRING,
        WorkflowPhase.VALIDATING,
        WorkflowPhase.DIAGNOSING,
        WorkflowPhase.SAFETY_BLOCKED,
        WorkflowPhase.FAILED,
    },
}


class WorkflowTransitionError(RuntimeError):
    pass


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class WorkflowState:
    workflow_id: str
    operator: str
    phase: str = WorkflowPhase.INITIALIZED.value
    sequence: int = 0
    generation_attempt: int = 0
    validation_iteration: int = 0
    repair_attempt: int = 0
    failure_stage: str | None = None
    terminal_status: str | None = None
    memory_ids: list[int] = field(default_factory=list)
    updated_at: str = field(default_factory=timestamp)


class WorkflowSession:
    """State transition authority and append-only event journal."""

    def __init__(self, run_dir: Path, workflow_id: str, operator: str) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "workflow-state.json"
        self.events_path = run_dir / "workflow-events.jsonl"
        self.state = WorkflowState(workflow_id=workflow_id, operator=operator)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_state()
        self._append_event(
            from_phase=None,
            to_phase=WorkflowPhase.INITIALIZED,
            reason="workflow session created",
            data={},
        )

    @property
    def phase(self) -> WorkflowPhase:
        return WorkflowPhase(self.state.phase)

    def transition(
        self,
        to_phase: WorkflowPhase,
        reason: str,
        *,
        data: dict[str, Any] | None = None,
        generation_attempt: int | None = None,
        validation_iteration: int | None = None,
        repair_attempt: int | None = None,
        failure_stage: str | None = None,
        terminal_status: str | None = None,
        memory_ids: list[int] | None = None,
    ) -> None:
        current = self.phase
        if current in TERMINAL_PHASES:
            raise WorkflowTransitionError(
                f"cannot transition terminal workflow from {current.value}"
            )
        allowed = ALLOWED_TRANSITIONS.get(current, set())
        if to_phase not in allowed:
            raise WorkflowTransitionError(
                f"invalid workflow transition {current.value} -> {to_phase.value}"
            )
        if generation_attempt is not None:
            self.state.generation_attempt = generation_attempt
        if validation_iteration is not None:
            self.state.validation_iteration = validation_iteration
        if repair_attempt is not None:
            self.state.repair_attempt = repair_attempt
        if failure_stage is not None:
            self.state.failure_stage = failure_stage
        if terminal_status is not None:
            self.state.terminal_status = terminal_status
        if memory_ids is not None:
            self.state.memory_ids = list(memory_ids)
        self.state.sequence += 1
        self.state.phase = to_phase.value
        self.state.updated_at = timestamp()
        self._append_event(current, to_phase, reason, data or {})
        self._write_state()

    def _write_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(self.state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _append_event(
        self,
        from_phase: WorkflowPhase | None,
        to_phase: WorkflowPhase,
        reason: str,
        data: dict[str, Any],
    ) -> None:
        event = {
            "schema_version": 1,
            "sequence": self.state.sequence,
            "timestamp": timestamp(),
            "workflow_id": self.state.workflow_id,
            "operator": self.state.operator,
            "from_phase": from_phase.value if from_phase else None,
            "to_phase": to_phase.value,
            "reason": reason,
            "data": data,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
