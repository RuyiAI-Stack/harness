"""Typed boundaries between orchestration policy and external capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    repo_root: Path
    run_dir: Path
    label: str
    timeout_seconds: int


@dataclass(frozen=True)
class ModelResponse:
    status: str
    exit_code: int
    duration_seconds: float = 0.0
    reason: str | None = None
    log_path: str | None = None
    final_message_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "status": self.status,
                "exit_code": self.exit_code,
                "duration_seconds": self.duration_seconds,
                "reason": self.reason,
                "log_path": self.log_path,
                "final_message_path": self.final_message_path,
            }.items()
            if value is not None
        }


class ModelGateway(Protocol):
    """Port implemented by Codex CLI or a future company model API."""

    name: str

    def invoke(self, request: ModelRequest) -> ModelResponse:
        """Execute one bounded model task."""


class MemoryPort(Protocol):
    """Port implemented by SQLite memory and future remote/vector stores."""

    def retrieve(self, query: Any, limit: int = 5) -> list[dict]:
        """Retrieve evidence records for one context."""

    def add(self, record: Any) -> tuple[int, bool]:
        """Store one evidence-gated record."""

    def stats(self) -> dict:
        """Return auditable memory statistics."""


class ValidationPort(Protocol):
    """Port for local, SSH, QEMU, or future MCP-backed validation."""

    def validate(self, request: Any) -> dict:
        """Run validation and return structured stage evidence."""
