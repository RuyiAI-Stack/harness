"""Separate installed program resources from the repository being edited."""
from __future__ import annotations

from pathlib import Path
from codex_agent.runtime_config import runtime_config


def repository_root() -> Path:
    value = runtime_config().repoRoot
    return Path(value).expanduser().resolve() if value else Path.cwd().resolve()


def state_root(repo_root: Path) -> Path:
    value = runtime_config().stateDir
    return Path(value).expanduser().resolve() if value else repo_root.resolve() / "agent-results"
