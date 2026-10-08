"""Bounded Triton-RISCV task context supplied to DeepSeek Harness."""

from __future__ import annotations

from codex_agent.harness.config import HarnessSettings


def build_task_prompt(user_request: str, settings: HarnessSettings) -> str:
    """Build a request envelope; the plugin owns stable lifecycle policy."""

    request = user_request.strip()
    if not request:
        raise ValueError("user request cannot be empty")

    remote = "not configured"
    if settings.remote_host and settings.remote_root:
        remote = f"{settings.remote_host}:{settings.remote_root}"

    return f"""Triton-RISCV task for the {settings.plugin.name} plugin.

User request:
{request}

Workspace:
- repository: {settings.repo_root}
- RISC-V validation host: {remote}

The stable lifecycle policy is supplied by the loaded Harness plugin. Treat
this envelope only as the current task and runtime location.
"""
