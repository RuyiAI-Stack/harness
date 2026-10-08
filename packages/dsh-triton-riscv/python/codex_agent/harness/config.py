"""Environment-backed configuration for the DeepSeek Harness control plane."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from codex_agent.harness.plugin import HarnessPluginDescriptor, default_plugin_root
from codex_agent.paths import state_root


def _optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class HarnessSettings:
    """Configuration that is safe to pass to the official Harness SDK."""

    repo_root: Path
    session_root: Path
    cordis_path: Path
    plugin: HarnessPluginDescriptor
    mcp_python: str
    provider: str = "isrc-proxy"
    model: str = "gpt-5.6-sol"
    base_url: Optional[str] = "https://llmapi.isrc.ac.cn/v1"
    api_key_env: str = "ISRC_API_KEY"
    api_key: Optional[str] = field(default=None, repr=False)
    remote_host: Optional[str] = None
    remote_root: Optional[str] = None
    request_timeout_seconds: float = 900.0
    managed_context: bool = True
    context_window_tokens: int = 16_384
    context_safety_ratio: float = 0.80
    context_output_reserve_tokens: int = 2_048
    context_recent_turns: int = 4
    context_summary_mode: str = "model"

    @classmethod
    def from_env(
        cls,
        repo_root: Path,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "HarnessSettings":
        env = os.environ if environ is None else environ
        resolved_root = repo_root.resolve()
        session_root = Path(
            env.get(
                "DSH_SESSION_ROOT",
                Path(env.get("TRITON_RISCV_STATE_DIR") or resolved_root / "agent-results") / "deepseek-harness",
            )
        ).expanduser()
        cordis_path = Path(
            env.get(
                "TRITON_RISCV_CORDIS_CONFIG",
                Path(__file__).with_name("cordis.yml"),
            )
        ).expanduser()
        if not cordis_path.is_absolute():
            cordis_path = resolved_root / cordis_path
        plugin_root = Path(
            env.get("TRITON_RISCV_HARNESS_PLUGIN_ROOT", default_plugin_root())
        ).expanduser()
        if not plugin_root.is_absolute():
            plugin_root = resolved_root / plugin_root
        api_key_env = env.get("DSH_API_KEY_ENV", "ISRC_API_KEY").strip()
        if not api_key_env:
            raise ValueError("DSH_API_KEY_ENV cannot be empty")
        return cls(
            repo_root=resolved_root,
            session_root=session_root.resolve(),
            cordis_path=cordis_path.resolve(),
            plugin=HarnessPluginDescriptor.load(plugin_root),
            mcp_python=env.get("TRITON_RISCV_MCP_PYTHON", sys.executable),
            provider=env.get("DSH_PROVIDER", "isrc-proxy").strip(),
            model=env.get("DSH_MODEL", "gpt-5.6-sol").strip(),
            base_url=_optional_text(
                env.get("ISRC_BASE_URL", "https://llmapi.isrc.ac.cn/v1")
            ),
            api_key_env=api_key_env,
            api_key=_optional_text(env.get(api_key_env)),
            remote_host=_optional_text(env.get("RISCV_HOST")),
            remote_root=_optional_text(env.get("RISCV_REPO")),
            request_timeout_seconds=float(
                env.get("DSH_REQUEST_TIMEOUT_SECONDS", "900")
            ),
            managed_context=env.get("TRITON_RISCV_MANAGED_CONTEXT", "1") == "1",
            context_window_tokens=int(env.get("TRITON_RISCV_CONTEXT_WINDOW_TOKENS", "16384")),
            context_safety_ratio=float(env.get("TRITON_RISCV_CONTEXT_SAFETY_RATIO", "0.80")),
            context_output_reserve_tokens=int(env.get("TRITON_RISCV_CONTEXT_OUTPUT_RESERVE", "2048")),
            context_recent_turns=int(env.get("TRITON_RISCV_CONTEXT_RECENT_TURNS", "4")),
            context_summary_mode=env.get("TRITON_RISCV_CONTEXT_SUMMARY_MODE", "model"),
        )

    def validate_runtime_config(self) -> None:
        if not 0.5 <= self.context_safety_ratio <= 0.95 or self.context_recent_turns < 1:
            raise ValueError("invalid conversation context budget")
        if self.context_summary_mode not in {"model", "extractive"}:
            raise ValueError("invalid context summary mode")
        if not self.repo_root.is_dir():
            raise ValueError(f"repository does not exist: {self.repo_root}")
        if not self.cordis_path.is_file():
            raise ValueError(f"Cordis config does not exist: {self.cordis_path}")
        HarnessPluginDescriptor.load(self.plugin.root)
        if not shutil.which(self.mcp_python):
            raise ValueError(f"MCP Python executable was not found: {self.mcp_python}")
        if not self.provider:
            raise ValueError("DSH_PROVIDER cannot be empty")
        if not self.model:
            raise ValueError("DSH_MODEL cannot be empty")
        if bool(self.remote_host) != bool(self.remote_root):
            raise ValueError("RISCV_HOST and RISCV_REPO must be configured together")

    def validate_live_run(self) -> None:
        self.validate_runtime_config()
        if not self.api_key:
            raise ValueError(
                f"{self.api_key_env} is required for a live Harness run"
            )
