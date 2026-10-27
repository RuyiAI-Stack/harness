"""Fresh SDK process, private profile, workspace and session store for every run."""
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .profile import composition, write_profile


@dataclass(frozen=True)
class HarnessRunResult:
    session_id: str
    final_response: str
    finish_reason: str | None = None
    events: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str | None
    api_key_env: str
    api_key: str | None = field(default=None, repr=False)
    request_timeout_seconds: float = 900.0

    def validate_live_run(self):
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("A provider and model are required")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env) or not self.api_key:
            raise ValueError("A credential environment name and value are required")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive and finite")


class IsolatedHarnessBackend:
    """No domain MCP, shell, filesystem tools, parent history or implicit context."""
    def __init__(self, settings, *, profile_name="isolated-task", policy_identity=None):
        self.settings = settings
        self.profile_name = profile_name
        self.policy_identity = policy_identity

    def run(self, policy, prompt, *, session_id, on_event=None):
        from deepseek_harness import DeepSeekHarness
        self.settings.validate_live_run()
        with tempfile.TemporaryDirectory(prefix="dsh-isolated-task-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            profile = write_profile(root / "home", composition(
                self.settings, policy, root / "sessions", policy_identity=self.policy_identity), name=self.profile_name)

            def notification(event):
                if on_event is not None:
                    on_event({"method": event.method, "payload": dict(event.payload)})

            runtime = DeepSeekHarness(provider=self.settings.provider, model=self.settings.model,
                cwd=str(workspace), runtime_cwd=str(workspace), **profile,
                env={self.settings.api_key_env: self.settings.api_key},
                request_timeout_seconds=self.settings.request_timeout_seconds)
            try:
                result = runtime.run(prompt, session_id=session_id, on_notification=notification)
                return HarnessRunResult(result.session_id, result.final_response, result.finish_reason, list(result.events))
            finally:
                runtime.close()
