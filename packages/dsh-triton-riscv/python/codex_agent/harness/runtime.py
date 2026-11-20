"""Small boundary around the official DeepSeek Harness Python SDK."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.context import build_task_prompt
from codex_agent.harness.sdk_profile import kernel, stage_composition, write_profile


class HarnessUnavailableError(RuntimeError):
    """Raised when the optional Harness runtime cannot be started."""


@dataclass(frozen=True)
class HarnessRunResult:
    session_id: str
    final_response: str
    finish_reason: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)


HarnessEventCallback = Callable[[dict[str, Any]], None]


class HarnessBackend(Protocol):
    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[HarnessEventCallback] = None,
    ) -> HarnessRunResult:
        ...

    def close(self) -> None:
        ...


class DeepSeekHarnessBackend:
    """Lazy adapter so normal agent tests do not require the optional SDK."""

    def __init__(self, settings: HarnessSettings) -> None:
        self.settings = settings
        self._harness: Any = None
        self._home = None
        self._lock = threading.RLock()

    def _runtime(self) -> Any:
        if self._harness is not None:
            return self._harness

        try:
            from deepseek_harness import DeepSeekHarness
        except ImportError as error:
            raise HarnessUnavailableError(
                "DeepSeek Harness SDK is not installed. Use Python 3.10+ and "
                "install dsh-triton-riscv-agent[workbench] in the plugin environment."
            ) from error

        runtime_env = {
            "TRITON_RISCV_REPO_ROOT": str(self.settings.repo_root),
            "TRITON_RISCV_WORKBENCH_AUTOSTART": "0",
            "DSH_API_KEY_ENV": self.settings.api_key_env,
            "DSH_MODEL": self.settings.model,
            "DSH_CWD": str(self.settings.repo_root),
            "DSH_SESSION_ROOT": str(self.settings.session_root),
            "TRITON_RISCV_MCP_PYTHON": self.settings.mcp_python,
            "TRITON_RISCV_ALLOW_VALIDATION": os.environ.get(
                "TRITON_RISCV_ALLOW_VALIDATION", "0"
            ),
            "TRITON_RISCV_ALLOW_REPAIR_APPLY": os.environ.get(
                "TRITON_RISCV_ALLOW_REPAIR_APPLY", "0"
            ),
            "TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY": os.environ.get(
                "TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY", "0"
            ),
            "TRITON_RISCV_REQUIRE_REMOTE": os.environ.get(
                "TRITON_RISCV_REQUIRE_REMOTE", "0"
            ),
            "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": os.environ.get(
                "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION", "1"
            ),
        }
        from codex_agent.runtime_config import CONFIG_ENV, runtime_config
        config = runtime_config()
        credential_keys = [config.storage.urlEnv, config.cache.urlEnv, config.queue.urlEnv,
                           config.memory.embedding.apiKeyEnv]
        from codex_agent.accounts.guard import OWNER_ENV, check_workspace
        check_workspace(self.settings.repo_root)
        if os.environ.get(OWNER_ENV):
            runtime_env[OWNER_ENV] = os.environ[OWNER_ENV]
        if config.indexing.enabled:
            credential_keys.extend([config.indexing.apiKeyEnv, config.indexing.embedding.apiKeyEnv])
        for key in (CONFIG_ENV, *credential_keys):
            if os.environ.get(key):
                runtime_env[key] = os.environ[key]
        for name, value in os.environ.items():
            if name.startswith(("TRITON_RISCV_MEMORY_", "TRITON_RISCV_EMBEDDING_")) or name in {
                "TRITON_RISCV_STATE_DIR", "AGENT_EMBEDDING_API_KEY",
            }:
                runtime_env[name] = value
        if self.settings.remote_host:
            runtime_env["RISCV_HOST"] = self.settings.remote_host
        if self.settings.remote_root:
            runtime_env["RISCV_REPO"] = self.settings.remote_root
        if self.settings.api_key:
            runtime_env[self.settings.api_key_env] = self.settings.api_key
        if self.settings.base_url:
            runtime_env["ISRC_BASE_URL"] = self.settings.base_url

        self._home = tempfile.TemporaryDirectory(prefix="triton-chat-sdk-")
        from codex_agent.harness.delegation import SCOPE_ENV, SCOPE
        runtime_env[SCOPE_ENV] = str(Path(self._home.name) / "parallel-scope.json")
        runtime_env["DSH_PROVIDER"] = self.settings.provider
        try:
            profile = write_profile(Path(self._home.name), kernel() + [{
                "id": "parallel-parent-guard", "name": str(Path(__file__).with_name("parallel-policy.mjs")),
                "config": {"typedOnly": SCOPE.get() is not None},
            }, {
                "id": "triton-composition", "name": "cordis:include",
                "config": {"path": (Path(self._home.name) / "profiles/triton/domain.yml").as_uri()},
            }])
            stage_composition(self.settings.cordis_path, self._home.name)
            self._harness = DeepSeekHarness(
                provider=self.settings.provider, model=self.settings.model,
                cwd=str(self.settings.repo_root), **profile, env=runtime_env,
                request_timeout_seconds=self.settings.request_timeout_seconds,
            )
        except Exception:
            self._home.cleanup()
            self._home = None
            raise
        return self._harness

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        on_event: Optional[HarnessEventCallback] = None,
    ) -> HarnessRunResult:
        self.settings.validate_live_run()
        self.settings.session_root.mkdir(parents=True, exist_ok=True)

        try:
            def forward_notification(notification: Any) -> None:
                if on_event is not None:
                    on_event(
                        {
                            "method": notification.method,
                            "payload": dict(notification.payload),
                        }
                    )

            # The official client owns one reusable subprocess. Serializing access
            # preserves session shell state and avoids sharing its stdio transport
            # unsafely across worker threads.
            with self._lock:
                runtime = self._runtime()
                from codex_agent.harness.delegation import SCOPE
                from codex_agent.execution_guard import atomic_json
                scope_path = Path(self._home.name) / "parallel-scope.json"
                scope = SCOPE.get()
                atomic_json(scope_path, {**scope, "base_model": self.settings.routing_base_model or self.settings.model} if scope else {})
                try:
                    result = runtime.run(prompt, session_id=session_id, on_notification=forward_notification)
                finally:
                    atomic_json(scope_path, {})
        except Exception as error:
            raise HarnessUnavailableError(f"DeepSeek Harness run failed: {error}") from error

        return HarnessRunResult(
            session_id=result.session_id,
            final_response=result.final_response,
            finish_reason=result.finish_reason,
            events=list(result.events),
        )

    def close(self) -> None:
        with self._lock:
            if self._harness is not None:
                try:
                    self._harness.close()
                finally:
                    self._harness = None
                    if self._home is not None:
                        self._home.cleanup()
                        self._home = None


class HarnessAgent:
    """Domain-facing application service independent of a concrete backend."""

    def __init__(
        self,
        settings: HarnessSettings,
        backend: Optional[HarnessBackend] = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or DeepSeekHarnessBackend(settings)
        self._custom_backend = backend is not None
        self._backends = {}
        self._routing_lock = threading.Lock()

    def prepare(self, user_request: str) -> str:
        return build_task_prompt(user_request, self.settings)

    def run(
        self,
        user_request: str,
        session_id: Optional[str] = None,
        on_event: Optional[HarnessEventCallback] = None,
    ) -> HarnessRunResult:
        from codex_agent.harness.model_routing import select_route, routed_settings
        from codex_agent.harness.delegation import SCOPE
        prompt = self.prepare(user_request)
        raw_request = (SCOPE.get() or {}).get("user_request", user_request)
        route = select_route(self.settings, raw_request, prompt)
        # A parent turn may await several child batches or a validation tool.
        # Preserve its outer request deadline; per-role/child limits stay separate.
        route["timeout_seconds"] = self.settings.request_timeout_seconds
        if not self.settings.managed_context:
            # SDK sessions cannot move between processes without restoring their log.
            # The unmanaged legacy path keeps its existing model and live history.
            route.update(tier="default", model=self.settings.model, reason="unmanaged_session_keeps_model")
        backend = self.backend
        if self._custom_backend and route["tier"] != "default":
            raise ValueError("Automatic routing requires the managed Harness backend")
        if route["tier"] != "default" and not self._custom_backend:
            with self._routing_lock:
                key = (route["model"], route["timeout_seconds"], route["policy_fingerprint"])
                if key not in self._backends:
                    self._backends[key] = DeepSeekHarnessBackend(routed_settings(self.settings, route))
                backend = self._backends[key]
        if on_event is not None:
            on_event({"method": "model.route", "payload": {**route, "scope": "chat"}})
        return backend.run(
            prompt,
            session_id=session_id,
            on_event=on_event,
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
