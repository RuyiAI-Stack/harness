"""One bounded role invocation through Harness, with no workflow side effects."""
import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from codex_agent.harness.runtime import HarnessRunResult
from .prompts import PROMPT_VERSION, system_prompt, task_prompt
from .roles import CommonResult, RoleRequest, parse_result, sha256


@dataclass(frozen=True)
class RoleRun:
    task_id: str
    step_id: str
    input_epoch: int
    role: str
    prompt_version: str
    prompt_sha256: str
    outcome: str
    result: CommonResult | None
    harness: HarnessRunResult
    error: str | None = None


def composition(settings, policy, session_root):
    return [
        {"id": "sdk-jsonrpc-server", "name": "@deepseek-ai/dsh-sdk-jsonrpc-server"},
        {"id": "agent-core", "name": "@deepseek-ai/dsh-agent-spine-demo",
         "config": {"workspaceContext": {"maxBytes": 0}}},
        {"id": "role-model", "name": "@deepseek-ai/dsh-llm-pi-ai", "config": {"providers": {
            settings.provider: {"apiKeyEnv": settings.api_key_env, "api": "openai-responses",
                "baseURL": settings.base_url, "models": [{"id": settings.model, "name": settings.model}]}}}},
        {"id": "sessions", "name": "@deepseek-ai/dsh-session-persistence-jsonl", "config": {"root": str(session_root)}},
        {"id": "checkpoints", "name": "@deepseek-ai/dsh-session-checkpoint-policy"},
        {"id": "role-policy", "name": str(Path(__file__).with_name("role-policy.mjs")), "config": {"systemPrompt": policy}},
    ]


class ProposalHarnessBackend:
    """Fresh SDK process/session per call. No domain MCP, shell or file tools."""
    def __init__(self, settings):
        self.settings = settings

    def run(self, policy, prompt, *, session_id, on_event=None):
        from deepseek_harness import DeepSeekHarness
        self.settings.validate_live_run()
        # Never expose the operator checkout, parent session history or workspace
        # instructions to a child role through implicit Harness context loading.
        with tempfile.TemporaryDirectory(prefix="triton-role-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            cordis = root / "cordis.yml"
            cordis.write_text(json.dumps(composition(self.settings, policy, root / "sessions")), encoding="utf-8")
            def notification(event):
                if on_event is not None:
                    on_event({"method": event.method, "payload": dict(event.payload)})
            runtime = DeepSeekHarness(provider=self.settings.provider, model=self.settings.model,
                cwd=str(workspace), runtime_cwd=str(workspace), session_root=str(root / "sessions"),
                cordis=str(cordis), env={self.settings.api_key_env: self.settings.api_key},
                request_timeout_seconds=self.settings.request_timeout_seconds)
            try:
                result = runtime.run(prompt, session_id=session_id, on_notification=notification)
                return HarnessRunResult(result.session_id, result.final_response, result.finish_reason, list(result.events))
            finally:
                runtime.close()


class RoleAgent:
    """Returns validated suggestions; never advances or approves a parent task."""
    def __init__(self, settings=None, *, backend=None):
        if backend is None and settings is None:
            raise ValueError("Harness settings or an explicit test backend is required")
        self.backend = backend if backend is not None else ProposalHarnessBackend(settings)

    def run(self, request, *, on_event=None):
        # Revalidate even prebuilt objects, so mutated nested fields cannot bypass checks.
        request = RoleRequest.model_validate(request.model_dump() if isinstance(request, RoleRequest) else request)
        policy, prompt = system_prompt(request.role), task_prompt(request)
        session_id = "role-" + uuid.uuid4().hex
        binding = dict(task_id=request.task_id, step_id=request.step_id, input_epoch=request.input_epoch,
                       role=request.role, prompt_version=PROMPT_VERSION, prompt_sha256=sha256(policy + "\n" + prompt))
        raw = self.backend.run(policy, prompt, session_id=session_id, on_event=on_event)
        if raw.session_id != session_id or raw.finish_reason != "completed":
            return RoleRun(**binding, outcome="incomplete", result=None, harness=raw,
                           error="Harness session or completion did not match this invocation")
        try:
            result = parse_result(request, raw.final_response)
        except (ValueError, TypeError) as error:
            # Keep raw output/events for caller-owned audit, but don't mistake it for a proposal.
            return RoleRun(**binding, outcome="invalid_output", result=None, harness=raw,
                           error=type(error).__name__ + ": output failed role schema or evidence/version checks")
        return RoleRun(**binding, outcome="validated", result=result, harness=raw)
