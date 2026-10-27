"""One bounded role invocation through Harness, with no workflow side effects."""
from dataclasses import dataclass

from dsh_multi_agent import IsolatedHarnessBackend, TaskDefinition, TaskRunner
from dsh_multi_agent.profile import composition as _composition

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
    usage: dict | None = None


POLICY_IDENTITY = dict(variable="triton_role_policy", section="triton:role", effect="triton.role",
                       denial="Proposal roles cannot execute tools")


def composition(settings, policy, session_root):
    return _composition(settings, policy, session_root, policy_identity=POLICY_IDENTITY)


class ProposalHarnessBackend(IsolatedHarnessBackend):
    """Fresh SDK process/session per call. No domain MCP, shell or file tools."""
    def __init__(self, settings):
        super().__init__(settings, profile_name="triton", policy_identity=POLICY_IDENTITY)

    def run(self, policy, prompt, *, session_id, on_event=None):
        raw = super().run(policy, prompt, session_id=session_id, on_event=on_event)
        return HarnessRunResult(raw.session_id, raw.final_response, raw.finish_reason, raw.events)


class RoleAgent:
    """Returns validated suggestions; never advances or approves a parent task."""
    def __init__(self, settings=None, *, backend=None):
        if backend is None and settings is None:
            raise ValueError("Harness settings or an explicit test backend is required")
        self.backend = backend if backend is not None else ProposalHarnessBackend(settings)

    def run(self, request, *, on_event=None, on_usage=None):
        # Revalidate even prebuilt objects, so mutated nested fields cannot bypass checks.
        request = RoleRequest.model_validate(request.model_dump() if isinstance(request, RoleRequest) else request)
        policy, prompt = system_prompt(request.role), task_prompt(request)
        binding = dict(task_id=request.task_id, step_id=request.step_id, input_epoch=request.input_epoch,
                       role=request.role, prompt_version=PROMPT_VERSION, prompt_sha256=sha256(policy + "\n" + prompt))
        task = TaskDefinition(policy, prompt, lambda text: parse_result(request, text),
                              rejection_message="output failed role schema or evidence/version checks")

        def failed_usage(usage):
            if on_usage is not None and usage["outcome"] in {"error", "cancelled"}:
                on_usage(usage)

        result = TaskRunner(self.backend, session_prefix="role-").run(
            task, on_event=on_event, on_usage=failed_usage)
        usage = result.usage
        usage["role_status"] = result.result.status if result.result is not None else None
        if on_usage is not None:
            on_usage(usage)
        return RoleRun(**binding, outcome=result.outcome, result=result.result,
                       harness=result.harness, error=result.error, usage=usage)
