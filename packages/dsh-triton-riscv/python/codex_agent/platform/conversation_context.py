"""Application-managed context assembly for long Harness conversations.

The platform database remains the source of truth.  This module only builds a
bounded model-facing view from immutable task state, a rolling summary, recent
raw turns, prior tool outcomes, and optional RAG evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Callable, Protocol, Sequence

from codex_agent.model_router import estimate_tokens


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ContextPolicy:
    # 16K is conservative under the current Harness workspaceContext.maxBytes
    # value of 64 KiB, including Chinese text and unbroken compiler logs.
    max_input_tokens: int = 16_384
    safety_ratio: float = 0.80
    output_reserve_tokens: int = 2_048
    recent_turns: int = 4
    max_tool_result_tokens: int = 6_000
    max_rag_tokens: int = 6_000

    @property
    def input_budget(self) -> int:
        safe = int(self.max_input_tokens * self.safety_ratio)
        return max(1_024, safe - self.output_reserve_tokens)


@dataclass(frozen=True)
class SummaryResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "deterministic"


class ConversationSummarizer(Protocol):
    def summarize(
        self,
        previous_summary: str,
        messages: Sequence[dict],
        anchors: Sequence[str],
    ) -> SummaryResult:
        ...


@dataclass(frozen=True)
class ContextBundle:
    prompt: str
    summary: str
    summarized_message_ids: tuple[str, ...]
    recent_message_ids: tuple[str, ...]
    input_budget: int
    estimated_tokens: int
    compacted: bool
    summary_usage: dict = field(default_factory=dict)


def _message_text(message: dict) -> str:
    role = str(message.get("role", "unknown")).upper()
    content = str(message.get("content", "")).strip()
    return f"[{role}] {content}" if content else ""


def _anchor_candidates(text: str) -> list[str]:
    patterns = (
        r"\b(?:run|proposal|development|session)-[A-Za-z0-9_-]+\b",
        r"(?:^|\s)(/[A-Za-z0-9_./-]+|[A-Za-z0-9_./-]+\.(?:py|json|jsonl|log|md|sqlite3))",
        r"`([^`]*(?:python|pytest|ssh|scp|cmake|ninja)[^`]*)`",
        r"\b(?:passed|failed|blocked|approved|rejected|pending)\b",
    )
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I | re.M):
            value = next((group for group in match.groups() if group), match.group(0)).strip()
            if value and value not in found:
                found.append(value[:300])
            if len(found) >= 128:
                return found
    return found


def _retain_anchors(summary: str, anchors: Sequence[str]) -> str:
    missing = [anchor for anchor in anchors if anchor not in summary]
    if not missing:
        return summary.strip()
    suffix = "\n\nRetained source anchors:\n" + "\n".join(f"- {item}" for item in missing)
    return summary.strip() + suffix


class ExtractiveConversationSummarizer:
    """Safe local fallback that never invents facts or executes commands."""

    def summarize(
        self,
        previous_summary: str,
        messages: Sequence[dict],
        anchors: Sequence[str],
    ) -> SummaryResult:
        rows = [
            "Objective and prior state:",
            previous_summary.strip() or "- No previous summary.",
            "Decisions and evidence from compacted turns:",
        ]
        for message in messages:
            text = " ".join(str(message.get("content", "")).split())
            if text:
                rows.append(
                    f"- {str(message.get('role', 'unknown'))} "
                    f"({message.get('id', 'unknown')}): {text[:900]}"
                )
        summary = _retain_anchors("\n".join(rows), anchors)
        return SummaryResult(
            text=summary,
            input_tokens=sum(estimate_tokens(_message_text(item)) for item in messages),
            output_tokens=estimate_tokens(summary),
        )


class OpenAIConversationSummarizer:
    """OpenAI-compatible structured summarizer used only after the threshold."""

    def __init__(self, *, base_url: str | None, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def summarize(
        self,
        previous_summary: str,
        messages: Sequence[dict],
        anchors: Sequence[str],
    ) -> SummaryResult:
        from openai import OpenAI

        payload = {
            "previous_summary": previous_summary,
            "messages": [
                {"id": item.get("id"), "role": item.get("role"), "content": item.get("content")}
                for item in messages
            ],
            "anchors_that_must_be_retained": list(anchors),
        }
        instruction = (
            "Compress the conversation evidence without adding facts. Use headings: Objective, "
            "Constraints, Decisions, Artifacts and IDs, Verified results, Unresolved items. "
            "Distinguish verified execution from suggestions. Preserve every supplied anchor "
            "verbatim and mention source message IDs. Return plain text only."
        )
        response = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=90,
            max_retries=0,
        ).responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        usage = response.usage.model_dump(exclude_none=False) if response.usage else {}
        summary = _retain_anchors(response.output_text, anchors)
        return SummaryResult(
            text=summary,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            provider=f"openai-compatible:{self.model}",
        )


class ConversationContextManager:
    def __init__(
        self,
        policy: ContextPolicy | None = None,
        *,
        summarizer: ConversationSummarizer | None = None,
        token_counter: TokenCounter = estimate_tokens,
    ) -> None:
        self.policy = policy or ContextPolicy()
        self.summarizer = summarizer or ExtractiveConversationSummarizer()
        self.count_tokens = token_counter

    @staticmethod
    def _visible_messages(messages: Sequence[dict]) -> list[dict]:
        return [
            item for item in messages
            if item.get("content")
            and (item.get("metadata") or {}).get("view") != "harness-plan"
        ]

    def _recent_split(self, messages: Sequence[dict]) -> tuple[list[dict], list[dict]]:
        keep = max(2, self.policy.recent_turns * 2)
        if len(messages) <= keep:
            return [], list(messages)
        return list(messages[:-keep]), list(messages[-keep:])

    def _bounded(self, text: str, budget: int) -> str:
        if self.count_tokens(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_tokens(text[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip() + "\n[truncated to context budget]"

    def _bounded_tail(self, text: str, budget: int) -> str:
        if self.count_tokens(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_tokens(text[-middle:]) <= budget:
                low = middle
            else:
                high = middle - 1
        return "[older recent text truncated]\n" + text[-low:].lstrip()

    def build(
        self,
        *,
        current_request: str,
        messages: Sequence[dict],
        previous_summary: str = "",
        task_state: str = "",
        tool_results: str = "",
        rag_context: str = "",
    ) -> ContextBundle:
        visible = self._visible_messages(messages)
        old, recent = self._recent_split(visible)
        summary = previous_summary.strip()
        compacted_ids: list[str] = []
        summary_usage: dict = {}

        raw_history = "\n\n".join(filter(None, (_message_text(item) for item in visible)))
        retained_anchors = _anchor_candidates(previous_summary + "\n" + raw_history)
        fixed = "\n\n".join(filter(None, (task_state.strip(), tool_results.strip(), rag_context.strip())))
        projected = (
            self.count_tokens(current_request)
            + self.count_tokens(raw_history)
            + self.count_tokens(previous_summary)
            + self.count_tokens(fixed)
        )
        should_compact = bool(old) and projected > self.policy.input_budget
        if should_compact:
            old_text = "\n".join(_message_text(item) for item in old)
            anchors = _anchor_candidates(previous_summary + "\n" + old_text)
            try:
                result = self.summarizer.summarize(previous_summary, old, anchors)
            except Exception as error:
                result = ExtractiveConversationSummarizer().summarize(previous_summary, old, anchors)
                summary_usage = {"provider": "deterministic-fallback", "error": type(error).__name__}
            else:
                summary_usage = {
                    "provider": result.provider,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            # Do not trust even a successful model call to preserve identifiers.
            # The application layer enforces anchors independently.
            summary = _retain_anchors(result.text, anchors)
            compacted_ids = [str(item.get("id", "")) for item in old if item.get("id")]

        recent_rows = recent if should_compact else visible
        recent_text = "\n\n".join(filter(None, (_message_text(item) for item in recent_rows)))
        budget = self.policy.input_budget
        sections = [
            "Current user request:\n" + self._bounded(current_request.strip(), int(budget * 0.12)),
            "Immutable task state:\n" + self._bounded(
                task_state.strip() or "No explicit task state recorded.", int(budget * 0.08)
            ),
            "Retained source anchors:\n" + self._bounded(
                "\n".join(f"- {item}" for item in retained_anchors)
                or "No source anchor detected.",
                int(budget * 0.10),
            ),
            "Rolling summary of older turns:\n" + self._bounded(
                summary or "No older turns were summarized.", int(budget * 0.18)
            ),
            "Recent raw conversation:\n" + self._bounded_tail(
                recent_text or "No recent conversation.", int(budget * 0.30)
            ),
            "Recent tool and execution evidence:\n" + self._bounded(
                tool_results.strip() or "No prior tool result supplied.",
                min(self.policy.max_tool_result_tokens, int(budget * 0.09)),
            ),
            "Retrieved historical evidence:\n" + self._bounded(
                rag_context.strip() or "No RAG evidence was preloaded; use the typed retrieval tool when relevant.",
                min(self.policy.max_rag_tokens, int(budget * 0.08)),
            ),
            (
                "Evidence rule: raw artifacts remain authoritative. A summary or retrieved case is not proof "
                "that the current command ran or that a repair passed."
            ),
        ]
        prompt = "\n\n".join(sections)
        prompt = self._bounded(prompt, self.policy.input_budget)
        return ContextBundle(
            prompt=prompt,
            summary=summary,
            summarized_message_ids=tuple(compacted_ids),
            recent_message_ids=tuple(
                str(item.get("id", "")) for item in recent_rows if item.get("id")
            ),
            input_budget=self.policy.input_budget,
            estimated_tokens=self.count_tokens(prompt),
            compacted=should_compact,
            summary_usage=summary_usage,
        )
