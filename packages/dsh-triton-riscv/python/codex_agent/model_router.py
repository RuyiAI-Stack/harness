"""Model selection and low-overhead execution telemetry.

The router is deliberately provider agnostic: callers receive a model name and
can pass it to their backend, while telemetry can be used with any model API.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    complexity: str
    failure_stage: int
    reason: str
    complexity_score: int = 0


class ModelRouter:
    """Choose a model from task intent and failure stage.

    ``models`` can override the names used by the policy, e.g.
    ``{"fast": "deepseek-v4-flash", "coder": "qwen-coder"}``.
    """

    DEFAULT_MODELS = {"rules": "rules", "fast": "fast", "coder": "coder", "deep": "deep"}
    _DEEP_RE = re.compile(r"\b(mlir|compiler|llvm|kernel|triton|lowering|optimization|codegen|performance)\b", re.I)
    _CODE_RE = re.compile(r"\b(code|implement|patch|debug|fix|refactor|operator|python|api|test)\b", re.I)
    _SUMMARY_RE = re.compile(r"\b(summar(?:y|ize)|explain|brief|classify|extract|rewrite)\b", re.I)
    _RULE_RE = re.compile(r"\b(rule|rules|validate|validation|lint|check|schema|deterministic)\b", re.I)

    def __init__(self, models: Optional[Mapping[str, str]] = None) -> None:
        self.models = {**self.DEFAULT_MODELS, **dict(models or {})}

    def classify(self, task: str) -> str:
        """Return ``rules``, ``simple``, ``medium`` or ``complex``."""
        text = task or ""
        if self._RULE_RE.search(text):
            return "rules"
        if self._DEEP_RE.search(text):
            return "complex"
        if self._CODE_RE.search(text):
            return "medium"
        if self._SUMMARY_RE.search(text) or len(text) < 280:
            return "simple"
        # Long prompts usually carry more context and benefit from the coder tier.
        return "medium" if len(text) < 2500 else "complex"

    def complexity_score(self, task: str) -> int:
        """Return a transparent 0-100 score used to explain routing choices."""
        text = task or ""
        score = min(50, len(text) // 80)
        score += 35 if self._DEEP_RE.search(text) else 0
        score += 15 if self._CODE_RE.search(text) else 0
        return min(100, score)

    def route(
        self,
        task: str,
        *,
        failure_stage: int | str = 0,
        complexity: Optional[str] = None,
    ) -> RoutingDecision:
        try:
            stage = max(0, int(failure_stage))
        except (TypeError, ValueError):
            # Failure classifiers commonly provide names (``buddy-opt``,
            # ``correctness``); any named failure is an escalation signal.
            stage = 1 if str(failure_stage).strip() else 0
        level = (complexity or self.classify(task)).lower()
        if level not in {"rules", "simple", "medium", "complex"}:
            level = self.classify(task)
        if level == "rules":
            key, reason = "rules", "deterministic task"
        elif level == "simple":
            key, reason = "fast", "short or summary-oriented task"
        elif level == "medium":
            key, reason = "coder", "implementation or debugging task"
        else:
            key, reason = "deep", "compiler, MLIR, kernel, or long-context task"
        # Each failed attempt escalates one tier; a rules task is kept deterministic
        # unless explicitly failed, then uses the fast model for recovery.
        if stage and key != "deep":
            key = {"rules": "fast", "fast": "coder", "coder": "deep"}[key]
            reason += f"; escalated after failure stage {stage}"
        return RoutingDecision(self.models[key], level, stage, reason, self.complexity_score(task))

    # Friendly aliases for integrations that only need the selected name.
    def select_model(self, task: str, *, failure_stage: int | str = 0, complexity: Optional[str] = None) -> str:
        return self.route(task, failure_stage=failure_stage, complexity=complexity).model

    route_model = select_model


@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: str
    request_id: str
    model: str
    task: str = ""
    complexity: str = ""
    failure_stage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class TelemetryRecorder:
    """Append telemetry events as one JSON object per line.

    The parent directory is created lazily. Writes are guarded by a process-local
    lock so concurrent HarnessAgent calls cannot interleave records.
    """

    def __init__(
        self,
        path: str | Path = "agent-results/model-telemetry.jsonl",
        *,
        pricing: Optional[Mapping[str, tuple[float, float]]] = None,
    ) -> None:
        self.path = Path(path)
        self.pricing = dict(pricing or {})  # model -> (USD per 1K input, output)
        self._lock = threading.Lock()

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = self.pricing.get(model, (0.0, 0.0))
        return (max(0, input_tokens) * in_rate + max(0, output_tokens) * out_rate) / 1000.0

    def record(
        self, event: TelemetryEvent | Mapping[str, Any] | None = None, **kwargs: Any
    ) -> TelemetryEvent:
        if not isinstance(event, TelemetryEvent):
            payload = dict(event or {})
            payload.update(kwargs)
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            payload.setdefault("request_id", uuid.uuid4().hex)
            payload.setdefault("model", "unknown")
            event = TelemetryEvent(**payload)
        payload = asdict(event)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    @contextmanager
    def span(
        self,
        model: str,
        *,
        task: str = "",
        complexity: str = "",
        failure_stage: int = 0,
        request_id: Optional[str] = None,
        input_tokens: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator["TelemetrySpan"]:
        span = TelemetrySpan(
            self,
            model,
            task=task,
            complexity=complexity,
            failure_stage=failure_stage,
            request_id=request_id,
            input_tokens=input_tokens,
            metadata=metadata,
        )
        try:
            yield span
        except Exception as exc:
            span.finish(success=False, error=str(exc))
            raise
        else:
            span.finish()

    def read(self) -> list[TelemetryEvent]:
        if not self.path.exists():
            return []
        events: list[TelemetryEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TelemetryEvent(**json.loads(line)))
        return events

    def summarize(self) -> dict[str, Any]:
        """Return aggregate counters useful for dashboards and smoke checks."""
        events = self.read()
        return {
            "calls": len(events),
            "successful_calls": sum(1 for event in events if event.success),
            "failed_calls": sum(1 for event in events if not event.success),
            "total_tokens": sum(event.total_tokens for event in events),
            "total_cost_usd": sum(event.cost_usd for event in events),
            "latency_ms": sum(event.latency_ms for event in events),
        }


class TelemetrySpan:
    def __init__(
        self,
        recorder: TelemetryRecorder,
        model: str,
        *,
        task: str,
        complexity: str,
        failure_stage: int,
        request_id: Optional[str],
        input_tokens: int,
        metadata: Optional[dict[str, Any]],
    ) -> None:
        self.recorder, self.model, self.task = recorder, model, task
        self.complexity, self.failure_stage = complexity, failure_stage
        self.request_id, self.input_tokens, self.metadata = request_id or uuid.uuid4().hex, input_tokens, metadata
        self.started = time.perf_counter()
        self.output_tokens = 0
        self._done = False

    def finish(
        self,
        *,
        output_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TelemetryEvent:
        if self._done:
            return self.event
        self._done = True
        self.output_tokens = output_tokens
        merged = {**(self.metadata or {}), **(metadata or {})} or None
        self.event = TelemetryEvent(
            timestamp=datetime.now(timezone.utc).isoformat(), request_id=self.request_id,
            model=self.model, task=self.task, complexity=self.complexity, failure_stage=self.failure_stage,
            input_tokens=self.input_tokens, output_tokens=output_tokens, total_tokens=self.input_tokens + output_tokens,
            cost_usd=self.recorder.cost(self.model, self.input_tokens, output_tokens),
            latency_ms=(time.perf_counter() - self.started) * 1000.0, success=success, error=error, metadata=merged,
        )
        return self.recorder.record(self.event)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for providers that do not return usage."""
    return max(0, (len(text or "") + 3) // 4)
