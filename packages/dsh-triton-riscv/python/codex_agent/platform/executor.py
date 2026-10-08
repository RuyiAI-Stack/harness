"""DeepSeek Harness execution and approval-event extraction for FastAPI runs."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from codex_agent.harness import HarnessAgent
from codex_agent.platform.store import PlatformStore
from codex_agent.platform.conversation_context import (
    ContextPolicy, ConversationContextManager, ExtractiveConversationSummarizer,
    OpenAIConversationSummarizer,
)
from codex_agent.validation_evidence import (
    audit_validation_receipt,
    no_validation_evidence,
)


PROPOSAL_ID_RE = re.compile(
    r"\b(development-proposal|repair)-\d{8}-\d{6}-[0-9a-f]{8}\b"
)


class HarnessRunExecutor:
    def __init__(
        self,
        repo_root: Path,
        store: PlatformStore,
        agent: HarnessAgent,
        workers: int = 2,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.store = store
        self.agent = agent
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dsh-agent")
        self._active: set[str] = set()
        self._lock = threading.Lock()
        settings = agent.settings
        summarizer = ExtractiveConversationSummarizer()
        if settings.context_summary_mode == "model" and settings.api_key:
            summarizer = OpenAIConversationSummarizer(base_url=settings.base_url,
                api_key=settings.api_key, model=settings.model)
        self.context_manager = ConversationContextManager(ContextPolicy(
            max_input_tokens=settings.context_window_tokens,
            safety_ratio=settings.context_safety_ratio,
            output_reserve_tokens=settings.context_output_reserve_tokens,
            recent_turns=settings.context_recent_turns,
        ), summarizer=summarizer)

    def confirm(self, run_id: str) -> dict:
        """Backward-compatible explicit confirmation for older API clients."""

        run = self.store.get_run(run_id)
        if run["status"] == "awaiting-confirmation":
            self._validate_request(run["request"])
            self.store.add_event(run_id, "confirmed", {"message": "用户已确认调用 Harness"})
            self._queue(run_id, phase="confirmed")
        return self.store.get_run(run_id)

    def schedule(self, run_id: str) -> dict:
        """Start a normal model turn without asking the user to select a model."""

        run = self.store.get_run(run_id)
        if run["status"] == "awaiting-confirmation":
            self._validate_request(run["request"])
            self.store.add_event(
                run_id,
                "scheduled",
                {"message": "Harness 已自动接收用户任务"},
            )
            self._queue(run_id, phase="scheduled")
        return self.store.get_run(run_id)

    def _queue(self, run_id: str, *, phase: str) -> None:
        self.store.update_run(run_id, status="queued", phase=phase)
        with self._lock:
            if run_id not in self._active:
                self._active.add(run_id)
                self.pool.submit(self._execute, run_id)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.agent.close()

    @staticmethod
    def _validate_request(request: dict) -> None:
        if request.get("action") != "deepseek-harness":
            raise ValueError("run request is not a DeepSeek Harness task")
        task = request.get("task")
        if not isinstance(task, str) or not task.strip() or "\x00" in task:
            raise ValueError("invalid Harness task")
        session_id = request.get("harness_session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("invalid Harness session id")

    @staticmethod
    def _notification_payload(notification: dict[str, Any]) -> dict[str, Any]:
        method = str(notification.get("method", "harness.notification"))
        payload = notification.get("payload", {})
        preview = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(preview) > 4000:
            preview = preview[:4000] + "..."
        return {"message": method, "detail": preview}

    @classmethod
    def _proposal_references(
        cls,
        notification: dict[str, Any],
    ) -> list[dict[str, str]]:
        event = cls._harness_event(notification)
        if event is None or event.get("type") != "tool/result":
            return []
        serialized = json.dumps(notification, ensure_ascii=False, sort_keys=True)
        references: list[dict[str, str]] = []
        for match in PROPOSAL_ID_RE.finditer(serialized):
            proposal_id = match.group(0)
            if any(item["proposal_id"] == proposal_id for item in references):
                continue
            proposal_type = (
                "development"
                if proposal_id.startswith("development-proposal-")
                else "repair"
            )
            references.append(
                {"proposal_id": proposal_id, "proposal_type": proposal_type}
            )
        return references

    @classmethod
    def _validation_plan_references(
        cls,
        notification: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Find structured dry-run validation receipts inside Harness events."""

        references: list[dict[str, str]] = []
        for item in cls._structured_dicts(notification):
            run_id = item.get("run_id")
            command = item.get("command")
            operator = item.get("operator")
            if (
                item.get("status") != "planned"
                or not isinstance(run_id, str)
                or not run_id.startswith("run-")
                or not isinstance(command, str)
                or not command.strip()
                or not isinstance(operator, str)
                or not operator.strip()
            ):
                continue
            if any(reference["validation_run_id"] == run_id for reference in references):
                continue
            references.append(
                {
                    "validation_run_id": run_id,
                    "operator": operator,
                    "command": command,
                }
            )
        return references

    @classmethod
    def _validation_receipt_references(
        cls,
        notification: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Find validation tool results that point to durable receipts."""

        references: list[dict[str, Any]] = []
        for item in cls._structured_dicts(notification):
            run_id = item.get("run_id")
            receipt_path = item.get("receipt_path")
            if (
                not isinstance(run_id, str)
                or not run_id.startswith("run-")
                or not isinstance(receipt_path, str)
                or not receipt_path
                or not isinstance(item.get("operator"), str)
                or item.get("status") not in {"planned", "passed", "failed"}
            ):
                continue
            if any(reference["run_id"] == run_id for reference in references):
                continue
            references.append(
                {
                    "run_id": run_id,
                    "operator": item["operator"],
                    "status": item["status"],
                    "receipt_path": receipt_path,
                }
            )
        return references

    @classmethod
    def _structured_dicts(cls, value: Any):
        """Yield dictionaries, including JSON encoded inside tool-result strings."""

        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from cls._structured_dicts(child)
            return
        if isinstance(value, list):
            for child in value:
                yield from cls._structured_dicts(child)
            return
        if not isinstance(value, str):
            return
        stripped = value.strip()
        if not stripped.startswith(("{", "[")):
            return
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return
        yield from cls._structured_dicts(decoded)

    @staticmethod
    def _harness_event(notification: dict[str, Any]) -> dict[str, Any] | None:
        payload = notification.get("payload")
        if not isinstance(payload, dict):
            return None
        event = payload.get("event")
        return event if isinstance(event, dict) else None

    @staticmethod
    def _message_text(message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        parts: list[str] = []
        for block in message.get("content", []):
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts)

    @staticmethod
    def _message_phase(message: Any) -> str | None:
        if not isinstance(message, dict):
            return None
        source = message.get("source")
        if not isinstance(source, dict):
            return None
        replay = source.get("replayState")
        if not isinstance(replay, dict):
            return None
        for block in replay.get("blocks", []):
            if not isinstance(block, dict):
                continue
            signature = block.get("textSignature")
            if not isinstance(signature, str):
                continue
            try:
                decoded = json.loads(signature)
            except json.JSONDecodeError:
                continue
            phase = decoded.get("phase") if isinstance(decoded, dict) else None
            if isinstance(phase, str):
                return phase
        return None

    @staticmethod
    def _tool_label(name: str) -> str:
        short_name = name.removeprefix("mcp__triton_riscv__")
        labels = {
            "check_validation_environment": "检查 RISC-V 验证环境",
            "discover_operator": "查找算子实现和测试",
            "prepare_operator_development": "准备新算子语义合同",
            "propose_operator_implementation": "生成新算子开发提案",
            "apply_operator_implementation": "应用已批准的新算子提案",
            "validate_operator": "规划或执行算子验证",
            "diagnose_failure": "诊断验证失败",
            "propose_repair": "生成源码修复提案",
            "apply_repair": "应用已批准的源码修复",
        }
        return labels.get(short_name, f"调用工具 {short_name}")

    @classmethod
    def _public_events(
        cls,
        notification: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Translate protocol noise into stream text and public progress summaries."""

        event = cls._harness_event(notification)
        if event is None:
            return []
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
        step = data.get("step")

        if event_type == "turn/start":
            return [
                (
                    "agent-step",
                    {
                        "kind": "analysis",
                        "message": "开始分析用户请求",
                        "turn": data.get("turn"),
                    },
                )
            ]

        if event_type == "assistant/chunk":
            chunk = data.get("chunk")
            if isinstance(chunk, dict) and chunk.get("type") == "text-delta":
                text = chunk.get("text")
                if isinstance(text, str) and text:
                    return [
                        (
                            "assistant-delta",
                            {
                                "text": text,
                                "step": step,
                            },
                        )
                    ]
            return []

        if event_type == "assistant/message":
            message = data.get("message")
            text = cls._message_text(message)
            phase = cls._message_phase(message)
            if phase == "commentary" and text:
                return [
                    (
                        "agent-step",
                        {
                            "kind": "model-progress",
                            "message": text,
                            "step": step,
                        },
                    )
                ]
            if phase == "final_answer":
                return [
                    (
                        "agent-step",
                        {
                            "kind": "final-answer",
                            "message": "最终回答已生成",
                            "step": step,
                        },
                    )
                ]
            return []

        if event_type == "tool/call":
            name = str(data.get("name", "unknown"))
            arguments: dict[str, Any] = {}
            raw_arguments = data.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    decoded = json.loads(raw_arguments)
                    if isinstance(decoded, dict):
                        arguments = decoded
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            context = []
            for key in (
                "operator_name",
                "run_id",
                "approved_run_id",
                "proposal_id",
                "development_id",
            ):
                value = arguments.get(key)
                if isinstance(value, str) and value:
                    context.append(f"{key}={value}")
            return [
                (
                    "agent-step",
                    {
                        "kind": "tool-call",
                        "tool": name.removeprefix("mcp__triton_riscv__"),
                        "message": cls._tool_label(name),
                        "detail": ", ".join(context),
                        "step": step,
                    },
                )
            ]

        if event_type == "tool/result":
            result: dict[str, Any] | None = None
            for candidate in cls._structured_dicts(data):
                if "status" in candidate and any(
                    key in candidate
                    for key in (
                        "operator",
                        "run_id",
                        "proposal_id",
                        "development_id",
                        "host",
                    )
                ):
                    result = candidate
                    break
            if result is None:
                return [
                    (
                        "agent-step",
                        {
                            "kind": "tool-result",
                            "message": "工具调用已返回",
                            "step": step,
                        },
                    )
                ]
            details = []
            for key in (
                "operator",
                "status",
                "failure_stage",
                "run_id",
                "proposal_id",
                "host",
                "architecture",
            ):
                value = result.get(key)
                if value is not None and value != "":
                    details.append(f"{key}={value}")
            status = result.get("status")
            return [
                (
                    "agent-step",
                    {
                        "kind": "tool-result",
                        "status": status,
                        "message": "工具返回：" + "，".join(details),
                        "step": step,
                    },
                )
            ]

        return []

    @staticmethod
    def _notification_error(notification: dict[str, Any]) -> str | None:
        """Extract a useful Harness error without persisting an empty reply."""

        def visit(value: Any) -> str | None:
            if isinstance(value, dict):
                error = value.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message.strip():
                        return message.strip()
                elif isinstance(error, str) and error.strip():
                    return error.strip()
                for child in value.values():
                    found = visit(child)
                    if found:
                        return found
            elif isinstance(value, list):
                for child in value:
                    found = visit(child)
                    if found:
                        return found
            return None

        return visit(notification)

    def _execute(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        request = run["request"]
        self.store.update_run(run_id, status="running", phase="harness-running")
        self.store.add_event(
            run_id,
            "started",
            {
                "message": "DeepSeek Harness agent loop started",
                "model": self.agent.settings.model,
            },
        )
        event_count = 0
        recorded_proposals: set[str] = set()
        recorded_validation_plans: set[str] = set()
        recorded_validation_receipts: set[str] = set()
        latest_validation_evidence = no_validation_evidence()
        harness_error: str | None = None

        def record_event(notification: dict[str, Any]) -> None:
            nonlocal event_count, harness_error, latest_validation_evidence
            event_count += 1
            harness_error = self._notification_error(notification) or harness_error
            self.store.add_event(
                run_id,
                "harness-event",
                self._notification_payload(notification),
            )
            for event_type, payload in self._public_events(notification):
                self.store.add_event(run_id, event_type, payload)
            for proposal in self._proposal_references(notification):
                proposal_id = proposal["proposal_id"]
                if proposal_id not in recorded_proposals:
                    recorded_proposals.add(proposal_id)
                    self.store.add_event(
                        run_id,
                        "approval-required",
                        {
                            **proposal,
                            "message": "A model proposal is waiting for human review",
                        },
                    )
            for plan in self._validation_plan_references(notification):
                validation_run_id = plan["validation_run_id"]
                if validation_run_id not in recorded_validation_plans:
                    recorded_validation_plans.add(validation_run_id)
                    self.store.add_event(
                        run_id,
                        "approval-required",
                        {
                            **plan,
                            "approval_type": "command",
                            "message": "A validation command is waiting for human review",
                        },
                    )
            for reference in self._validation_receipt_references(notification):
                receipt_run_id = reference["run_id"]
                if receipt_run_id in recorded_validation_receipts:
                    continue
                recorded_validation_receipts.add(receipt_run_id)
                latest_validation_evidence = audit_validation_receipt(
                    self.repo_root,
                    reference,
                )
                self.store.add_event(
                    run_id,
                    "validation-evidence",
                    latest_validation_evidence.model_dump(),
                )

        try:
            result = self.agent.run(
                self._managed_task(run) if self.agent.settings.managed_context else request["task"],
                session_id=request["harness_session_id"],
                on_event=record_event,
            )
            final_response = result.final_response.strip()
            finish_reason = (result.finish_reason or "unknown").lower()
            if finish_reason in {"error", "cancelled", "canceled"}:
                detail = harness_error or f"Harness finish_reason={finish_reason}"
                raise RuntimeError(detail)
            if not final_response:
                detail = harness_error or "Harness completed without a final response"
                raise RuntimeError(detail)

            for line in final_response.splitlines()[-80:]:
                self.store.add_event(run_id, "output", {"line": line})
            final = {
                "harness_session_id": result.session_id,
                "finish_reason": result.finish_reason,
                "event_count": event_count,
                "validation_evidence": latest_validation_evidence.model_dump(),
            }
            self.store.add_event(run_id, "completed", final)
            self.store.add_message(
                run["session_id"],
                "assistant",
                final_response,
                {
                    "view": "harness-result",
                    "run_id": run_id,
                    "harness_session_id": result.session_id,
                },
            )
            # Publish terminal status last. Consumers treat it as a barrier after
            # which all events and the final assistant message are durable.
            self.store.update_run(run_id, status="completed", phase="completed", result=final)
        except Exception as error:
            final = {"error": str(error), "event_count": event_count}
            self.store.add_event(run_id, "failed", {"message": str(error)})
            self.store.add_message(
                run["session_id"],
                "assistant",
                f"DeepSeek Harness 执行失败：`{error}`",
                {"view": "harness-error", "run_id": run_id},
            )
            self.store.update_run(run_id, status="failed", phase="failed", result=final)
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _managed_task(self, run: dict) -> str:
        session_id = run["session_id"]
        request = run["request"]
        checkpoint = self.store.get_context_checkpoint(session_id) or {}
        summarized = set(checkpoint.get("source_ids") or [])
        current_message_id = request.get("user_message_id")
        messages = [
            item for item in self.store.list_messages(session_id)
            if item.get("id") not in summarized and item.get("id") != current_message_id
        ]
        prior_runs = [
            item for item in self.store.list_runs(session_id)
            if item.get("id") != run["id"]
        ][:6]
        tool_rows = []
        for prior in prior_runs:
            events = self.store.list_events(prior["id"])
            terminal = [
                event for event in events
                if event.get("event_type") in {"completed", "failed", "approved", "output"}
                or (
                    event.get("event_type") == "harness-event"
                    and any(
                        marker in json.dumps(event.get("payload") or {}).lower()
                        for marker in ("tool", "mcp__", "command", "result")
                    )
                )
            ][-12:]
            tool_rows.append({
                "run_id": prior["id"],
                "intent": prior.get("intent"),
                "status": prior.get("status"),
                "phase": prior.get("phase"),
                "result": prior.get("result"),
                "terminal_events": terminal,
            })
        bundle = self.context_manager.build(
            current_request=request["task"],
            messages=messages,
            previous_summary=str(checkpoint.get("summary") or ""),
            task_state=json.dumps(
                {
                    "platform_session_id": session_id,
                    "current_run_id": run["id"],
                    "current_intent": run.get("intent"),
                    "current_operator": run.get("operator"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            tool_results=json.dumps(tool_rows, ensure_ascii=False, sort_keys=True),
        )
        if bundle.compacted:
            source_ids = list(dict.fromkeys([
                *(checkpoint.get("source_ids") or []),
                *bundle.summarized_message_ids,
            ]))
            self.store.save_context_checkpoint(
                session_id,
                summary=bundle.summary,
                through_message_id=(
                    bundle.summarized_message_ids[-1]
                    if bundle.summarized_message_ids else checkpoint.get("through_message_id")
                ),
                source_ids=source_ids,
                metrics={
                    "estimated_tokens": bundle.estimated_tokens,
                    "input_budget": bundle.input_budget,
                    "summary_usage": bundle.summary_usage,
                },
            )
        self.store.add_event(run["id"], "context-prepared", {
            "compacted": bundle.compacted,
            "estimated_tokens": bundle.estimated_tokens,
            "input_budget": bundle.input_budget,
            "summarized_messages": len(bundle.summarized_message_ids),
            "recent_messages": len(bundle.recent_message_ids),
            "summary_usage": bundle.summary_usage,
        })
        return bundle.prompt
