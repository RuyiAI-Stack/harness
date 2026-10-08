"""FastAPI-facing application service backed by DeepSeek Harness."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from codex_agent.harness import HarnessAgent, HarnessSettings
from codex_agent.platform.store import PlatformStore
from codex_agent.paths import state_root


class PlatformService:
    """Persist a user turn and prepare one Harness run."""

    def __init__(
        self,
        repo_root: Path,
        store: PlatformStore,
        agent: HarnessAgent | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.store = store
        self.results_dir = state_root(self.repo_root)
        self.settings = agent.settings if agent else HarnessSettings.from_env(self.repo_root)
        self.agent = agent or HarnessAgent(self.settings)
        # A Harness session id may have a persisted log on disk. Reusing that
        # id after the API process restarts causes the runtime to reject the
        # fresh in-memory session as an id collision. Keep ids stable during
        # one API lifetime for conversational context, but rotate them on boot.
        self.harness_instance_id = uuid.uuid4().hex[:12]

    def operator_inventory(self) -> dict:
        return self._read_json(
            self.results_dir / "operators.json",
            {"summary": {}, "operators": []},
        )

    def project_inventory(self) -> dict:
        return self._read_json(
            self.results_dir / "project-targets.json",
            {"summary": {}, "targets": []},
        )

    def create_session(self, title: str = "新对话") -> dict:
        return self.store.create_session(title)

    def bootstrap(self) -> dict:
        return {
            "product": "Triton-RISCV Agent",
            "operators": self.operator_inventory().get("summary", {}),
            "project": self.project_inventory().get("summary", {}),
            "capabilities": [
                "repository-analysis",
                "operator-discovery",
                "operator-validation",
                "failure-diagnosis",
                "bounded-operator-development",
            ],
            "harness": {
                "runtime": "DeepSeek Harness",
                "plugin": self.settings.plugin.public_metadata(),
                "provider": self.settings.provider,
                "model": self.settings.model,
                "api_configured": bool(self.settings.api_key),
                "remote_configured": bool(
                    self.settings.remote_host and self.settings.remote_root
                ),
            },
            "suggestions": [
                "介绍一下当前 Triton-RISCV 项目的测试覆盖情况",
                "分析 gelu_and_mul 算子的实现和风险",
                "验证 relu_and_mul 算子",
                "查找以前处理 linalg lowering 失败的经验",
            ],
        }

    def handle_message(self, session_id: str, text: str) -> dict:
        self.store.get_session(session_id)
        request_text = text.strip()
        if not request_text:
            raise ValueError("message cannot be empty")

        user_message = self.store.add_message(session_id, "user", request_text)
        if len(self.store.list_messages(session_id)) == 1:
            self.store.update_session_title(session_id, self._title(request_text))

        # Validate the bounded domain context before queueing a model call. The
        # prompt is rebuilt at execution time so credentials never enter SQLite.
        self.agent.prepare(request_text)
        task = {
            "intent": "harness-agent",
            "operator": None,
            "confidence": 1.0,
            "runtime": "deepseek-harness",
        }
        request = {
            "action": "deepseek-harness",
            "task": request_text,
            "harness_session_id": (
                f"triton-ui-{self.harness_instance_id}-{session_id}"
                + (f"-{user_message['id']}" if self.settings.managed_context else "")
            ),
            "user_message_id": user_message["id"],
            "requires_confirmation": False,
        }
        run = self.store.create_run(
            session_id,
            "harness-agent",
            None,
            request,
        )
        return {
            "user_message": user_message,
            "assistant_message": None,
            "task": task,
            "run": run,
        }

    @staticmethod
    def _read_json(path: Path, fallback: dict) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _title(text: str) -> str:
        compact = " ".join(text.split())
        return compact[:32] + ("..." if len(compact) > 32 else "")
