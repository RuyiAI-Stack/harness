"""Submission validates capabilities and binds inputs before creating durable work."""
import json
import uuid

from codex_agent.execution_guard import approval_digest, artifact
from codex_agent.runtime_config import permission_enabled, runtime_config, remote_execution_identity
from codex_agent.platform.store import timestamp
from codex_agent.storage.schema import messages, runs, sessions
from .store import JobStore, Conflict, encode, digest


def request_key(kind, key):
    if not isinstance(key, str) or not 1 <= len(key) <= 160:
        raise ValueError("request_id must contain 1..160 characters")
    # User keys cannot occupy internal follow-up identities.
    return "user:" + kind + ":" + digest(key)


class TaskService:
    def __init__(self, root):
        self.root = root.resolve()
        self.jobs = JobStore(self.root)

    def submit_validation(self, operator, approved_run_id, key):
        key = request_key("validation", key)
        if not runtime_config().queue.enabled:
            raise ValueError("queue execution is disabled")
        if not permission_enabled("validation"):
            raise PermissionError("validation capability is disabled")
        plan = artifact(self.root, "validation", approved_run_id)
        if plan.get("operator") != operator or plan.get("approval", {}).get("status") != "approved":
            raise PermissionError("a matching host-approved validation plan is required")
        seal = approval_digest(self.root, "validation", plan)
        if plan.get("approval_digest") != seal:
            raise PermissionError("approved plan changed; request approval again")
        return self.jobs.submit("validation", {"operator": operator, "approved_run_id": approved_run_id,
            "approval_digest": seal, "source_env": plan.get("source_env", True),
            "timeout_seconds": plan.get("timeout_seconds", 900), "remote_identity": remote_execution_identity()}, key)

    def submit_turn(self, platform, session_id, text, key):
        key = request_key("agent", key)
        if not runtime_config().queue.enabled:
            raise ValueError("queue execution is disabled")
        if not text.strip() or "\x00" in text:
            raise ValueError("invalid task text")
        db, store = self.jobs.db, platform.store
        with db.transaction() as conn:
            db.lock(conn, "task-submit")
            old = self.jobs.find_key(conn, key)
            if old:
                payload = json.loads(old["payload_json"])
                if old["kind"] != "agent" or payload.get("session_id") != session_id or payload.get("text") != text:
                    raise Conflict("idempotency key reused for a different message")
                run = store._get(runs, old["run_id"], connection=conn)
                message_id = json.loads(run["request_json"])["user_message_id"]
                message = store._get(messages, message_id, connection=conn)
            else:
                session = store._get(sessions, session_id, connection=conn, lock=True)
                now, run_id, message_id = timestamp(), uuid.uuid4().hex, uuid.uuid4().hex
                message = dict(id=message_id, session_id=session_id, ordinal=db.allocate(conn, "messages"),
                    role="user", content=text, metadata_json="{}", created_at=now)
                db.insert(conn, messages, message)
                db.update(conn, sessions, {"updated_at": now, "title": text[:80] if session["title"] == "新对话" else session["title"]}, sessions.c.id == session_id)
                request = {"action": "deepseek-harness", "task": text,
                    "harness_session_id": "triton-job-" + run_id, "user_message_id": message_id,
                    "requires_confirmation": False}
                run = dict(id=run_id, session_id=session_id, intent="harness-agent", operator=None,
                    status="queued", phase="outbox-pending", request_json=encode(request), result_json="{}",
                    created_at=now, updated_at=now, next_sequence=0)
                db.insert(conn, runs, run)
                submitted = self.jobs.submit("agent", {"run_id": run_id, "session_id": session_id, "text": text},
                    key, run_id=run_id, connection=conn)
                old = {"id": submitted["id"]}
                store._add_event(conn, run_id, "queued", {"message": "Task persisted for RabbitMQ delivery"})
        return {"user_message": store._row(message), "assistant_message": None,
                "task": {"intent": "harness-agent", "operator": None, "confidence": 1.0,
                         "runtime": "deepseek-harness"}, "run": store.get_run(run["id"]),
                "job": self.jobs.get(old["id"])}
