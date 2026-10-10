"""MySQL persistence for workbench conversations, runs and event streams."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert
from codex_agent.storage.database import WorkspaceDatabase
from codex_agent.storage.schema import sessions, messages, runs, events, checkpoints


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class PlatformStore:
    def __init__(self, workspace: Path) -> None:
        self.db = WorkspaceDatabase(workspace)
        self.db.register()

    @staticmethod
    def _row(row):
        if row is None:
            return None
        item = dict(row)
        for key in ("workspace_id", "ordinal", "next_sequence"):
            item.pop(key, None)
        for key in ("metadata_json", "request_json", "result_json", "payload_json",
                    "source_ids_json", "metrics_json"):
            if key in item:
                item[key.removesuffix("_json")] = json.loads(item.pop(key) or "{}")
        return item

    def _get(self, table, identifier, *, connection=None, lock=False):
        rows = self.db.rows(table, table.c.id == identifier, connection=connection, lock=lock)
        if not rows:
            raise KeyError(f"unknown {table.name}: {identifier}")
        return rows[0]

    def create_session(self, title="新对话"):
        identifier, now = uuid.uuid4().hex, timestamp()
        with self.db.transaction() as conn:
            self.db.insert(conn, sessions, dict(id=identifier, title=title[:80] or "新对话",
                status="active", created_at=now, updated_at=now, pinned=0))
        return self.get_session(identifier)

    def get_session(self, session_id):
        return self._row(self._get(sessions, session_id))

    def list_sessions(self, limit=50):
        return [self._row(row) for row in self.db.rows(sessions,
            order=(sessions.c.pinned.desc(), sessions.c.updated_at.desc(), sessions.c.id), limit=limit)]

    def set_session_pinned(self, session_id, pinned):
        with self.db.transaction() as conn:
            self._get(sessions, session_id, connection=conn, lock=True)
            self.db.update(conn, sessions, {"pinned": int(pinned)}, sessions.c.id == session_id)
        return self.get_session(session_id)

    def delete_session(self, session_id):
        with self.db.transaction() as conn:
            self._get(sessions, session_id, connection=conn, lock=True)
            active = self.db.rows(runs, runs.c.session_id == session_id,
                runs.c.status.in_(("queued", "running")), connection=conn, limit=1)
            if active:
                raise ValueError("cannot delete a session while a run is active")
            from codex_agent.tasks.schema import jobs
            run_ids = select(runs.c.id).where(*self.db.predicate(runs, runs.c.session_id == session_id))
            if self.db.rows(jobs, jobs.c.run_id.in_(run_ids), jobs.c.status == "needs-reconciliation", connection=conn, limit=1):
                raise ValueError("resolve the uncertain task before deleting its session")
            self.db.delete(conn, sessions, sessions.c.id == session_id)

    def update_session_title(self, session_id, title):
        with self.db.transaction() as conn:
            self._get(sessions, session_id, connection=conn, lock=True)
            self.db.update(conn, sessions, {"title": title[:80], "updated_at": timestamp()},
                           sessions.c.id == session_id)

    def add_message(self, session_id, role, content, metadata=None):
        identifier, now = uuid.uuid4().hex, timestamp()
        with self.db.transaction() as conn:
            self._get(sessions, session_id, connection=conn, lock=True)
            self.db.insert(conn, messages, dict(id=identifier, session_id=session_id,
                ordinal=self.db.allocate(conn, "messages"), role=role, content=content,
                metadata_json=json.dumps(metadata or {}, sort_keys=True), created_at=now))
            self.db.update(conn, sessions, {"updated_at": now}, sessions.c.id == session_id)
        return self.get_message(identifier)

    def get_message(self, message_id):
        return self._row(self._get(messages, message_id))

    def list_messages(self, session_id):
        return [self._row(row) for row in self.db.rows(messages, messages.c.session_id == session_id,
                                                       order=(messages.c.ordinal,))]

    def create_run(self, session_id, intent, operator, request, *,
                   status="awaiting-confirmation", phase="planned"):
        identifier, now = uuid.uuid4().hex, timestamp()
        with self.db.transaction() as conn:
            self._get(sessions, session_id, connection=conn, lock=True)
            self.db.insert(conn, runs, dict(id=identifier, session_id=session_id, intent=intent,
                operator=operator, status=status, phase=phase,
                request_json=json.dumps(request, sort_keys=True), result_json="{}",
                created_at=now, updated_at=now, next_sequence=0))
            self._add_event(conn, identifier, "planned", {"intent": intent, "operator": operator})
        return self.get_run(identifier)

    def get_run(self, run_id):
        return self._row(self._get(runs, run_id))

    def list_runs(self, session_id):
        return [self._row(row) for row in self.db.rows(runs, runs.c.session_id == session_id,
                                                      order=(runs.c.created_at.desc(), runs.c.id))]

    def update_run(self, run_id, *, status=None, phase=None, result=None):
        with self.db.transaction() as conn:
            current = self._get(runs, run_id, connection=conn)
            # Lock the session first, matching delete_session/create_run lock order.
            self._get(sessions, current["session_id"], connection=conn, lock=True)
            current = self._get(runs, run_id, connection=conn, lock=True)
            values = {"status": status or current["status"], "phase": phase or current["phase"],
                      "updated_at": timestamp()}
            if result is not None:
                values["result_json"] = json.dumps(result, sort_keys=True)
            self.db.update(conn, runs, values, runs.c.id == run_id)
        return self.get_run(run_id)

    def _add_event(self, conn, run_id, event_type, payload):
        run = self._get(runs, run_id, connection=conn, lock=True)
        sequence = run["next_sequence"]
        self.db.update(conn, runs, {"next_sequence": sequence + 1}, runs.c.id == run_id)
        value = dict(id=self.db.allocate(conn, "events"), run_id=run_id, sequence=sequence,
                     event_type=event_type, payload_json=json.dumps(payload, sort_keys=True),
                     created_at=timestamp())
        self.db.insert(conn, events, value)
        return self._row(value)

    def add_event(self, run_id, event_type, payload):
        with self.db.transaction() as conn:
            return self._add_event(conn, run_id, event_type, payload)

    def list_events(self, run_id, after=-1):
        return [self._row(row) for row in self.db.rows(events, events.c.run_id == run_id,
                events.c.sequence > after, order=(events.c.sequence,))]

    def list_session_events(self, session_id, *, event_type=None):
        self.get_session(session_id)
        selected_runs = select(runs.c.id).where(*self.db.predicate(runs, runs.c.session_id == session_id))
        conditions = [events.c.run_id.in_(selected_runs)]
        if event_type is not None:
            conditions.append(events.c.event_type == event_type)
        return [self._row(row) for row in self.db.rows(events, *conditions,
                order=(events.c.created_at, events.c.id))]

    def session_bundle(self, session_id):
        return {"session": self.get_session(session_id), "messages": self.list_messages(session_id),
                "runs": self.list_runs(session_id)}

    def get_context_checkpoint(self, session_id):
        rows = self.db.rows(checkpoints, checkpoints.c.session_id == session_id)
        return self._row(rows[0]) if rows else None

    def save_context_checkpoint(self, session_id, *, summary, through_message_id, source_ids, metrics):
        with self.db.transaction() as conn:
            self._get(sessions, session_id, connection=conn, lock=True)
            values = dict(workspace_id=self.db.workspace_id, session_id=session_id, summary=summary,
                through_message_id=through_message_id, source_ids_json=json.dumps(source_ids, sort_keys=True),
                metrics_json=json.dumps(metrics, sort_keys=True), updated_at=timestamp())
            stmt = insert(checkpoints).values(**values)
            conn.execute(stmt.on_duplicate_key_update(**{
                key: stmt.inserted[key] for key in values if key not in {"workspace_id", "session_id"}}))
        return self.get_context_checkpoint(session_id)
