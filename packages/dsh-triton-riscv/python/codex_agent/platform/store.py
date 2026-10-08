"""SQLite persistence for chat sessions, messages, runs, and event streams."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class PlatformStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    intent TEXT NOT NULL,
                    operator TEXT,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, sequence);
                CREATE TABLE IF NOT EXISTS context_checkpoints (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    through_message_id TEXT,
                    source_ids_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "pinned" not in session_columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for key in ("metadata_json", "request_json", "result_json", "payload_json", "source_ids_json", "metrics_json"):
            if key in item:
                item[key.removesuffix("_json")] = json.loads(item.pop(key) or "{}")
        return item

    def create_session(self, title: str = "新对话") -> dict:
        session_id = uuid.uuid4().hex
        now = timestamp()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, title, status, created_at, updated_at, pinned)
                VALUES (?, ?, 'active', ?, ?, 0)
                """,
                (session_id, title[:80] or "新对话", now, now),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        item = self._row(row)
        if item is None:
            raise KeyError(f"unknown session: {session_id}")
        return item

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY pinned DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def set_session_pinned(self, session_id: str, pinned: bool) -> dict:
        self.get_session(session_id)
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE sessions SET pinned = ? WHERE id = ?",
                (int(pinned), session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self.get_session(session_id)
        with self._lock, self._connection() as connection:
            active_run = connection.execute(
                """
                SELECT 1 FROM runs
                WHERE session_id = ? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if active_run is not None:
                raise ValueError("cannot delete a session while a run is active")
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def list_session_events(
        self,
        session_id: str,
        *,
        event_type: str | None = None,
    ) -> list[dict]:
        self.get_session(session_id)
        query = """
            SELECT events.*
            FROM events
            JOIN runs ON runs.id = events.run_id
            WHERE runs.session_id = ?
        """
        parameters: list[object] = [session_id]
        if event_type is not None:
            query += " AND events.event_type = ?"
            parameters.append(event_type)
        query += " ORDER BY events.created_at, events.id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row(row) for row in rows]

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title[:80], timestamp(), session_id),
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        message_id = uuid.uuid4().hex
        now = timestamp()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        return self.get_message(message_id)

    def get_message(self, message_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        item = self._row(row)
        if item is None:
            raise KeyError(f"unknown message: {message_id}")
        return item

    def list_messages(self, session_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def create_run(
        self,
        session_id: str,
        intent: str,
        operator: str | None,
        request: dict,
        *,
        status: str = "awaiting-confirmation",
        phase: str = "planned",
    ) -> dict:
        run_id = uuid.uuid4().hex
        now = timestamp()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)",
                (
                    run_id,
                    session_id,
                    intent,
                    operator,
                    status,
                    phase,
                    json.dumps(request, sort_keys=True),
                    now,
                    now,
                ),
            )
        self.add_event(run_id, "planned", {"intent": intent, "operator": operator})
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        item = self._row(row)
        if item is None:
            raise KeyError(f"unknown run: {run_id}")
        return item

    def list_runs(self, session_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        phase: str | None = None,
        result: dict | None = None,
    ) -> dict:
        current = self.get_run(run_id)
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, phase = ?, result_json = ?, updated_at = ? WHERE id = ?",
                (
                    status or current["status"],
                    phase or current["phase"],
                    json.dumps(result if result is not None else current["result"], sort_keys=True),
                    timestamp(),
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def add_event(self, run_id: str, event_type: str, payload: dict) -> dict:
        with self._lock, self._connection() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            cursor = connection.execute(
                "INSERT INTO events(run_id, sequence, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, sequence, event_type, json.dumps(payload, sort_keys=True), timestamp()),
            )
            row = connection.execute(
                "SELECT * FROM events WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._row(row)

    def list_events(self, run_id: str, after: int = -1) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
                (run_id, after),
            ).fetchall()
        return [self._row(row) for row in rows]

    def session_bundle(self, session_id: str) -> dict:
        return {
            "session": self.get_session(session_id),
            "messages": self.list_messages(session_id),
            "runs": self.list_runs(session_id),
        }

    def get_context_checkpoint(self, session_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM context_checkpoints WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._row(row)


    def save_context_checkpoint(
        self,
        session_id: str,
        *,
        summary: str,
        through_message_id: str | None,
        source_ids: list[str],
        metrics: dict,
    ) -> dict:
        self.get_session(session_id)
        now = timestamp()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO context_checkpoints(
                    session_id, summary, through_message_id, source_ids_json,
                    metrics_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    through_message_id = excluded.through_message_id,
                    source_ids_json = excluded.source_ids_json,
                    metrics_json = excluded.metrics_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    summary,
                    through_message_id,
                    json.dumps(source_ids, sort_keys=True),
                    json.dumps(metrics, sort_keys=True),
                    now,
                ),
            )
        checkpoint = self.get_context_checkpoint(session_id)
        assert checkpoint is not None
        return checkpoint
