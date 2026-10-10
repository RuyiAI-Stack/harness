"""MySQL integration tests. Use a disposable schema, never a production database."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout

from codex_agent.memory import MemoryStore
from codex_agent.platform.store import PlatformStore
from codex_agent.process_control import validation_environment
from codex_agent.storage.database import StorageError, WorkspaceDatabase, _engine
from codex_agent.storage.import_sqlite import import_sqlite, snapshot
from codex_agent.storage.schema import memories, references, sessions, workspaces
from codex_agent.tests.test_memory import FakeEmbeddingProvider, record


def insert_legacy(conn, table, row):
    names = ", ".join('"' + name + '"' for name in row)
    conn.execute('INSERT INTO "' + table + '" (' + names + ') VALUES (' +
                 ",".join("?" for _ in row) + ")", tuple(row.values()))


class MySQLStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def legacy_source(self, *, old_schema=False):
        path = self.root / "legacy.sqlite3"
        conn = sqlite3.connect(path)
        self.addCleanup(conn.close)
        conn.executescript((Path(__file__).parent / "fixtures/sqlite_v4.sql").read_text())
        conn.execute("INSERT INTO metadata VALUES ('schema_version', ?)", ("2" if old_schema else "4",))
        now = "2026-09-01T00:00:00Z"
        insert_legacy(conn, "sessions", dict(id="session", title="legacy", status="active",
                                           created_at=now, updated_at=now, pinned=1))
        for identifier in ("z-first", "a-second"):
            insert_legacy(conn, "messages", dict(id=identifier, session_id="session", role="user",
                content="unchanged " + identifier, metadata_json='{"kind":"fixture"}', created_at=now))
        insert_legacy(conn, "runs", dict(id="run", session_id="session", intent="validate", operator="demo",
            status="failed", phase="completed", request_json="{}", result_json='{"exit_code":1}',
            created_at=now, updated_at=now))
        insert_legacy(conn, "events", dict(id=42, run_id="run", sequence=7, event_type="completed",
            payload_json='{"status":"failed"}', created_at=now))
        insert_legacy(conn, "context_checkpoints", dict(session_id="session", summary="failure not repaired",
            through_message_id="a-second", source_ids_json='["run"]', metrics_json="{}", updated_at=now))
        insert_legacy(conn, "references_catalog", dict(id="reference", admitted=1,
                                                      payload='{"source":"docs/rule.md"}'))
        source = MemoryStore(self.root / "fixture-seed")
        source.add(record(outcome="failed", confidence_grade="C", evidence={
            "error_excerpt": ["linalg error"],
            "recommended_actions": ["try supported primitives"],
        }))
        for row in source.db.rows(memories):
            row.pop("workspace_id")
            row["id"] = 17
            insert_legacy(conn, "memories", row)
        for row in source.chunk_rows():
            row.pop("workspace_id")
            row["memory_id"] = 17
            insert_legacy(conn, "memory_chunks", row)
        if old_schema:
            conn.execute("DROP TABLE memory_chunks")
            conn.execute("ALTER TABLE memories DROP COLUMN embedding_json")
        conn.commit()
        return path

    def test_real_mysql_and_workspace_isolation(self):
        left, right = MemoryStore(self.root / "left"), MemoryStore(self.root / "right")
        with left.db.transaction() as conn:
            self.assertTrue(conn.execute(text("SELECT VERSION()")).scalar().startswith("8."))
        self.assertEqual(left.add(record())[0], 1)
        self.assertEqual(right.add(record(operator="other"))[0], 1)
        left.archive(1, "fixture")
        self.assertEqual(left.list(), [])
        self.assertEqual(right.list()[0]["operator"], "other")

    def test_concurrent_duplicate_ingest_has_one_parent_and_complete_chunks(self):
        root = self.root / "concurrent-memory"
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: MemoryStore(root).add(record()), range(24)))
        store = MemoryStore(root)
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertEqual(len(store.list()), 1)
        self.assertGreater(len(store.chunk_rows()), 0)
        self.assertEqual({identifier for identifier, _ in results}, {1})

    def test_identical_session_id_in_other_workspace_is_not_visible(self):
        left, right = PlatformStore(self.root / "one"), PlatformStore(self.root / "two")
        original = left.create_session("one")
        with right.db.transaction() as conn:
            right.db.insert(conn, sessions, {**original, "title": "two"})
        self.assertEqual(left.get_session(original["id"])["title"], "one")
        self.assertEqual(right.get_session(original["id"])["title"], "two")
        left.delete_session(original["id"])
        self.assertEqual(right.get_session(original["id"])["title"], "two")

    def test_event_serialization_across_independent_processes(self):
        root = self.root / "processes"
        store = PlatformStore(root)
        run = store.create_run(store.create_session()["id"], "validate", "demo", {})
        code = ("from codex_agent.platform.store import PlatformStore; from pathlib import Path; "
                "import sys; s=PlatformStore(Path(sys.argv[1])); "
                "[s.add_event(sys.argv[2], 'output', {'n':n}) for n in range(10)]")
        def worker(_):
            subprocess.run([sys.executable, "-I", "-c", code, str(root), run["id"]],
                           check=True, capture_output=True, text=True, timeout=30)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(4)))
        self.assertEqual([r["sequence"] for r in store.list_events(run["id"])], list(range(41)))

    def test_connection_pool_has_a_finite_wait_and_no_unbounded_overflow(self):
        engine = _engine(os.environ["TRITON_MYSQL_URL"], 1, 0, 1, os.getpid())
        try:
            with engine.connect():
                with self.assertRaises(PoolTimeout):
                    engine.connect()
            self.assertEqual(engine.pool.checkedout(), 0)
        finally:
            engine.dispose()

    def test_concurrent_event_sequences_are_unique_and_resumable(self):
        root = self.root / "events"
        store = PlatformStore(root)
        session = store.create_session()
        run = store.create_run(session["id"], "validate", "demo", {})
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda n: PlatformStore(root).add_event(run["id"], "output", {"n": n}), range(40)))
        events = PlatformStore(root).list_events(run["id"])
        self.assertEqual([row["sequence"] for row in events], list(range(41)))
        self.assertEqual(len({row["id"] for row in events}), 41)
        self.assertEqual(PlatformStore(root).list_events(run["id"], after=39), [events[-1]])

    def test_failed_parent_chunk_update_rolls_back(self):
        store = MemoryStore(self.root / "atomic")
        store.add(record())
        before, before_chunks = store.list(), store.chunk_rows()
        with patch.object(store, "_insert_chunks", side_effect=RuntimeError("injected disk failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                store.add(record(evidence={"error_excerpt": ["changed evidence"]}))
        self.assertEqual(store.list(), before)
        self.assertEqual(store.chunk_rows(), before_chunks)

    def test_reingestion_without_provider_keeps_existing_vectors(self):
        root = self.root / "vectors"
        store = MemoryStore(root, FakeEmbeddingProvider())
        store.add(record())
        before = store.chunk_rows()
        MemoryStore(root).add(record())
        self.assertEqual(MemoryStore(root).chunk_rows(), before)
        self.assertTrue(MemoryStore(root).list()[0]["embedding"])
        with patch.object(store.embedding_provider, "embed", wraps=store.embedding_provider.embed) as embed:
            store.add(record())
            embed.assert_not_called()
        store.add(record(confidence_grade="D"))
        self.assertEqual(store.list()[0]["confidence_grade"], "A")

    def test_active_run_blocks_delete_and_finished_delete_cascades(self):
        store = PlatformStore(self.root / "deletion")
        session = store.create_session()
        store.add_message(session["id"], "user", "test")
        run = store.create_run(session["id"], "validate", "demo", {}, status="running")
        store.save_context_checkpoint(session["id"], summary="test", through_message_id=None,
                                      source_ids=[], metrics={})
        with self.assertRaises(ValueError):
            store.delete_session(session["id"])
        store.update_run(run["id"], status="completed")
        store.delete_session(session["id"])
        self.assertEqual(store.list_messages(session["id"]), [])
        self.assertEqual(store.list_events(run["id"]), [])
        self.assertIsNone(store.get_context_checkpoint(session["id"]))

    def test_readonly_import_preserves_ids_evidence_order_and_counters(self):
        source = self.legacy_source()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        target = self.root / "imported"
        receipt = import_sqlite(target, platform=source, memory=source, references=source)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertFalse(receipt["labels_reaudited"])
        self.assertFalse(receipt["evidence_paths_rewritten"])
        platform = PlatformStore(target)
        self.assertEqual([r["id"] for r in platform.list_messages("session")], ["z-first", "a-second"])
        self.assertEqual(platform.get_context_checkpoint("session")["summary"], "failure not repaired")
        event = platform.add_event("run", "output", {})
        self.assertEqual((event["id"], event["sequence"]), (43, 8))
        platform.add_message("session", "user", "third")
        self.assertEqual(platform.list_messages("session")[-1]["content"], "third")
        memories_store = MemoryStore(target)
        item = memories_store.list()[0]
        self.assertEqual(item["id"], 17)
        self.assertEqual(item["outcome"], "failed")
        self.assertEqual(item["confidence_grade"], "C")
        self.assertEqual(item["evidence"]["recommended_actions"], ["try supported primitives"])
        self.assertNotIn("applied_action", item["evidence"])
        self.assertEqual(memories_store.add(record(operator="new"))[0], 18)
        self.assertEqual(len(memories_store.db.rows(references)), 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            import_sqlite(target, memory=source)

    def test_old_schema_rebuilds_chunks_without_fabricating_results(self):
        source = self.legacy_source(old_schema=True)
        target = self.root / "old"
        result = import_sqlite(target, memory=source)
        self.assertEqual(result["old_memory_version"], 2)
        store = MemoryStore(target)
        self.assertGreater(len(store.chunk_rows(17)), 0)
        self.assertEqual(store.list()[0]["embedding"], [])
        self.assertEqual(store.list()[0]["outcome"], "failed")
        self.assertNotIn("Validation result", " ".join(r["text"] for r in store.chunk_rows()))

    def test_import_foreign_key_failure_rolls_back_entire_workspace(self):
        source = self.legacy_source()
        with closing(sqlite3.connect(source)) as conn, conn:
            conn.execute("UPDATE messages SET session_id = 'orphan'")
        target = self.root / "broken"
        with self.assertRaises(StorageError):
            import_sqlite(target, platform=source, memory=source)
        db = WorkspaceDatabase(target)
        self.assertEqual(db.rows(workspaces), [])
        self.assertEqual(db.rows(memories), [])
        with closing(sqlite3.connect(source)) as conn, conn:
            conn.execute("UPDATE messages SET session_id = 'session'")
        self.assertEqual(import_sqlite(target, platform=source)["counts"]["sessions"], 1)

    def test_sqlite_snapshot_is_readonly(self):
        source = self.legacy_source()
        with snapshot(source) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM memories")

    def test_readonly_import_includes_committed_wal_records(self):
        source = self.legacy_source()
        with closing(sqlite3.connect(source)) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("UPDATE sessions SET title='committed WAL'")
            conn.commit()
            self.assertTrue(Path(str(source) + "-wal").exists())
            target = self.root / "wal"
            import_sqlite(target, platform=source)
            self.assertEqual(PlatformStore(target).get_session("session")["title"], "committed WAL")

    def test_missing_configuration_fails_without_creating_sqlite_files(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(StorageError, "TRITON_MYSQL_URL"):
                MemoryStore(self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_database_credentials_are_not_forwarded_to_validation(self):
        with patch.dict(os.environ, {"TRITON_MYSQL_URL": "mysql+pymysql://user:secret@localhost/db"}):
            self.assertNotIn("TRITON_MYSQL_URL", validation_environment())

    def test_invalid_url_errors_do_not_expose_secret(self):
        with self.assertRaises(StorageError) as caught:
            _engine("postgresql://user:secret@localhost/db", 1, 0, 1, os.getpid())
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
