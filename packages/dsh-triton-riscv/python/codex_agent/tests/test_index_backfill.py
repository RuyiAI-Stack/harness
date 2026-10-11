"""Bounded, resumable enqueue checks; vector quality is not evaluated here."""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

from codex_agent.indexing.__main__ import main
from codex_agent.tests.test_memory_indexing import Fixture
from codex_agent.storage.schema import memories, memory_index_chunks, memory_indexes, memory_index_entries
from codex_agent.tasks.schema import jobs, outbox


class BackfillTests(Fixture, unittest.TestCase):
    def sources(self, count=5):
        self.cfg["indexing"]["enabled"] = False
        self.configure()
        ids = [self.add(source_run=f"batch-fixture-{i}") for i in range(count)]
        self.cfg["indexing"]["enabled"] = True
        self.configure()
        return ids

    def drain(self, cursor=None, **kwargs):
        results = []
        for _ in range(20):
            result = self.index.backfill_batch(cursor=cursor, page_size=2, **kwargs)
            results.append(result)
            if result["done"]:
                return results
            cursor = result["next_cursor"]
        self.fail("backfill did not terminate")

    def test_pause_resume_and_replay_do_not_duplicate_jobs(self):
        self.sources()
        first = self.index.backfill_batch(page_size=2)
        self.assertEqual((first["scanned"], first["submitted"]), (2, 2))
        self.assertEqual(len(self.queued()), 2)
        cursor = json.loads(json.dumps(first["next_cursor"]))
        second = self.index.backfill_batch(cursor=cursor, page_size=2)
        replay = self.index.backfill_batch(cursor=cursor, page_size=2)
        self.assertEqual(second["submitted"], 2)
        self.assertEqual(replay["submitted"], 0)
        self.drain(second["next_cursor"])
        self.assertEqual(len(self.queued()), 5)
        self.assertEqual(len(self.jobs.db.rows(outbox)), 5)
        self.assertEqual(sum(r["submitted"] for r in self.drain()), 0)

    def test_empty_workspace_finishes_without_jobs(self):
        pages = self.drain()
        self.assertTrue(pages[-1]["done"])
        self.assertEqual(sum(p["scanned"] for p in pages), 0)

    def test_invalid_page_sizes_are_rejected(self):
        for value in (0, -1, 101, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.index.backfill_batch(page_size=value)
        self.assertEqual(self.jobs.db.rows(memory_indexes), [])

    def test_cursor_bound_to_workspace_database_generation_and_policy(self):
        self.sources()
        cursor = self.index.backfill_batch(page_size=1)["next_cursor"]
        for key in ("workspace", "database", "generation", "fingerprint", "retry", "schema"):
            bad = dict(cursor, **{key: "different"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.index.backfill_batch(cursor=bad)
        self.assertEqual(len(self.queued()), 1)

    def test_malformed_cursors_are_rejected(self):
        self.sources()
        cursor = self.index.backfill_batch(page_size=1)["next_cursor"]
        for bad in ([], {}, "text", dict(cursor, extra=1), dict(cursor, phase="unknown"),
                    dict(cursor, after=-1), dict(cursor, source_until=True),
                    dict(cursor, after=cursor["source_until"] + 1)):
            with self.subTest(cursor=bad), self.assertRaises(ValueError):
                self.index.backfill_batch(cursor=bad)

    def test_concurrent_insert_is_incremental_not_added_to_scan(self):
        self.sources()
        first = self.index.backfill_batch(page_size=2)
        late = self.add(source_run="new-incremental-case")
        self.assertGreater(late, first["next_cursor"]["source_until"])
        rest = self.drain(first["next_cursor"])
        self.assertEqual(sum(r["scanned"] for r in rest), 3)
        self.assertEqual(sum(r["submitted"] for r in rest), 3)
        self.assertEqual(len(self.queued()), 6)

    def test_pages_read_current_source_revision_not_stale_snapshot(self):
        ids = self.sources()
        first = self.index.backfill_batch(page_size=2)
        self.add(source_run="batch-fixture-4", evidence={"test_summary": "latest fixture revision"})
        current = self.entry(ids[-1])
        self.drain(first["next_cursor"])
        self.assertEqual(self.entry(ids[-1])["case_version"], current["case_version"])
        self.assertEqual(len(self.queued()), 5)

    def test_batch_only_enqueues_configured_generation(self):
        mid = self.add()
        old_job = self.entry(mid)["job_id"]
        self.cfg["indexing"]["generation"] = "second-generation"
        self.configure()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, jobs, {"status": "dead"}, jobs.c.id == old_job)
        self.drain(retry=True)
        self.assertEqual(len(self.queued()), 2)

    def test_deleted_sources_get_cleanup_jobs(self):
        mid = self.add()
        self.run_job(self.queued()[0])
        with self.jobs.db.transaction() as conn:
            self.jobs.db.delete(conn, memories, memories.c.id == mid)
        pages = self.drain()
        self.assertEqual(sum(r["submitted"] for r in pages), 1)
        self.run_job(self.queued()[-1])
        self.assertEqual(self.vectors.points, {})
        self.assertTrue(self.jobs.db.rows(memory_index_chunks))

    def test_failed_jobs_require_explicit_retry(self):
        mid = self.add()
        with self.jobs.db.transaction() as conn:
            self.jobs.db.update(conn, jobs, {"status": "dead"}, jobs.c.id == self.entry(mid)["job_id"])
        self.assertEqual(sum(r["submitted"] for r in self.drain()), 0)
        self.assertEqual(sum(r["submitted"] for r in self.drain(retry=True)), 1)
        self.assertEqual(sum(r["submitted"] for r in self.drain(retry=True)), 0)

    def test_page_rolls_back_if_enqueue_fails(self):
        self.sources()
        original = self.jobs.__class__._enqueue
        def fail(*args):
            original(*args)
            raise RuntimeError("enqueue failure fixture")
        with patch.object(self.jobs.__class__, "_enqueue", autospec=True, side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.index.backfill_batch(page_size=2)
        for table in (jobs, outbox, memory_index_entries, memory_indexes):
            self.assertEqual(self.jobs.db.rows(table), [])
        self.assertEqual(self.index.backfill_batch(page_size=2)["submitted"], 2)

    def test_cli_cursor_round_trip(self):
        self.sources()
        def call(*args):
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["indexing", "backfill-batch", "--workspace", str(self.root),
                                           "--batch-size", "2", *args]), contextlib.redirect_stdout(stdout):
                main()
            return json.loads(stdout.getvalue())
        first = call()
        second = call("--cursor", json.dumps(first["next_cursor"]))
        self.assertEqual(second["submitted"], 2)
        self.assertFalse(second["retrieval_changed"])
