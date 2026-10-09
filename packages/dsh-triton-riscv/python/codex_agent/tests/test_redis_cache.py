"""Redis/MySQL integration tests; only disposable services may be supplied."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import redis
from sqlalchemy import select

from codex_agent.memory import MemoryQuery, MemoryStore
from codex_agent.process_control import validation_environment
from codex_agent.runtime_config import CONFIG_ENV, runtime_config
from codex_agent.storage.cache import CacheBusy, RELEASE, VersionChanged
from codex_agent.storage.schema import memories
from codex_agent.tests.test_memory import record


@unittest.skipUnless(os.environ.get("TRITON_TEST_REDIS_URL") and os.environ.get("TRITON_MYSQL_URL"),
                     "disposable Redis and MySQL required")
class RedisCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"enabled": True, "requestsPerSecond": 10000, "rebuildsPerSecond": 1000}
        self.env = patch.dict(os.environ, {"TRITON_REDIS_URL": os.environ["TRITON_TEST_REDIS_URL"],
            CONFIG_ENV: self.document()})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store = MemoryStore(self.root)
        self.backend = self.store.cache.backend
        self.backend.retry_at = 0
        self.backend.fallback_times = []
        self.client = self.backend.client
        self.query = MemoryQuery("relu_and_mul", "relu multiply", "torch.relu(x) * y")

    def document(self):
        return json.dumps({"schemaVersion": 1, "repoRoot": str(self.root),
            "stateDir": str(self.root / 'state'), "cache": self.cfg})

    def reconfigure(self, **kwargs):
        self.cfg.update(kwargs)
        os.environ[CONFIG_ENV] = self.document()
        self.store = MemoryStore(self.root)
        return self.store

    def test_same_query_reuses_ranking_and_keeps_usage_update(self):
        self.store.add(record(operator="relu_and_mul"))
        with patch.object(self.store, "rank_candidates", wraps=self.store.rank_candidates) as rank:
            first, second = self.store.retrieve(self.query), self.store.retrieve(self.query)
            self.assertEqual(first, second)
            self.assertEqual(rank.call_count, 1)
        self.assertIsNotNone(self.store.db.rows(memories)[0]["last_used_at"])

    def test_negative_cache_and_creation_invalidation(self):
        with patch.object(self.store.db, "rows", wraps=self.store.db.rows) as rows:
            self.assertIsNone(self.store.get(1))
            self.assertIsNone(self.store.get(1))
            self.assertEqual(sum(c.args[0] is memories for c in rows.call_args_list), 1)
        identifier, _ = self.store.add(record())
        self.assertEqual(self.store.get(identifier)["id"], identifier)

    def test_archive_stats_and_write_invalidation(self):
        identifier, _ = self.store.add(record())
        self.assertEqual(self.store.stats()["active"], 1)
        self.assertIsNotNone(self.store.get(identifier))
        self.store.archive(identifier, "test")
        self.assertIsNone(self.store.get(identifier))
        self.assertEqual(self.store.stats()["active"], 0)

    def test_failed_write_does_not_publish_revision(self):
        self.store.add(record())
        before = self.store.db.revision("memory")
        with self.assertRaises(RuntimeError):
            with self.store.db.transaction() as conn:
                self.store.db.update(conn, memories, {"summary": "not committed"})
                raise RuntimeError("rollback")
        self.assertEqual(self.store.db.revision("memory"), before)
        self.assertNotEqual(self.store.get(1)["summary"], "not committed")

    def test_query_environment_mode_limit_and_exclusions_do_not_share_cache(self):
        self.store.add(record(operator="relu_and_mul"))
        with patch.object(self.store, "rank_candidates", wraps=self.store.rank_candidates) as rank:
            self.store.retrieve(self.query)
            self.store.retrieve(self.query, limit=1)
            self.store.retrieve(self.query, score_mode="jaccard")
            self.store.retrieve(self.query, exclude_source_runs=["different-run"])
            self.store.retrieve(MemoryQuery("relu_and_mul", "relu multiply", "torch.relu(x) * y",
                                          environment={"triton": "different-version"}))
            self.assertEqual(rank.call_count, 5)

    def test_workspace_isolation(self):
        self.store.add(record())
        other = MemoryStore(self.root / "other")
        self.assertIsNotNone(self.store.get(1))
        self.assertIsNone(other.get(1))

    def test_shared_tool_retrieval_path_uses_cache_without_changing_context(self):
        from codex_agent.diagnostic_memory import retrieve_memories
        self.store.add(record(operator='relu_and_mul'))
        original = MemoryStore.rank_candidates
        with patch.object(MemoryStore, 'rank_candidates', autospec=True, side_effect=original) as rank:
            first = retrieve_memories(self.root, operator_name='relu_and_mul')
            second = retrieve_memories(self.root, operator_name='relu_and_mul')
            self.assertEqual(first['status'], 'found')
            self.assertEqual(first, second)
            self.assertEqual(rank.call_count, 1)

    def test_cached_reference_still_rechecks_source_evidence(self):
        from codex_agent.tests import test_reference_library as fixtures
        fixture = fixtures.ReferenceLibraryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build()
        self.assertEqual(len(fixtures.search_library(fixture.output, 'demo', environment=fixtures.ENV)['items']), 1)
        fixture.impl.write_text(fixtures.IMPLEMENTATION + '# changed\n')
        result = fixtures.search_library(fixture.output, 'demo', environment=fixtures.ENV)
        self.assertEqual(result['items'], [])
        self.assertEqual(result['rejected'][0]['reason'], 'source-or-evidence-changed')

    def test_same_key_concurrency_has_one_rebuild(self):
        barrier = threading.Barrier(12)
        count = 0
        def load():
            nonlocal count
            count += 1
            time.sleep(0.15)
            return {"ok": True}
        def request(_):
            barrier.wait()
            return self.store.cache.get("concurrency", {}, load)
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(request, range(12)))
        self.assertEqual(count, 1)
        self.assertTrue(all(r == {"ok": True} for r in results))

    def test_lease_expiry_cannot_publish_or_delete_successors_lock(self):
        self.reconfigure(lockMs=100)
        entered = threading.Event()
        def slow():
            entered.set()
            time.sleep(0.3)
            return "old"
        with ThreadPoolExecutor(max_workers=2) as pool:
            old = pool.submit(self.store.cache.get, "lease", {}, slow)
            entered.wait(2)
            time.sleep(0.15)
            self.assertEqual(self.store.cache.get("lease", {}, lambda: "new"), "new")
            with self.assertRaises(CacheBusy):
                old.result()
        self.assertEqual(self.store.cache.get("lease", {}, lambda: "wrong"), "new")
        key = self.store.cache.prefix + ':owner-test'
        self.client.set(key, 'successor', px=1000)
        self.assertEqual(self.client.eval(RELEASE, 1, key, 'predecessor'), 0)
        self.assertEqual(self.client.get(key), b'successor')

    def test_crashed_process_lock_expires(self):
        cache = self.store.cache
        key = cache.key("crash", {}, self.store.db.revision("memory")) + ':lock'
        code = "import os,redis,sys; r=redis.Redis.from_url(os.environ['TRITON_REDIS_URL']); r.set(sys.argv[1],'crashed',px=150); os._exit(0)"
        subprocess.run([sys.executable, '-I', '-c', code, key], check=True, timeout=10)
        self.assertEqual(cache.get("crash", {}, lambda: "recovered"), "recovered")

    def test_concurrent_write_during_load_retries_new_version(self):
        self.store.add(record())
        count = 0
        def load():
            nonlocal count
            count += 1
            old = self.store.db.rows(memories)[0]["summary"]
            if count == 1:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(self.store.archive, 1, "changed").result()
            return old if count == 1 else "new"
        self.assertEqual(self.store.cache.get("changed", {}, load), "new")
        self.assertEqual(count, 2)

    def test_error_is_not_negative_cached(self):
        with self.assertRaisesRegex(RuntimeError, "database fixture"):
            self.store.cache.get("error", {}, lambda: (_ for _ in ()).throw(RuntimeError("database fixture")), exact_id=1)
        self.assertEqual(self.store.cache.get("error", {}, lambda: {"exists": True}, exact_id=1), {"exists": True})

    def test_empty_semantic_results_are_not_bloom_rejections(self):
        self.reconfigure(bloom=True)
        result = self.store.retrieve(self.query)
        self.assertEqual(result, [])
        key = self.store.cache.key('detail', {'id': 1000}, 0)
        self.assertFalse(self.client.exists(key))

    def test_positive_and_negative_ttls_are_jittered(self):
        for n in range(12):
            self.store.cache.get("ttl", {"n": n}, lambda: {"value": True})
        ttls = [self.client.pttl(self.store.cache.key("ttl", {"n": n}, 0)) for n in range(12)]
        self.assertTrue(all(47000 < ttl <= 72000 for ttl in ttls))
        self.assertGreater(max(ttls) - min(ttls), 500)
        self.store.get(1)
        self.assertTrue(15000 < self.client.pttl(self.store.cache.key("detail", {"id": 1}, 0)) <= 24000)

    def test_oversized_values_are_not_admitted(self):
        self.reconfigure(maxBytes=1024)
        calls = []
        for _ in range(2):
            self.store.cache.get("large", {}, lambda: calls.append(1) or 'x' * 2048)
        self.assertEqual(len(calls), 2)

    def test_request_rate_is_shared_between_store_instances(self):
        self.reconfigure(requestsPerSecond=1)
        # Freeze Redis's one-second bucket by starting near the beginning of a second.
        while self.client.time()[1] > 100000:
            time.sleep(0.01)
        self.store.get(999)
        with self.assertRaises(CacheBusy):
            MemoryStore(self.root).get(998)

    def test_db_admission_blocks_other_process_and_releases_on_exit(self):
        with self.store.db.cache_read_slot(1):
            code = ("from pathlib import Path; import sys; from codex_agent.storage.database import WorkspaceDatabase; "
                    "from codex_agent.storage.cache import CacheBusy; "
                    "db=WorkspaceDatabase(Path(sys.argv[1]));\n"
                    "try:\n with db.cache_read_slot(1): print('unexpected')\n"
                    "except CacheBusy: print('blocked')\n")
            result = subprocess.run([sys.executable, '-I', '-c', code, str(self.root)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'blocked')
        with self.store.db.cache_read_slot(1):
            pass

    def test_real_connection_failure_has_bounded_fallback(self):
        os.environ['TRITON_REDIS_URL'] = 'unix://' + str(self.root / 'missing.sock')
        store = MemoryStore(self.root)
        self.assertEqual(store.stats()['total'], 0)
        for _ in range(3):
            store.stats()
        with self.assertRaises(CacheBusy):
            store.stats()
        self.assertGreater(store.cache.backend.snapshot()['fallback'], 0)

    def test_oom_does_not_turn_successful_read_into_missing(self):
        original = self.client.config_get('maxmemory')['maxmemory']
        self.client.config_set('maxmemory', 1)
        try:
            self.assertEqual(self.store.cache.get('oom', {}, lambda: {'real': True}), {'real': True})
        finally:
            self.client.config_set('maxmemory', original)
            self.backend.retry_at = 0

    def test_credentials_are_not_given_to_generated_code(self):
        self.assertNotIn('TRITON_REDIS_URL', validation_environment())
        self.assertNotIn('TRITON_REDIS_URL', self.document())

    def test_bloom_rebuild_new_record_and_loss(self):
        if not any(m.get('name', m.get(b'name')) in {'bf', b'bf'} for m in self.client.module_list()):
            self.skipTest('Redis Bloom commands unavailable')
        self.reconfigure(bloom=True)
        self.store.add(record())
        self.assertEqual(self.store.warm_cache_ids()['ids'], 1)
        with patch.object(self.store.db, 'rows', wraps=self.store.db.rows) as rows:
            self.assertIsNone(self.store.get(99999))
            self.assertEqual(sum(c.args[0] is memories for c in rows.call_args_list), 0)
        self.assertIsNotNone(self.store.get(1))
        new_id, _ = self.store.add(record(operator='new'))
        self.assertIsNotNone(self.store.get(new_id))
        self.store.warm_cache_ids()
        revision = self.store.db.revision('memory')
        self.client.delete(f'{self.store.cache.prefix}:{revision}:ids')
        self.assertIsNone(self.store.get(123456))

    def test_malformed_entry_is_rebuilt(self):
        key = self.store.cache.key('corrupt', {}, 0)
        for value in ('not json', '{"wrong": 1}', '[]'):
            self.client.set(key, value, ex=30)
            self.assertEqual(self.store.cache.get('corrupt', {}, lambda: {'ok': True}), {'ok': True})

    def test_eviction_policy_cannot_silently_evict_locks(self):
        self.client.config_set('maxmemory-policy', 'allkeys-lfu')
        self.backend.policy_until = 0
        try:
            before = self.backend.snapshot().get('fallback', 0)
            self.assertEqual(self.store.stats()['total'], 0)
            self.assertEqual(self.backend.snapshot()['fallback'], before + 1)
        finally:
            self.client.config_set('maxmemory-policy', 'noeviction')
            self.backend.retry_at = self.backend.policy_until = 0

    def test_independent_processes_share_one_rebuild(self):
        code = ("from pathlib import Path; import sys,time,secrets; "
                "from codex_agent.memory import MemoryStore; root=Path(sys.argv[1]); s=MemoryStore(root);\n"
                "def load():\n (root / ('rebuild-' + secrets.token_hex(8))).write_text('fixture'); "
                "time.sleep(.2); return {'ok':True}\n"
                "assert s.cache.get('process-hot', {}, load)=={'ok':True}\n")
        def worker(_):
            result = subprocess.run([sys.executable, '-I', '-c', code, str(self.root)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(worker, range(6)))
        self.assertEqual(len(list(self.root.glob('rebuild-*'))), 1)

    def test_real_redis_stop_and_restart_recovers(self):
        binary = os.environ.get('TRITON_TEST_REDIS_SERVER')
        if not binary:
            self.skipTest('temporary Redis executable not supplied')
        sock = self.root / 'recovery.sock'
        command = [binary, '--port', '0', '--unixsocket', str(sock), '--save', '',
                   '--appendonly', 'no', '--maxmemory', '64mb', '--maxmemory-policy', 'noeviction']
        processes = []
        def start():
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(process)
            client = redis.Redis(unix_socket_path=str(sock))
            for _ in range(100):
                try:
                    client.ping()
                    return process
                except redis.RedisError:
                    time.sleep(.02)
            self.fail('Redis did not restart')
        try:
            process = start()
            os.environ['TRITON_REDIS_URL'] = 'unix://' + str(sock)
            store = MemoryStore(self.root)
            self.assertEqual(store.stats()['total'], 0)
            process.terminate()
            process.wait(timeout=5)
            self.assertEqual(store.stats()['total'], 0)
            self.assertEqual(store.cache.backend.snapshot().get('fallback'), 1)
            start()
            time.sleep(2.1)
            self.assertEqual(store.stats()['total'], 0)
            self.assertEqual(store.stats()['total'], 0)
            self.assertGreaterEqual(store.cache.backend.snapshot().get('hits', 0), 1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)


class CacheConfigurationTests(unittest.TestCase):
    def test_defaults_off_and_invalid_settings_rejected(self):
        self.assertFalse(runtime_config({}).cache.enabled)
        for cfg in ({'enabled': 'true'}, {'lockMs': 0}, {'maxConcurrent': 0},
                    {'urlEnv': 'PATH'}, {'urlEnv': 'TRITON_MYSQL_URL'},
                    {'bloomErrorRate': 0}, {'maxBytes': 0}, {'unknown': 1}):
            with self.assertRaises(ValueError):
                runtime_config({CONFIG_ENV: json.dumps({'schemaVersion': 1,
                    'repoRoot': '/tmp/project', 'stateDir': '/tmp/state', 'cache': cfg})})


if __name__ == '__main__':
    unittest.main()
