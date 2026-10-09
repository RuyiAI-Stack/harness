"""Optional read-through Redis cache. MySQL remains the version authority."""
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
import random
import secrets
import threading
import time
from urllib.parse import urlsplit

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from codex_agent.runtime_config import runtime_config


class CacheBusy(RuntimeError):
    pass


class CacheUnavailable(RuntimeError):
    pass


class VersionChanged(RuntimeError):
    pass


RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
PUBLISH = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'PX', ARGV[3])
redis.call('DEL', KEYS[1])
return 1
"""
RATE = """
local now = tonumber(redis.call('TIME')[1])
for i = 1, #KEYS do
  local entry = redis.call('HMGET', KEYS[i], 'second', 'count')
  if tonumber(entry[1]) == now and tonumber(entry[2]) >= tonumber(ARGV[i]) then return 0 end
end
for i = 1, #KEYS do
  if tonumber(redis.call('HGET', KEYS[i], 'second')) ~= now then
    redis.call('HSET', KEYS[i], 'second', now, 'count', 0)
  end
  redis.call('HINCRBY', KEYS[i], 'count', 1)
  redis.call('EXPIRE', KEYS[i], 2)
end
return 1
"""
BLOOM_CHECK = """
if redis.call('EXISTS', KEYS[1]) == 0 or redis.call('GET', KEYS[2]) ~= 'ready' then return -1 end
return redis.call('BF.EXISTS', KEYS[1], ARGV[1])
"""
BLOOM_CREATE = """
redis.call('BF.RESERVE', KEYS[1], ARGV[1], ARGV[2], 'NONSCALING')
redis.call('EXPIRE', KEYS[1], 120)
return 1
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class RedisBackend:
    def __init__(self, url):
        if urlsplit(url).scheme not in {"redis", "rediss", "unix"}:
            raise ValueError("Cache URL must use redis, rediss or unix")
        try:
            self.client = redis.Redis.from_url(url, socket_timeout=0.2,
                socket_connect_timeout=0.2, max_connections=32,
                retry=Retry(NoBackoff(), 0), decode_responses=False)
        except (ValueError, TypeError):
            raise ValueError("Invalid cache connection URL") from None
        self.lock = threading.Lock()
        self.metrics = Counter()
        self.retry_at = 0.0
        self.policy_until = 0.0
        self.fallback_slots = threading.BoundedSemaphore(2)
        self.fallback_times = []

    def count(self, name):
        with self.lock:
            self.metrics[name] += 1

    def snapshot(self):
        with self.lock:
            return dict(self.metrics)

    def call(self, function, *args, **kwargs):
        if time.monotonic() < self.retry_at:
            raise CacheUnavailable("Redis temporarily unavailable")
        try:
            if time.monotonic() >= self.policy_until:
                settings = self.client.config_get('maxmemory*')
                policy = settings.get('maxmemory-policy')
                if policy not in {b'noeviction', 'noeviction'} or int(settings.get('maxmemory', 0)) <= 0:
                    raise redis.RedisError("coordination keys must not be evicted")
                self.policy_until = time.monotonic() + 5
            return function(*args, **kwargs)
        except redis.RedisError:
            self.retry_at = time.monotonic() + 2
            self.count("redis_errors")
            logging.getLogger(__name__).warning(
                "Redis read cache unavailable; bounded MySQL fallback enabled for two seconds")
            raise CacheUnavailable("Redis unavailable; require bounded memory and noeviction") from None

    @contextmanager
    def fallback(self):
        with self.lock:
            now = time.monotonic()
            self.fallback_times = [t for t in self.fallback_times if t > now - 1]
            allowed = len(self.fallback_times) < 4
            if allowed:
                self.fallback_times.append(now)
        if not allowed or not self.fallback_slots.acquire(blocking=False):
            self.count("fallback_rejected")
            raise CacheBusy("Historical retrieval is busy while Redis recovers; retry later")
        try:
            self.count("fallback")
            yield
        finally:
            self.fallback_slots.release()


@lru_cache(maxsize=8)
def _backend(url, pid):
    return RedisBackend(url)


class ReadCache:
    def __init__(self, database, domain="memory"):
        self.db = database
        self.domain = domain
        self.config = runtime_config().cache
        self.backend = None
        self.prefix = f"triton:cache:v1:{database.cache_namespace}:{database.workspace_id}:{domain}"
        self.global_prefix = f"triton:cache:v1:{database.cache_namespace}"
        if self.config.enabled:
            raw = os.environ.get(self.config.urlEnv, "")
            if not raw:
                raise ValueError(f"Set {self.config.urlEnv} before enabling Redis caching")
            self.backend = _backend(raw, os.getpid())

    def _rate(self, operation, limit):
        keys = [self.prefix + ':rate:' + operation, self.global_prefix + ':rate:' + operation]
        if not self.backend.call(self.backend.client.eval, RATE, 2, *keys, limit, limit * 4):
            self.backend.count("rate_rejected")
            raise CacheBusy("Historical retrieval rate limit reached; retry later")

    def key(self, kind, parameters, revision):
        return f"{self.prefix}:{revision}:{kind}:{digest(parameters)}"

    def _compute(self, loader, revision):
        with self.db.cache_read_slot(self.config.maxConcurrent):
            self.backend.count("rebuilds")
            result = loader()
            if self.db.revision(self.domain) != revision:
                raise VersionChanged()
            return result

    def get(self, kind, parameters, loader, *, exact_id=None):
        if self.backend is None:
            return loader()
        backend, cfg = self.backend, self.config
        backend.count("requests")
        try:
            self._rate('request', cfg.requestsPerSecond)
            for _ in range(2):
                revision = self.db.revision(self.domain)
                try:
                    return self._get(kind, parameters, loader, revision, exact_id)
                except VersionChanged:
                    backend.count("version_retries")
            raise CacheBusy("History changed during retrieval; retry with the new revision")
        except CacheUnavailable:
            # No unrestricted fail-open. DB named slots also bound independent processes.
            with backend.fallback():
                for _ in range(2):
                    revision = self.db.revision(self.domain)
                    try:
                        return self._compute(loader, revision)
                    except VersionChanged:
                        backend.count("version_retries")
                raise CacheBusy("History changed during retrieval; retry later")

    def _get(self, kind, parameters, loader, revision, exact_id):
        backend, client, cfg = self.backend, self.backend.client, self.config
        key = self.key(kind, parameters, revision)
        owner = secrets.token_hex(24)
        lock_key = key + ':lock'
        end = time.monotonic() + cfg.waitMs / 1000
        while True:
            cached = backend.call(client.get, key)
            if cached is not None:
                try:
                    envelope = json.loads(cached)
                    if not isinstance(envelope, dict) or set(envelope) != {'value'}:
                        raise ValueError()
                    backend.count("negative_hits" if envelope['value'] is None else "hits")
                    return envelope['value']
                except (ValueError, TypeError, UnicodeError):
                    backend.count("invalid_entries")
                    backend.call(client.delete, key)
            if exact_id is not None and cfg.bloom:
                bloom = f"{self.prefix}:{revision}:ids"
                exists = backend.call(client.eval, BLOOM_CHECK, 2, bloom, bloom + ':ready', str(exact_id))
                if exists == 0:
                    backend.count("bloom_negatives")
                    return None
            if backend.call(client.set, lock_key, owner, nx=True, px=cfg.lockMs):
                break
            backend.count("lock_waits")
            if time.monotonic() >= end:
                raise CacheBusy("Historical cache is rebuilding; retry later")
            time.sleep(random.uniform(0.01, 0.025))
        try:
            # Recheck after acquiring: a previous owner may have published between GET/SET.
            cached = backend.call(client.get, key)
            if cached is not None:
                try:
                    envelope = json.loads(cached)
                    if not isinstance(envelope, dict) or set(envelope) != {'value'}:
                        raise ValueError()
                    backend.count("negative_hits" if envelope['value'] is None else "hits")
                    return envelope['value']
                except (ValueError, TypeError, UnicodeError):
                    backend.count("invalid_entries")
                    backend.call(client.delete, key)
            self._rate('rebuild', cfg.rebuildsPerSecond)
            result = self._compute(loader, revision)
            payload = json.dumps({'value': result}, ensure_ascii=False, allow_nan=False).encode()
            ttl = cfg.negativeTtlSeconds if exact_id is not None and result is None else cfg.ttlSeconds
            ttl_ms = max(1, int(ttl * 1000 * random.uniform(0.8, 1.2)))
            if len(payload) <= cfg.maxBytes:
                try:
                    published = backend.call(client.eval, PUBLISH, 2, lock_key, key, owner, payload, ttl_ms)
                except CacheUnavailable:
                    # A successful DB read is not repeated just because cache storage failed.
                    return result
                if not published:
                    backend.count("lease_lost")
                    raise CacheBusy("Cache rebuild lease expired; retry later")
            else:
                backend.count("oversized")
            backend.count("misses")
            return result
        finally:
            try:
                backend.call(client.eval, RELEASE, 1, lock_key, owner)
            except CacheUnavailable:
                pass

    def warm_bloom(self, identifiers):
        """Build from authoritative IDs; publish only for an unchanged DB revision."""
        if not self.backend or not self.config.bloom:
            return {"enabled": False}
        backend, client = self.backend, self.backend.client
        revision = self.db.revision(self.domain)
        self._rate('rebuild', self.config.rebuildsPerSecond)
        values = self._compute(identifiers, revision)
        name = f"{self.prefix}:{revision}:ids"
        temporary = name + ':' + secrets.token_hex(12)
        try:
            backend.call(client.eval, BLOOM_CREATE, 1, temporary, self.config.bloomErrorRate,
                         max(self.config.bloomCapacity, math.ceil(len(values) * 1.5)))
            for start in range(0, len(values), 500):
                backend.call(client.execute_command, 'BF.MADD', temporary, *values[start:start + 500])
            if self.db.revision(self.domain) != revision:
                raise VersionChanged()
            pipe = client.pipeline(transaction=True)
            pipe.rename(temporary, name)
            pipe.expire(name, 600)
            pipe.set(name + ':ready', 'ready', ex=590)
            backend.call(pipe.execute)
            return {"enabled": True, "revision": revision, "ids": len(values), "ttl_seconds": 590}
        finally:
            try:
                backend.call(client.delete, temporary)
            except CacheUnavailable:
                pass
