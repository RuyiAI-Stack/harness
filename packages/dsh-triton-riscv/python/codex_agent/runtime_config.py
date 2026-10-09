"""Validated host configuration shared by native tools and the approval bridge.

TRITON_RISCV_CONFIG carries settings, never credentials. Old environment names
are read only at this compatibility boundary when no native document is present.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

CONFIG_ENV = "TRITON_RISCV_CONFIG"
MAX_CONFIG_BYTES = 65536


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class Permissions(Settings):
    validation: bool = False
    development: bool = False
    repair: bool = False


class Remote(Settings):
    host: str = ""
    repository: str = ""
    required: bool = False
    requireTaskQuotas: bool = False


class Embedding(Settings):
    provider: str = "none"
    model: str | None = None
    baseUrl: str | None = None
    tokenizerJson: str | None = None
    tokenBudget: int | None = Field(default=None, gt=0)
    apiKeyEnv: str = "AGENT_EMBEDDING_API_KEY"


class Storage(Settings):
    urlEnv: str = "TRITON_MYSQL_URL"
    poolSize: int = Field(default=5, ge=1, le=50)
    maxOverflow: int = Field(default=5, ge=0, le=50)
    poolTimeout: int = Field(default=10, ge=1, le=60)


class Cache(Settings):
    enabled: bool = False
    urlEnv: str = "TRITON_REDIS_URL"
    ttlSeconds: int = Field(default=60, ge=1, le=3600)
    negativeTtlSeconds: int = Field(default=20, ge=1, le=60)
    lockMs: int = Field(default=10000, ge=100, le=60000)
    waitMs: int = Field(default=1000, ge=10, le=5000)
    maxBytes: int = Field(default=262144, ge=1024, le=1048576)
    maxConcurrent: int = Field(default=4, ge=1, le=32)
    requestsPerSecond: int = Field(default=100, ge=1, le=10000)
    rebuildsPerSecond: int = Field(default=10, ge=1, le=1000)
    bloom: bool = False
    bloomCapacity: int = Field(default=10000, ge=100, le=10000000)
    bloomErrorRate: float = Field(default=0.001, gt=0, le=0.1)


class Memory(Settings):
    retrievalMode: str = "legacy"
    contextFormat: str = "classic"
    embedding: Embedding = Field(default_factory=Embedding)


class Queue(Settings):
    enabled: bool = False
    urlEnv: str = "TRITON_AMQP_URL"
    leaseSeconds: int = Field(default=60, ge=15, le=600)
    maxAttempts: int = Field(default=3, ge=1, le=10)
    jobTimeoutSeconds: int = Field(default=1200, ge=30, le=7200)
    maxPending: int = Field(default=100, ge=1, le=10000)


class RuntimeConfig(Settings):
    schemaVersion: Literal[1] = 1
    repoRoot: str = ""
    stateDir: str = ""
    permissions: Permissions = Field(default_factory=Permissions)
    remote: Remote = Field(default_factory=Remote)
    memory: Memory = Field(default_factory=Memory)
    storage: Storage = Field(default_factory=Storage)
    cache: Cache = Field(default_factory=Cache)
    queue: Queue = Field(default_factory=Queue)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration field")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("non-finite configuration value")


def _single_line(value):
    if isinstance(value, dict):
        for item in value.values():
            _single_line(item)
    elif isinstance(value, str) and any(c in value for c in "\0\r\n"):
        raise ValueError("configuration string contains control characters")


@lru_cache(maxsize=16)
def _parse_native(raw: str) -> RuntimeConfig:
    try:
        data = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        if not isinstance(data, dict) or type(data.get("schemaVersion")) is not int or data["schemaVersion"] != 1:
            raise ValueError("unsupported configuration version")
        _single_line(data)
        cfg = RuntimeConfig.model_validate(data)
        for path in (cfg.repoRoot, cfg.stateDir):
            if not path or not Path(path).is_absolute():
                raise ValueError("workspace and state paths must be absolute")
        for path in (cfg.memory.embedding.tokenizerJson,):
            if path and not Path(path).is_absolute():
                raise ValueError("memory paths must be absolute")
        remote = cfg.remote
        if bool(remote.host) != bool(remote.repository) or (remote.required and not remote.host):
            raise ValueError("remote target must be complete")
        if remote.host and not re.fullmatch(r"[A-Za-z0-9_.-]+", remote.host):
            raise ValueError("invalid remote host")
        if remote.repository and (not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote.repository)
                                  or ".." in remote.repository.split("/")):
            raise ValueError("invalid remote repository")
        if remote.requireTaskQuotas and not remote.required:
            raise ValueError("task quotas require remote mode")
        db_key = cfg.storage.urlEnv
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", db_key)
                or db_key.startswith(("TRITON_RISCV_", "RISCV_"))
                or db_key in {"PATH", "HOME", "PYTHONPATH", "PYTHONHOME"}):
            raise ValueError("reserved or invalid database credential variable")
        cache_key = cfg.cache.urlEnv
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cache_key)
                or cache_key.startswith(("TRITON_RISCV_", "RISCV_"))
                or cache_key in {"PATH", "HOME", "PYTHONPATH", "PYTHONHOME", db_key}):
            raise ValueError("reserved or invalid cache credential variable")
        key = cfg.memory.embedding.apiKeyEnv
        mq_key = cfg.queue.urlEnv
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", mq_key)
                or mq_key.startswith(("TRITON_RISCV_", "RISCV_"))
                or mq_key in {"PATH", "HOME", "PYTHONPATH", "PYTHONHOME", db_key, cache_key, key}):
            raise ValueError("reserved or invalid queue credential variable")
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or key.startswith(("TRITON_RISCV_", "RISCV_"))
                or key in {"PATH", "HOME", "PYTHONPATH", "PYTHONHOME"}):
            raise ValueError("reserved or invalid credential variable")
        return cfg
    except (ValueError, TypeError, RecursionError):
        # Do not echo supplied values: malformed documents may accidentally include secrets.
        raise ValueError("Invalid TRITON_RISCV_CONFIG: check version, fields, types and absolute paths") from None


def runtime_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = os.environ if environ is None else environ
    if CONFIG_ENV in env:
        raw = env[CONFIG_ENV]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise ValueError("TRITON_RISCV_CONFIG exceeds 64 KiB or is not text")
        return _parse_native(raw)
    return _legacy_config(env)


def _legacy_config(env: Mapping[str, str]) -> RuntimeConfig:
    budget = env.get("TRITON_RISCV_EMBEDDING_TOKEN_BUDGET")
    return RuntimeConfig(
        repoRoot=env.get("TRITON_RISCV_REPO_ROOT") or env.get("TRITON_RISCV_CHECKOUT") or "",
        stateDir=env.get("TRITON_RISCV_STATE_DIR") or "",
        permissions=Permissions(
            validation=env.get("TRITON_RISCV_ALLOW_VALIDATION") == "1",
            development=env.get("TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY") == "1",
            repair=env.get("TRITON_RISCV_ALLOW_REPAIR_APPLY") == "1"),
        remote=Remote(
            host=env.get("RISCV_HOST", "").strip(),
            repository=env.get("RISCV_REPO", "").strip(),
            required=env.get("TRITON_RISCV_REQUIRE_REMOTE") == "1",
            requireTaskQuotas=env.get("TRITON_RISCV_REQUIRE_TASK_QUOTAS", "").strip().lower() in {"1", "true"}),
        memory=Memory(
            retrievalMode=env.get("TRITON_RISCV_MEMORY_RETRIEVAL_MODE", "legacy"),
            contextFormat=env.get("TRITON_RISCV_MEMORY_CONTEXT_FORMAT", "classic"),
            embedding=Embedding(
                provider=env.get("TRITON_RISCV_EMBEDDING_PROVIDER", "none"),
                model=env.get("TRITON_RISCV_EMBEDDING_MODEL"),
                baseUrl=env.get("TRITON_RISCV_EMBEDDING_BASE_URL"),
                tokenizerJson=env.get("TRITON_RISCV_EMBEDDING_TOKENIZER_JSON"),
                tokenBudget=int(budget) if budget else None,
                apiKeyEnv=env.get("TRITON_RISCV_EMBEDDING_API_KEY_ENV", "AGENT_EMBEDDING_API_KEY"))))


def permission_enabled(kind: str) -> bool:
    field = {"development": "development", "repair": "repair", "validation": "validation", "job": "validation"}.get(kind)
    return bool(field and getattr(runtime_config().permissions, field))


def approval_required() -> bool:
    if CONFIG_ENV in os.environ:
        runtime_config()
        return True
    return os.environ.get("TRITON_RISCV_REQUIRE_APPROVED_VALIDATION") == "1"


def remote_execution_identity() -> list:
    """Keep existing journal fingerprints valid across the transport migration."""
    if CONFIG_ENV not in os.environ:
        return [os.environ.get(key) for key in ("RISCV_HOST", "RISCV_REPO", "TRITON_RISCV_REQUIRE_REMOTE")]
    remote = runtime_config().remote
    return [remote.host, remote.repository, "1" if remote.required else "0"]
