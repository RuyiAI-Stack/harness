# Triton-RISCV Python Backend

The `codex_agent` package provides the plugin's MCP tools, operator lifecycle, remote execution, RAG, and optional FastAPI workbench. Harness runs the native agent loop.

For installation and usage, start with the [plugin README](../../README.md). The backend is installed once in the plugin environment, not in each operator checkout.

## Local Development

From `packages/dsh-triton-riscv/` after plugin setup:

```sh
.venv/bin/python -m pip install -e './python[workbench]'
.venv/bin/python -m codex_agent.storage upgrade
npm run test:backend
```

To inspect a standalone command, run `.venv/bin/python -I -m codex_agent.operator_agent --help`. Use `-I` with installed CLI tools to avoid importing an older backend from the target checkout.

Generated operators and validation artifacts belong to the target checkout. Credentials, databases, and execution logs are not package source files.

MySQL 8.4 and `TRITON_MYSQL_URL` are required; use disposable services for tests
as shown in the [plugin README](../../README.md#development-tests). Vitest runs each
Python test module with pytest, including both `unittest.TestCase` and function
tests. It reports Python failures and skips; no separate Python CI job is needed.
`storage.urlEnv` in the plugin config can name a different credential variable.
Schema changes are explicit; startup never migrates data or falls back to SQLite.

## Existing Data

Stop writers and back up the old databases first. After the schema upgrade,
import into a workspace that has not yet been opened in MySQL:

```sh
.venv/bin/python -m codex_agent.storage.import_sqlite \
  --workspace /absolute/path/to/triton-riscv \
  --platform /path/to/platform.sqlite3 \
  --memory /path/to/memory.sqlite3
```

Omit absent sources. Import each legacy reference catalog separately with
`--workspace /absolute/path/to/library --references /path/to/catalog.sqlite3`.
All supplied sources commit together after field/count checks; existing target
workspaces are rejected. IDs, labels, source paths and message order are preserved.
Older chunk layouts are rebuilt and their stale vectors cleared; use
`codex_agent.memory --workspace PATH embed-missing --help` to re-index if needed.

Evidence files stay in place. Retain them and the original databases for rollback;
the importer never reclassifies historical results as verified successes.
`MemoryStore` and `PlatformStore` now take a workspace directory, not a database
file. The memory CLI uses `--workspace`; development uses `--memory-workspace`.

## Redis Options

Configure `cache` alongside `storage` in the plugin config. CLI/workbench processes
can use the same versioned `TRITON_RISCV_CONFIG` JSON document; the URL remains in
`TRITON_REDIS_URL` (or the name in `cache.urlEnv`). Restart workers after changing it.

Defaults: `ttlSeconds: 60` and `negativeTtlSeconds: 20` (both jittered by 20%),
`lockMs: 10000`, `waitMs: 1000`, `maxBytes: 262144`, `maxConcurrent: 4`,
`requestsPerSecond: 100`, `rebuildsPerSecond: 10`. Rates are per workspace/domain
per second, with a database-wide cap of four times each rate. Use identical limits
across workers. Redis needs script commands and read-only `CONFIG GET` permission.

Redis loss opens a two-second circuit breaker. Fallback is limited to four attempts
per second and two callers per process; MySQL connection-owned slots also enforce
the configured cross-process rebuild concurrency. Cache locks never authorize writes.

Bloom is optional (`bloom: true`, `bloomCapacity: 10000`, `bloomErrorRate: 0.001`)
and requires `BF.*` support. With the same runtime config, run:

```sh
.venv/bin/python -I -m codex_agent.memory --workspace /path/to/checkout warm-cache-ids
.venv/bin/python -I -m codex_agent.memory --workspace /path/to/checkout show 1
```

The filter only applies to exact case IDs, never natural-language retrieval.
It expires after ten minutes and is bypassed after data changes until rebuilt.
This version supports standalone Redis, not Redis Cluster. `allkeys-lfu/lru` are
rejected because the instance also holds locks. Keep services private; use ACLs/TLS
for remote connections. Do not give database or cache credentials to generated tests.

For integration tests, configure all three disposable services from the plugin
README, upgrade the schema, then run `npm run test:backend`.
`TRITON_TEST_REDIS_SERVER` enables the real stop/restart test. These tests change
Redis memory settings; never target a shared service. Cache metrics are available
through `store.cache.backend.snapshot()` and are process-local, not a monitoring service.

## RabbitMQ Workers

Queue mode is opt-in. Install the updated backend and run `codex_agent.storage upgrade`
against a backed-up MySQL database. Set `TRITON_AMQP_URL` to a private RabbitMQ
connection URL. Use the same versioned config, credentials and checkout in every process:

```sh
export TRITON_RISCV_CONFIG='{"schemaVersion":1,"repoRoot":"/path/to/checkout","stateDir":"/path/to/checkout/agent-results","permissions":{"validation":true},"queue":{"enabled":true}}'
.venv/bin/python -I -m codex_agent.storage upgrade
.venv/bin/python -I -m codex_agent.platform --host 127.0.0.1 --port 8765
```

In separate terminals (or a process supervisor), start each role:

```sh
.venv/bin/python -I -m codex_agent.workers relay --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.workers recover --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.workers agent --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.workers validation --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.workers memory --workspace /path/to/checkout
```

The workbench persists messages/jobs/outbox together and returns HTTP 202. Clients
must reuse `request_id` on retries. Read `/api/jobs/{id}` for attempts, delivery
status and artifact links; `POST /api/jobs/{id}/cancel` requests cancellation.
`POST /api/jobs/validation` accepts `operator`, `approved_run_id`, `request_id`.
It never grants approval. Native `execute_approved_validation` queues work when enabled;
`inspect_queued_task` reads its result on a later turn. Batch/project tools remain synchronous.

Defaults: 60-second worker leases, 3 safe attempts, 1,200-second job deadline,
100 outstanding user jobs per workspace, one delivery per worker. Mutating jobs
are serialized per workspace. Memory ingestion follows audited validation, not broker ACK.
Broker confirms mean "delivered to broker", not "tests passed". Delivery is at least once.

An interrupted effectful task becomes `needs-reconciliation`, never an automatic
new model/SSH run. After inspecting and stopping any remaining processes, an operator
can close it with `workers reconcile --workspace PATH --job ID --note "checks performed"
--confirm-no-active-execution`. This records an attestation, not a verified success;
existing lifecycle journals still guard later execution.

This release uses separate local processes, durable classic queues and local
content-addressed artifacts. API/workers must share the same host, absolute checkout
and state path. It is not a multi-host HA deployment: shared object storage, quorum
queues, authentication, monitoring and retention policies remain deployment work.
Redis is never the task source of truth. Keep all services private.

With the same disposable service configuration, run just the queue module:
`npm run test:backend -- -t test_task_queue.py`.
The optional private-broker restart test also needs `TRITON_TEST_RABBITMQ_CTL`.
