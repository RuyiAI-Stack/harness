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

### Case and Chunk Versions

Schema upgrade `0003_memory_lineage` adds nullable lineage fields without rewriting
historical content. New writes keep source snapshots in `memory_versions` and bind
each chunk to its workspace, case version, source unit and part number. These are
content versions, not Git revisions or proof that a repair passed.

To backfill existing cases, stop writers, back up MySQL, then run:

```sh
.venv/bin/python -I -m codex_agent.storage upgrade
.venv/bin/python -I -m codex_agent.memory --workspace /path/to/checkout rebuild-chunks --embedding-provider none
```

The command above makes no model calls and uses character-bounded chunks; unchanged chunk
text keeps its vectors, changed layouts clear child vectors for explicit re-indexing.
Old cases (including SQLite imports) remain readable with unknown lineage until
rebuilt. Snapshots start at the first write/rebuild; earlier overwritten versions
cannot be recovered. Retrieval defaults and prompt assembly are unchanged.

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

### Experimental Operator Workflows

`queue.proposalWorkflows` defaults to `false` and requires `queue.enabled`.
Enabling it permits model calls with frozen source, tests and task evidence.
Normal chat routing is unchanged; roles propose files but cannot use tools.

Select **Multi-agent** in the workbench, enter an operator name and requirements,
and review the extracted specification and patch before approving. Repairs require
an audited failure receipt ID. Nothing is written or tested merely by submitting.
API clients use `POST /api/operator-workflows` with `request`, `request_id` and either
a structured `specification`, or `request.auto_publish: true` for interactive creation.
Repair submissions require `failure_run_id`; repeated requests must reuse `request_id`.
The Agent Worker plans tests, proposes code and independently reviews it.
Read `GET /api/operator-workflows/{id}` for state, role steps and artifact IDs.

Host actions use `POST /api/operator-workflows/{id}/{action}` with
`{"expected_revision": N}`. The sequence is `publish`, human approval through the
existing proposal decision endpoint, `apply`, `plan-validation`, human approval
through the existing validation decision endpoint, then `enqueue-validation`.
No action accepts a caller-selected proposal or result ID. Write/validation
permissions remain separate; repairing an operator cannot replace its tests.

The Validation Worker collects the result automatically. If collection fails,
use `collect-validation`; it checks the saved evidence without rerunning tests.
Only matching receipt, log, source, plan and worker artifact can mark success.
Interrupted writes require reconciliation, not blind retries. Review rework and
optional `request.auto_repair` are bounded by `request.max_repairs` (default 3).
Failed attempts stay failed and link to separate repair tasks. Source-irreparable
failures stop; all new writes/tests still require approval. Roles receive bounded
stage-specific RAG evidence from the configured retriever; missing RAG is recorded,
not reported as verified knowledge. Each role's input, retrieval trace and output
are saved as artifacts. This experimental route has not been live-model calibrated.

## Async Vector Indexing

Indexing is off by default and independent of `queue.enabled`. Configure `indexing`
in the same plugin/runtime document, with `enabled`, `generation`, `dimensions`,
`modelRevision` and a separate `embedding` definition. For example, an installed
local model can use `embedding: {provider: "ollama", model: "YOUR_MODEL", tokenBudget: 180}`.
Use its actual dimension and an immutable model revision, not this placeholder.
Hosted embeddings additionally require the matching `tokenizerJson` and `tokenBudget`.
Install `[embedding-client]` in the backend environment for endpoint token counting;
the model server's tokenizer installation is not shared with MCP or workers.

Run Qdrant privately (`qdrantUrl` defaults to `http://127.0.0.1:6333`); its optional
key comes from `TRITON_QDRANT_API_KEY`. Run the schema upgrade before enabling indexing.
Start `relay`, `recover` and the new `index` worker in separate terminals:

```sh
.venv/bin/python -I -m codex_agent.indexing prepare --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.workers index --workspace /path/to/checkout
```

New/changed memories commit an indexing job and Outbox event with the source data.
Unchanged writes do not enqueue again. For existing data and failed jobs:

```sh
.venv/bin/python -I -m codex_agent.indexing backfill --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.indexing status --workspace /path/to/checkout
.venv/bin/python -I -m codex_agent.indexing backfill --workspace /path/to/checkout --retry-failed
```

For large histories, enqueue at most 100 sources per call instead:

```sh
.venv/bin/python -I -m codex_agent.indexing backfill-batch --workspace /path/to/checkout --batch-size 25
# Resume with the returned next_cursor JSON; repeat until done is true.
.venv/bin/python -I -m codex_agent.indexing backfill-batch --workspace /path/to/checkout --batch-size 25 --cursor 'NEXT_CURSOR_JSON'
```

Batch cursors target the configured generation and freeze ID upper bounds, not
source contents. Each page uses current versions; new writes still index incrementally.
Replaying a page is safe. `done` means enqueue finished, not vectors ready; check
`status` before activation. Keep the same workspace, generation and retry policy.

Model, tokenizer or chunk-policy changes require a new `generation`; old collections
and source snapshots remain. `activate --generation NAME` checks every current active
case before changing the index ledger's active pointer. **It does not enable vector
retrieval**: existing ranking, context assembly and retrieval defaults stay unchanged.
After activating/configuring a replacement, `retire --generation OLD` stops updates
to the old generation without deleting its collection. Staging generations receive
updates until retired. A model revision label must identify a pinned deployment;
the label alone cannot prove that a remote provider has not changed its model.
Archive schedules vector removal. After direct SQL deletions, run `backfill`; run
`cleanup --generation NAME` to remove obsolete points after interrupted stale writes.
Neither command deletes source snapshots or indexed text. Check readiness in MySQL,
not only Qdrant point counts; physical deletion is eventually consistent.

Tests: set `TRITON_TEST_QDRANT_URL` to a disposable Qdrant server in addition to the
three test services, then run `npm run test:backend -- -t test_memory_indexing.py`.
Vectors in these tests are synthetic; passing tests do not establish retrieval quality.

### Vector Retrieval (Opt-in)

After backfill and activation, set `memory.retrievalMode: qdrant` in the plugin
configuration. Default `legacy` is unchanged. Queries use the active index's
embedding model, not `memory.embedding`. Local Sentence Transformers load
`modelRevision`; remote deployments must independently guarantee model stability.
Install the backend's optional `[embedding]` dependencies for local inference.

Preview a staging generation without changing the active index or product mode:

```sh
.venv/bin/python -I -m codex_agent.indexing query --workspace /path/to/checkout \
  --generation NAME --query 'Which historical cases explain this lowering error?' --limit 3
```

Qdrant returns candidate IDs, not all vectors. MySQL checks workspace, current
case version, ready state and chunk checksum before returning evidence. Top-K is
distinct cases (1-50), recalled from at most 200 chunks. The trace reports rejected
or excluded hits and candidate-budget exhaustion; fewer results are not proof that
no relevant case exists. Queries over 2,048 model tokens are rejected, not truncated.
Missing/unready indexes and service errors do not silently fall back to legacy.
Vector reads bypass Redis until index-version-aware caching is implemented.

### Three-Layer Evidence Context

With `qdrant` selected, MCP and native automatic retrieval expand each matched case:

1. Recorded semantics, reference and environment, bound to the current case version.
2. Stage-specific evidence: contracts/results for generation; errors for diagnosis;
   actions, patches, results and separately labelled recommendations for repair.
3. At most one preceding and following part of the same source unit. Missing lineage
   is not guessed; a neighbor is a labelled fragment, not a reconstructed whole file.

The host selects `task_stage` from lifecycle tools. MCP callers can explicitly use
`lookup`, `generation`, `diagnosis`, `repair` or `validation`; this never changes ranking
or grants execution permission. CLI preview also accepts `--task-stage repair`.

Expanded evidence uses source-bound JSON within the existing 6,000 Unicode-character
budget, including metadata and omission markers. This is **not a model-token budget**;
Harness still owns the complete conversation window. Whole selected snippets are kept
or omitted, and recommendations never become verified repairs. Legacy mode/format is
unchanged. Activation and switching the workbench remain explicit operator choices.
