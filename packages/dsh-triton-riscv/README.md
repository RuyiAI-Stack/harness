# Triton-RISCV Agent

A Harness plugin for operator development, testing, repair, and historical evidence retrieval.

### Required environment

```text
node >= 22.19 (with npm)
pnpm >= 11
python3 >= 3.10
MySQL 8.4
```

An existing Triton-RISCV checkout is required as the workspace.

### Quick Build

From the Harness repository root, create the config if absent:

```sh
cp -n config.yml.example config.yml
```

Follow the [host configuration](../../README.md). Set `config.enabled: true` in the `triton-riscv-native-host` entry, then build:

```sh
./tools/scripts/install-all.sh
```

Create an empty MySQL database named `triton_agent`. Export its connection URL
in the terminal that starts Harness (URL-encode special characters in passwords):

```sh
export TRITON_MYSQL_URL='mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/triton_agent'
packages/dsh-triton-riscv/.venv/bin/python -m codex_agent.storage upgrade
```

Keep credentials outside Git. Each selected workspace has isolated records.
For existing SQLite data, [import it before first launch](python/codex_agent/README.md#existing-data).

### Quick Activate

Set `DEEPSEEK_API_KEY` in your terminal environment, then run from the repository root:

```sh
./dsh web
```

Open the printed URL and select your Triton-RISCV checkout as the conversation workspace. Try:

```text
Find relu_and_mul and show its implementation and tests. Do not execute tests.
```

### Run Tests or Apply Changes

Enable the needed `permissions` in the plugin config: `validation`, `development`, or `repair`. Actions still require approval.

```text
Validate relu_and_mul. Show the plan and run it after my approval.
```

For remote tests, set `remote.host` and `remote.repository`; the server needs SSH access and the Triton/Buddy/LLVM toolchain.

### Optional Workbench

After installation, run from the repository root:

```sh
cd packages/dsh-triton-riscv
export TRITON_RISCV_REPO_ROOT=/absolute/path/to/triton-riscv
npm run workbench
```

Open **http://127.0.0.1:8765**. FastAPI serves the built frontend; no separate frontend server is needed. The native sidebar opens this page but does not start the service.

Live requests need separate `ISRC_API_KEY` and `DSH_MODEL` settings. Workbench and native sessions are independent.

### Optional Redis Cache

Use a private standalone Redis with a memory limit and `maxmemory-policy noeviction`.
Export `TRITON_REDIS_URL` in the Harness terminal; set `cache.enabled: true` in the
plugin config. Credentials stay outside Git. Redis is off by default.

```sh
export TRITON_REDIS_URL='redis://127.0.0.1:6379/0'
```

Historical retrieval, case details and statistics are cached; approvals and run
state are not. Writes invalidate versioned results. Cache errors use bounded
MySQL fallback, then report busy rather than returning false "not found" results.
All workers must use the same limits and MySQL endpoint. This is not multi-user authentication.
See [backend options](python/codex_agent/README.md#redis-options).

### Development Tests

Vitest drives JavaScript, Python and native-host integration tests. After setup,
from the plugin directory, configure **disposable** services (never production):

```sh
export TRITON_MYSQL_URL='mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/triton_agent_test'
export TRITON_TEST_REDIS_URL='redis://127.0.0.1:6379/0'
export TRITON_TEST_AMQP_URL='amqp://guest:guest@127.0.0.1:5672/'
.venv/bin/python -I -m codex_agent.storage upgrade
npm test
```

Use `npm run test:unit` for JavaScript-only checks or `npm run test:backend` for
Python tests through Vitest. Python failures, collection errors and timeouts fail
the Vitest run. Model API keys and a RISC-V server are not required. CI provisions
its own service containers; private-service restart tests need the optional
controls described in the backend README and otherwise report skips.

Before pushing, run from the repository root:

```sh
python3 -m pip install pre-commit
pre-commit run --all-files
pnpm test:all
```

For trusted local use only. Never commit credentials. Backend development commands are in [python/codex_agent/README.md](python/codex_agent/README.md).
