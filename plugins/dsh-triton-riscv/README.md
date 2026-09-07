# dsh-triton-riscv

An installable DeepSeek Harness bundle for the guarded Triton-RISCV operator
development and validation tools.

## What The Plugin Adds

The bundle adds two Cordis rows to an existing Harness profile:

1. A domain policy plugin contributes the Triton-RISCV lifecycle rules to the
   assembled system prompt.
2. The official Harness MCP client starts
   `python -m codex_agent.harness.mcp_server` and exposes its typed tools as
   `mcp__triton_riscv__*`.
3. A Harness Web client plugin occupies the root UI slot and displays the
   existing Triton-RISCV React/FastAPI workbench.

The model and Agent loop remain owned by Harness. Discovery, contract checking,
source proposals, validation, diagnosis, bounded repair, remote SSH execution,
and trusted validation evidence remain owned by the Python domain engine in a
Triton-RISCV checkout.

```text
Harness profile
  -> dsh-triton-riscv bundle
     -> system-prompt policy
     -> official dsh-mcp-client
        -> Python Triton-RISCV MCP server
           -> operator lifecycle and RISC-V remote executor
```

## Prerequisites

- DeepSeek Harness with the `web` profile and Node.js 22 or newer
- a Triton-RISCV checkout containing `codex_agent`
- a Python environment with `codex_agent/harness/requirements.txt` installed
- optional SSH configuration and a prepared RISC-V host for live validation

The bundle fails at Harness startup when the Python MCP module cannot be
imported instead of silently starting without the domain tools.

## Install From A Harness Checkout

From the Harness repository root, point the plugin at a separate Triton-RISCV
checkout:

```sh
export TRITON_RISCV_CHECKOUT=/absolute/path/to/triton-riscv
export TRITON_RISCV_MCP_PYTHON="$TRITON_RISCV_CHECKOUT/.harness-venv/bin/python"
export TRITON_RISCV_WORKBENCH_PORT=8765
dsh plugin --profile web add ./plugins/dsh-triton-riscv
dsh --profile web --dump-config
dsh web
```

The config dump should contain both `triton-riscv-domain-policy` and
`mcp-triton-riscv`. Starting `dsh web` also starts the Python workbench and the
Harness page displays that workbench in its root slot. Add `?nativeHarness=1`
to the Harness URL when the stock Harness interface is needed.

The embedded client defaults to `http://127.0.0.1:8765`. For a different port
or externally hosted workbench, open the Harness URL with an encoded
`?tritonWorkbenchUrl=https://host/path` query parameter. Set
`TRITON_RISCV_WORKBENCH_AUTOSTART=0` when that external service is already
managed by another process.

For read-only discovery, no RISC-V server is required. Try:

```text
Call the Triton-RISCV discovery tool for relu_and_mul and summarize its
implementation, tests, and validation command. Do not run validation.
```

## Enable Remote Validation Deliberately

All expensive or mutating operations remain disabled by default. Configure the
host before starting Harness:

```sh
export RISCV_HOST=sg2044
export RISCV_REPO=/home/lichunbo/work/triton-riscv
export TRITON_RISCV_REQUIRE_REMOTE=1
export TRITON_RISCV_ALLOW_VALIDATION=1
export TRITON_RISCV_REQUIRE_APPROVED_VALIDATION=1
```

Enable source application only on a trusted development checkout:

```sh
export TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY=1
export TRITON_RISCV_ALLOW_REPAIR_APPLY=1
```

The model cannot select the SSH host, repository path, arbitrary command, or
files to synchronize. It also cannot approve its own proposal or validation
plan.

## Test The Bundle

The package has no install-time dependencies:

```sh
cd plugins/dsh-triton-riscv
npm test
npm pack --dry-run
```

The tests check the bundle manifest, Cordis rows, safe default switches, and
system-prompt lifecycle registration without starting a model or remote host.

## Standalone Triton-RISCV Demo Platform

The React and FastAPI workbench remains in the Triton-RISCV checkout and can
load the same plugin policy and MCP server:

```sh
cd "$TRITON_RISCV_CHECKOUT"
.harness-venv/bin/python -m codex_agent.platform --port 8765
```

Open `http://127.0.0.1:8765`. This standalone command remains useful while
developing the workbench without starting a complete Harness profile.

## Remove

```sh
dsh plugin --profile web remove dsh-triton-riscv
```

## Current Boundary

This bundle packages the model-facing policy, MCP connection, and Harness Web
entry. The Python domain engine and compiled React assets stay in the target
Triton-RISCV checkout. A later release can publish that engine and its assets
independently so the bundle no longer depends on a source checkout.
