# dsh-self-update

[中文](README.zh.md) | English

Purpose: Run the repository install script (`tools/scripts/install-all.sh`) in the background so the agent can rebuild and re-register the harness and its plugins — the harness's online self-evolution entry point.

Package: `@dangosys/dsh-self-update`

Dependencies: `tools`

config.yaml: optional `root` (repo root, defaults to the repo that contains this package), `script` (install script relative to `root`, default `tools/scripts/install-all.sh`), `logDir` (log directory under `root`, default `.dsh`), `enabled` (default `true`).

### Tools

1. self_update: Start a background self-update by running `tools/scripts/install-all.sh`; returns the pid and log path. Refuses to start a second run while one is active.

2. self_update_status: Report whether the background run is running / finished / failed, its exit code, and the tail of the log.

### Web APIs

none

### Service state

- `ctx.selfUpdate.runner`: the `InstallRunner` holding the current run state and log path.
