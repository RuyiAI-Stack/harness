# dsh-agent-observer

[中文](README.zh.md) | English

Purpose: Observe the agent's tool execution during a session and, on session dispose, file the recorded report as a GitHub issue via `github-bot`.

Package: `@dangosys/dsh-agent-observer`

Dependencies: `githubBot` (from `@ruyiAi/dsh-osc-github-bot`), `tools`

config.yaml: `org` (github-bot org key), `repo` (owner/name), optional `title` (supports `{session}`), optional `enabled` (default `true`).

### Tools

1. agent_observer_report: Send the recorded session observations to the configured GitHub issue.

### Web APIs

none

### Service state

- `store`: in-memory `ObserverStore` of recorded tool calls for the current session.
