# dsh-agent-observer

[English](README.md) | 中文

功能：观察 agent 在会话中的工具执行，在会话结束时把记录下的报告通过 `github-bot` 创建为 GitHub issue。

包名：`@dangosys/dsh-agent-observer`

依赖：`githubBot`（来自 `@ruyiAi/dsh-osc-github-bot`）、`tools`

config.yaml：`org`（github-bot 的 org 键）、`repo`（owner/name）、可选 `title`（支持 `{session}` 占位）、可选 `enabled`（默认 `true`）。

### 可调用 Tools

1. agent_observer_report：把已记录的会话观察发送到配置的 GitHub issue。

### 注册的 Web API

无

### 维护的 Service 状态

- `store`：当前会话已记录工具调用的内存 `ObserverStore`。
