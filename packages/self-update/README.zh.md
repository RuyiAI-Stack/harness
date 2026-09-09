# dsh-self-update

[English](README.md) | 中文

功能：在后台执行仓库安装脚本（`tools/scripts/install-all.sh`），重建并重新注册 harness 及其插件——即 harness 的在线自进化入口。

包名：`@dangosys/dsh-self-update`

依赖：tools

config.yaml：可选 `root`（仓库根目录，默认取本包所在仓库根）、`script`（相对 `root` 的安装脚本，默认 `tools/scripts/install-all.sh`）、`logDir`（`root` 下的日志目录，默认 `.dsh`）、`enabled`（默认 `true`）。

### 可调用 Tools

1. self_update：后台启动自更新，运行 `tools/scripts/install-all.sh`；返回 pid 与日志路径。已有一个运行中的更新时拒绝再次启动。

2. self_update_status：报告后台更新运行中 / 成功 / 失败状态、退出码及日志尾部。

### 注册的 Web API

无

### 维护的 Service 状态

- `ctx.selfUpdate.runner`：保存当前运行状态与日志路径的 `InstallRunner`。
