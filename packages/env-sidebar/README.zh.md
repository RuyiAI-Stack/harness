# dsh-env-sidebar

[English](README.md) | 中文

功能：官方右侧栏 Environments（环境）页面 tab。新建/安装只向当前会话投递意图（人类不编排）。标签与引导页双语，带图标。

包名：`@dangosys/dsh-env-sidebar`

依赖：无（服务端空 apply）；客户端 ui-primitives, ui-sidebar-right, client-locale；HTTP 读 env-builder；经 sessions 向 agent 提问

依赖的config.yaml配置：无

### 可调用Tools

无

### 注册的 Web API：

无（列表/选中仍读 `/integrations/env-builder/*`；创建/安装走 agent）

### 维护的 Service 状态

无
