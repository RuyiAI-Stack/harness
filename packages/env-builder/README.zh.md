# dsh-env-builder

[English](README.md) | 中文

功能：管理 ephemeral environment 目录与 GitHub component 安装，供 Singularity / workload 使用。

包名：`@dangosys/dsh-env-builder`

依赖：tools, webServer

依赖的config.yaml配置：无

### 可调用Tools

1. env_list：列出 environment/projectN 及其组件。

2. env_register_component：bash 安装完成后，将已克隆并 build 的仓库登记到 manifest。

3. env_remove_component：从环境移除组件。

4. env_set_component_status：设置组件状态 ready | modified | installing。

5. env_select：选择当前活动环境。

6. env_select_repo：在已选环境中选择活动仓库。

7. env_ensure_component：确认已选仓库组件已克隆，返回本地路径。

8. env_bind_component_session：绑定 session 到组件。

9. env_attach_session / env_detach_session：环境 session 树挂载 / 卸载。

10. env_rename：重命名环境 id。

11. env_set_running：设置 Live 运行态。

12. env_reset：对所有组件 git reset --hard && git clean -fd。

13. env_delete：删除环境目录与 manifest 条目。

Singularity 模式下的环境创建由 `@dangosys/dsh-singularity-agent` 的 `env_create` 负责。

### 注册的 Web API：

前缀：`/integrations/env-builder`

1. GET `/environments`：列出环境。

2. GET `/selected`：当前选中环境与仓库。

3. POST `/selected-repo`：选择仓库（body: `{ repo }`）。

4. DELETE `/environments/:id`：删除环境。

5. POST `/environments/:id/select|reset|running|rename`：选中 / 重置 / 运行态 / 重命名。

6. DELETE `/environments/:id/components`：删除组件（安装由 agent bash + env_register_component 完成）。

7. POST `/environments/:id/component-status|component-session`：组件状态 / session 绑定。

8. POST|DELETE `/environments/:id/sessions`：挂载 / 卸载 session。

### 维护的 Service 状态

1. ctx.envBuilder.store：`$RUYI_ROOT/environment` 下的 manifest 与 project 目录
