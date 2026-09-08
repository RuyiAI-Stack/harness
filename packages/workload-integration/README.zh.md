# dsh-workload-integration

[English](README.md) | 中文

功能：给定 HuggingFace 模型链接，在 Buckyball ModelTest e2e `models/models/<X>` 下搭建初始模型适配 workload。

包名：`@dangosys/dsh-workload-integration`

依赖：tools, systemPrompt, role, envBuilder

依赖的config.yaml配置：`hfToken`（必填）；可选 `hfEndpoint`、`fetchTimeoutMs`

### 可调用Tools

1. buckyball_model_info：拉取 HuggingFace 模型 metadata 与文件列表。

2. buckyball_existing_models_index：列出本地已有 ModelTest e2e 模型与注册点。

3. buckyball_workload_audit：只读检查某个已生成 workload 的完整度。

### 注册的 Web API：

无

### 维护的 Service 状态

无

### 依赖的 Skills

@DangoSys/workload
