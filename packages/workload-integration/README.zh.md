# dsh-workload-integration

[English](README.md) | 中文

DeepSeek Harness 插件：给定 HuggingFace 模型链接，在 Buckyball ModelTest e2e `models/models/<X>` 下搭建初始模型适配 workload。

依赖已选中的 env-builder 仓库。使用触及仓库的工具前，先调用 `env_ensure_component({ name: "buckyball" })`。
