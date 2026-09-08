# dsh-workload-integration

[English](README.md) | 中文

DeepSeek Harness 插件：给定 HuggingFace 模型链接，在 Buckyball ModelTest e2e `models/models/<X>` 下搭建初始模型适配 workload。

依赖 `@dangosys/dsh-open-source-collaboration`。使用触及仓库的工具前，先调用 `ensure_repo({ name: "buckyball" })`，仓库落在 `workspace/buckyball`。
