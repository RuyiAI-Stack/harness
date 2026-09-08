# dsh-workload-integration

[中文](README.zh.md) | English

A DeepSeek Harness plugin: given a HuggingFace model link, build an initial model-adaptation workload under Buckyball's ModelTest e2e `models/models/<X>`.

Requires a selected env-builder repository. Call `env_ensure_component({ name: "buckyball" })` before tools that touch the checkout.
