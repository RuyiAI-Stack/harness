# dsh-workload-integration

[中文](README.zh.md) | English

Purpose: Given a HuggingFace model link, build an initial model-adaptation workload under Buckyball ModelTest e2e `models/models/<X>`.

Package: `@dangosys/dsh-workload-integration`

Dependencies: tools, systemPrompt, role, envBuilder

config.yaml: `hfToken` (required); optional `hfEndpoint`, `fetchTimeoutMs`

### Tools

1. buckyball_model_info: Fetch HuggingFace model metadata and file list.

2. buckyball_existing_models_index: List local ModelTest e2e models and registration points.

3. buckyball_workload_audit: Read-only completeness checks for a generated workload.

### Web APIs

none

### Service state

none (registers systemPrompt section `buckyball:workload`)
