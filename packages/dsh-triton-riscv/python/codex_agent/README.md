# Triton-RISCV Python Backend

The `codex_agent` package provides the plugin's MCP tools, operator lifecycle, remote execution, RAG, and optional FastAPI workbench. Harness runs the native agent loop.

For installation and usage, start with the [plugin README](../../README.md). The backend is installed once in the plugin environment, not in each operator checkout.

## Local Development

From `packages/dsh-triton-riscv/` after plugin setup:

```sh
.venv/bin/python -m pip install -e './python[workbench]'
.venv/bin/python -m unittest discover -s python/codex_agent/tests -v
```

To inspect a standalone command, run `.venv/bin/python -I -m codex_agent.operator_agent --help`. Use `-I` with installed CLI tools to avoid importing an older backend from the target checkout.

Generated operators and validation artifacts belong to the target checkout. Credentials, databases, and execution logs are not package source files.

To copy historical memory, inspect the options with `.venv/bin/python -I -m codex_agent.migrate_memory --help`. Keep the original data until the copied evidence has been checked.
