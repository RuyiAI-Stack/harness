# Triton-RISCV Agent

A Harness plugin for operator development, testing, repair, and historical evidence retrieval.

### Required environment

```text
node >= 22.19 (with npm)
pnpm >= 11
python3 >= 3.10
```

An existing Triton-RISCV checkout is required as the workspace.

### Quick Build

From the Harness repository root, create the config if absent:

```sh
cp -n config.yml.example config.yml
```

Follow the [host configuration](../../README.md). Set `config.enabled: true` in the `triton-riscv-native-host` entry, then build:

```sh
./tools/scripts/install-all.sh
```

### Quick Activate

Set `DEEPSEEK_API_KEY` in your terminal environment, then run from the repository root:

```sh
./dsh web
```

Open the printed URL and select your Triton-RISCV checkout as the conversation workspace. Try:

```text
Find relu_and_mul and show its implementation and tests. Do not execute tests.
```

### Run Tests or Apply Changes

Enable the needed `permissions` in the plugin config: `validation`, `development`, or `repair`. Actions still require approval.

```text
Validate relu_and_mul. Show the plan and run it after my approval.
```

For remote tests, set `remote.host` and `remote.repository`; the server needs SSH access and the Triton/Buddy/LLVM toolchain.

### Optional Workbench

After installation, run from the repository root:

```sh
cd packages/dsh-triton-riscv
export TRITON_RISCV_REPO_ROOT=/absolute/path/to/triton-riscv
npm run workbench
```

Open **http://127.0.0.1:8765**. FastAPI serves the built frontend; no separate frontend server is needed. The native sidebar opens this page but does not start the service.

Live requests need separate `ISRC_API_KEY` and `DSH_MODEL` settings. Workbench and native sessions are independent.

### Development Tests

From the plugin directory:

```sh
npm test
.venv/bin/python -m unittest discover -s python/codex_agent/tests -v
```

For trusted local use only. Never commit credentials. Backend development commands are in [python/codex_agent/README.md](python/codex_agent/README.md).
