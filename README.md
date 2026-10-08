# Harness

This repository is built on [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) and developed as plugins. All plugins live under `packages/`.

### Required environment

Please install these before running the build script.

```
node >= 22
pnpm >= 11
python3 >= 3.8
```

### Quick Build

Clone the repo, copy `config.yml.example` to `config.yml` and fill in the required keys, then run `install-all.sh`. That script installs dependencies, builds deepseek-harness and registers our designed dsh plugins on the `web` profile.

```sh
git clone https://github.com/DangoSys/harness.git
cd harness/
cp config.yml.example config.yml
# edit config.yml

./tools/scripts/install-all.sh
```

### Quick Activate

After a successful build, start the web profile through the launcher. `DEEPSEEK_API_KEY` is required.

```sh
cd harness/
DEEPSEEK_API_KEY=sk-... ./dsh web
```

## Triton-RISCV Agent

[`dsh-triton-riscv`](packages/dsh-triton-riscv/README.md) adds guarded
Triton-RISCV operator discovery, development, validation, diagnosis, repair,
memory retrieval, and remote RISC-V execution to the Harness web profile.

The install-all script also installs this package's Python backend, which requires
Python >=3.10. If `python3` is older, run
`PYTHON=/absolute/path/to/python3.10 ./tools/scripts/install-all.sh`.
Configure `triton-riscv-native-host` in `config.yml` to enable it; installation
alone does not enable operator execution or write permissions.
