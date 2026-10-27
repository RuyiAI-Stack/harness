#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Triton-RISCV plugin requires Python >=3.10; set PYTHON to that executable"'
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PYTHON" -m venv "$ROOT/.venv"
fi
SOURCES=()
for name in dsh-vector-retrieval dsh-multi-agent; do
  if [ -f "$ROOT/../$name/python/pyproject.toml" ]; then
    SOURCES+=("$ROOT/../$name/python")
  fi
done
# Install local plugin distributions together so pip need not find unpublished
# repository packages on PyPI. Outside this checkout, install their wheels first.
"$ROOT/.venv/bin/python" -m pip install "${SOURCES[@]}" "$ROOT/python[workbench]"
"$ROOT/.venv/bin/python" -I -c 'import codex_agent; print("Installed backend:", codex_agent.__file__)'
