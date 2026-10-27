#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DSH_MULTI_AGENT_VENV:-$ROOT/.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  "${PYTHON:-python3}" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install "$ROOT/python"
