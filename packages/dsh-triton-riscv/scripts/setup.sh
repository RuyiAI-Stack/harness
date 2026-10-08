#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
node "$ROOT/scripts/build.mjs"
bash "$ROOT/scripts/setup-backend.sh"
