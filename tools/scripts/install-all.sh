#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DSH="$ROOT/thirdparty/deepseek-harness"

export DSH_HOME="$ROOT/.dsh" DSH_PROFILE=web CI=true

mkdir -p "$DSH_HOME" "$DSH_HOME/profiles/web"

# ===------------------------------------------------------===#
# Step 1. Install harness dependencies
# ===------------------------------------------------------===#
cd "$ROOT"
pnpm install --frozen-lockfile

# ===------------------------------------------------------===#
# Step 2. Install deepseek-harness and build it
# ===------------------------------------------------------===#
git submodule update --init
cd "$DSH"
pnpm install --frozen-lockfile
pnpm run build

# ===------------------------------------------------------===#
# Step 3. Build local packages
# ===------------------------------------------------------===#
for pkg in "$ROOT"/packages/*/; do
  if [ -f "$pkg/package.json" ]; then
    (cd "$pkg" && pnpm install --frozen-lockfile && pnpm build)
  fi
done

# ===------------------------------------------------------===#
# Step 4. Install local plugins into profile
# ===------------------------------------------------------===#
pnpm dsh plugin --profile web add "$ROOT"/packages/*/
pnpm dsh plugin --profile web add "$ROOT/packages/collaboration/bundle"
pnpm dsh plugin --profile web add "$ROOT/packages/singularity/bundle"

# ===------------------------------------------------------===#
# Step 5. Install third-party plugins from npm
# ===------------------------------------------------------===#
PROFILE="$ROOT/.dsh/profiles/web"
set +e
pnpm dsh plugin --profile web add dsh-better-sidebar@latest
set -e
(cd "$PROFILE" && pnpm approve-builds --all)
pnpm dsh plugin --profile web add dsh-better-sidebar@latest

# ===------------------------------------------------------===#
# Step 6. Apply profile patch after all plugin adds
# ===------------------------------------------------------===#
cp "$ROOT/config.yml" "$DSH_HOME/profiles/web/cordis.patch.yml"
