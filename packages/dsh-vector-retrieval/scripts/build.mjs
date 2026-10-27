import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

for (const file of ['index.js', 'lib/config.js', 'lib/service.js', 'scripts/setup-backend.mjs']) {
  const result = spawnSync(process.execPath, ['--check', fileURLToPath(new URL('../' + file, import.meta.url))], { stdio: 'inherit' })
  if (result.status !== 0) process.exit(result.status || 1)
}
