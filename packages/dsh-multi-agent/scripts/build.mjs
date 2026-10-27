import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
for (const path of ['index.js', 'lib/service.js', 'python/dsh_multi_agent/role-policy.mjs'])
  execFileSync(process.execPath, ['--check', path], { cwd: root, stdio: 'inherit' })
