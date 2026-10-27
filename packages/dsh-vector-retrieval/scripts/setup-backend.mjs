import { existsSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const root = fileURLToPath(new URL('..', import.meta.url))
const environment = join(root, '.venv')
const python = join(environment, 'bin/python')
function run(command, args) {
  const result = spawnSync(command, args, { stdio: 'inherit' })
  if (result.status !== 0) process.exit(result.status || 1)
}
if (!existsSync(python)) run(process.env.DSH_VECTOR_PYTHON || 'python3', ['-m', 'venv', environment])
run(python, ['-m', 'pip', 'install', join(root, 'python')])
