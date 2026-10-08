import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { resolveWorkbenchLaunch } from '../index.js'
import { loadLocalEnvironment } from './local-environment.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const env = loadLocalEnvironment(root)
const launch = resolveWorkbenchLaunch({
  enabled: true,
  repoRoot: env.TRITON_RISCV_REPO_ROOT || env.TRITON_RISCV_CHECKOUT,
  python: env.TRITON_RISCV_MCP_PYTHON,
  port: Number(env.TRITON_RISCV_WORKBENCH_PORT || 8765),
})
const child = spawn(launch.command, launch.args, {
  cwd: launch.cwd,
  env,
  stdio: 'inherit',
})
child.on('error', error => {
  console.error(error.message)
  process.exitCode = 1
})
child.on('exit', code => {
  process.exitCode = code ?? 1
})
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal))
