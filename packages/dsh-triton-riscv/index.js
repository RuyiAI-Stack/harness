import { spawn } from 'node:child_process'
import { dirname, isAbsolute, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PROMPT_SECTIONS } from './lib/prompts.js'
export { TRITON_RISCV_SYSTEM_PROMPT } from './lib/prompts.js'

export const name = 'triton-riscv-domain-policy'

export const inject = ['systemPrompt']

export function resolveWorkbenchLaunch(config = {}) {
  if (config.enabled === undefined || config.enabled === false) return null
  if (config.enabled !== true) throw new Error('workbench.enabled must be a boolean')

  const checkout = (config.repoRoot ?? '').trim()
  if (!checkout) {
    throw new Error('workbench.repoRoot is required to start the Triton-RISCV workbench')
  }
  if (!isAbsolute(checkout)) {
    throw new Error('workbench.repoRoot must be an absolute path')
  }

  const portText = String(config.port ?? 8765).trim()
  if (!/^\d+$/.test(portText)) {
    throw new Error('TRITON_RISCV_WORKBENCH_PORT must be an integer')
  }
  const port = Number(portText)
  if (port < 1 || port > 65535) {
    throw new Error('TRITON_RISCV_WORKBENCH_PORT must be between 1 and 65535')
  }

  return {
    command: config.python ?? join(dirname(fileURLToPath(import.meta.url)), '.venv', 'bin', 'python'),
    args: ['-I', '-m', 'codex_agent.platform', '--host', '127.0.0.1', '--port', String(port)],
    cwd: checkout,
    url: `http://127.0.0.1:${port}`,
  }
}

export function apply(ctx, config = {}) {
  for (const section of PROMPT_SECTIONS) {
    ctx.effect(() => ctx.systemPrompt.section(section), section.name)
  }

  // Native Harness already owns its model loop. The separate workbench is opt-in.
  const launch = resolveWorkbenchLaunch(config.workbench)
  if (launch === null) return
  ctx.effect(() => {
    const child = spawn(launch.command, launch.args, {
      cwd: launch.cwd,
      env: process.env,
      stdio: 'inherit',
    })
    child.on('error', error => {
      console.error(`dsh-triton-riscv: failed to start workbench: ${error.message}`)
    })
    return () => {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM')
    }
  }, 'triton-riscv.workbench')
}
