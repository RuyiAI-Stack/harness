import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { isAbsolute, join } from 'node:path'

export const name = 'triton-riscv-domain-policy'

export const inject = ['systemPrompt']

export const TRITON_RISCV_SYSTEM_PROMPT = readFileSync(
  new URL('./policy.md', import.meta.url),
  'utf8',
).trim()

export function resolveWorkbenchLaunch(env = process.env) {
  if (env.TRITON_RISCV_WORKBENCH_AUTOSTART === '0') return null

  const checkout = (env.TRITON_RISCV_CHECKOUT ?? '').trim()
  if (!checkout) {
    throw new Error('TRITON_RISCV_CHECKOUT is required to start the Triton-RISCV workbench')
  }
  if (!isAbsolute(checkout)) {
    throw new Error('TRITON_RISCV_CHECKOUT must be an absolute path')
  }

  const portText = (env.TRITON_RISCV_WORKBENCH_PORT ?? '8765').trim()
  if (!/^\d+$/.test(portText)) {
    throw new Error('TRITON_RISCV_WORKBENCH_PORT must be an integer')
  }
  const port = Number(portText)
  if (port < 1 || port > 65535) {
    throw new Error('TRITON_RISCV_WORKBENCH_PORT must be between 1 and 65535')
  }

  return {
    command: env.TRITON_RISCV_MCP_PYTHON ?? join(checkout, '.harness-venv', 'bin', 'python'),
    args: ['-m', 'codex_agent.platform', '--host', '127.0.0.1', '--port', String(port)],
    cwd: checkout,
    url: `http://127.0.0.1:${port}`,
  }
}

export function apply(ctx) {
  ctx.effect(
    () => ctx.systemPrompt.section({
      name: 'tool:triton-riscv',
      order: 180,
      text: TRITON_RISCV_SYSTEM_PROMPT,
    }),
    'triton-riscv.system-prompt',
  )

  const launch = resolveWorkbenchLaunch()
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
