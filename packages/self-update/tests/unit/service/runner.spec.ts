import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'
import { describe, expect, it } from 'vitest'
import { InstallRunner, type RunState } from '../../../src/service/runner.ts'

async function waitFor(runner: InstallRunner, timeoutMs = 5000): Promise<RunState> {
  const deadline = Date.now() + timeoutMs
  let state = runner.snapshot()
  while (state.phase === 'running' && Date.now() < deadline) {
    await sleep(25)
    state = runner.snapshot()
  }
  return state
}

function withScript(body: string, fn: (dir: string, script: string, log: string) => Promise<void>): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'self-update-'))
  const script = join(dir, 'install.sh')
  const log = join(dir, 'log.txt')
  writeFileSync(script, `#!/usr/bin/env bash\n${body}\n`)
  return fn(dir, script, log).finally(() => rmSync(dir, { recursive: true, force: true }))
}

describe('InstallRunner', () => {
  it('runs a successful script and records the log', async () => {
    await withScript('echo hello-from-script', async (dir, script, log) => {
      const runner = new InstallRunner({ script, cwd: dir, logPath: log, restart: false })
      const started = runner.start()
      expect(started?.phase).toBe('running')
      expect(started?.pid).toBeTypeOf('number')

      const state = await waitFor(runner)
      expect(state.phase).toBe('ok')
      expect(state.exitCode).toBe(0)
      expect(readFileSync(log, 'utf8')).toContain('hello-from-script')
    })
  })

  it('records a failing script as failed', async () => {
    await withScript('echo boom >&2\nexit 3', async (dir, script, log) => {
      const runner = new InstallRunner({ script, cwd: dir, logPath: log, restart: false })
      runner.start()
      const state = await waitFor(runner)
      expect(state.phase).toBe('failed')
      expect(state.exitCode).toBe(3)
      expect(state.error).toContain('exit code 3')
      expect(readFileSync(log, 'utf8')).toContain('boom')
    })
  })

  it('rejects a second start while running', async () => {
    await withScript('sleep 1', async (dir, script, log) => {
      const runner = new InstallRunner({ script, cwd: dir, logPath: log, restart: false })
      runner.start()
      expect(runner.start()).toBeUndefined()
      const state = await waitFor(runner)
      expect(state.phase).toBe('ok')
    })
  })
})
