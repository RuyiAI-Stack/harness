import { EventEmitter } from 'node:events'
import { existsSync } from 'node:fs'
import { PassThrough } from 'node:stream'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { spawn } from 'node:child_process'
import { apply, createMultiAgent } from '../../index.js'

vi.mock('node:child_process', () => ({ spawn: vi.fn() }))

const config = {
  python: '/trusted/venv/bin/python',
  model: {
    provider: 'fixture',
    model: 'classification-model',
    apiKeyEnv: 'FIXTURE_API_KEY',
    requestTimeoutSeconds: 10,
  },
  tasks: { classify: { systemPrompt: 'Classify mail.', resultSchema: { type: 'object' } } },
}
let children: any[]
let services: any[]

function service() {
  const value = createMultiAgent(config)
  services.push(value)
  return value
}
function message(child: any, value: object) {
  child.stdout.write(JSON.stringify(value) + '\n')
}
function finish(child: any, outcome = 'validated') {
  message(child, {
    type: 'result',
    result: {
      session_id: 'task-fixture',
      outcome,
      result: outcome === 'validated' ? { category: 'question' } : null,
      usage: null,
      error: null,
    },
  })
  child.emit('close', 0)
}

beforeEach(() => {
  children = []
  services = []
  vi.spyOn(process, 'kill').mockReturnValue(true)
  vi.mocked(spawn).mockImplementation((...args: any[]) => {
    const child: any = new EventEmitter()
    child.pid = 987654
    child.stdin = new PassThrough()
    child.stdout = new PassThrough()
    child.stderr = new PassThrough()
    child.kill = vi.fn(() => true)
    child.args = args
    child.input = ''
    child.stdin.on('data', (value: Buffer) => {
      child.input += value.toString()
    })
    children.push(child)
    return child
  })
})
afterEach(async () => {
  for (const child of children) child.emit('close', 0)
  for (const value of services) await value.close()
  vi.restoreAllMocks()
  vi.mocked(spawn).mockReset()
})

describe('trusted multiAgent host service', () => {
  it('launches the installed generic module and offers usable status/result/usage APIs', async () => {
    const api = service()
    vi.stubEnv('PARENT_PRIVATE_TOKEN', 'never-forward')
    vi.stubEnv('DSH_HOME', '/parent/home')
    try {
      const { id } = api.run({ task: 'classify', input: { message: 'Where is it?' } })
      const child = children[0]
      expect(child.args[0]).toBe(config.python)
      expect(child.args[1]).toEqual(['-I', '-B', '-m', 'dsh_multi_agent.bridge'])
      expect(child.args[2].cwd).not.toBe(process.cwd())
      expect(child.args[2].env).not.toHaveProperty('PARENT_PRIVATE_TOKEN')
      expect(child.args[2].env).not.toHaveProperty('DSH_HOME')
      expect(child.args[2].env).not.toHaveProperty('PYTHONPATH')
      expect(child.args[2].env.HOME).not.toBe(process.env.HOME)
      expect(JSON.parse(child.input)).toMatchObject({
        input: { message: 'Where is it?' },
        settings: { model: 'classification-model' },
        task: { systemPrompt: 'Classify mail.' },
      })
      expect(api.status(id).status).toBe('running')
      message(child, { type: 'started', session_id: 'task-fixture' })
      const usage = { session_id: 'task-fixture', totals: { total_tokens: 12 } }
      message(child, { type: 'usage', usage })
      expect(api.usage(id)).toEqual(usage)
      finish(child)
      expect(await api.result(id)).toMatchObject({ outcome: 'validated', result: { category: 'question' } })
      expect(api.status(id)).toMatchObject({ sessionId: 'task-fixture', status: 'validated' })
      expect(existsSync(child.args[2].cwd)).toBe(false)
      api.release(id)
      expect(() => api.status(id)).toThrow(/Unknown/)
    } finally {
      vi.unstubAllEnvs()
    }
  })
  it.each(['python', 'cwd', 'tools', 'model', 'systemPrompt', 'sessionId', 'resultSchema'])(
    'rejects request-selected %s',
    key => {
      const api = service()
      expect(() => api.run({ task: 'classify', input: {}, [key]: 'unsafe' })).toThrow(/only task/)
      expect(spawn).not.toHaveBeenCalled()
    },
  )
  it('rejects unknown tasks and oversized input before launching', () => {
    const api = service()
    expect(() => api.run({ task: 'other', input: {} })).toThrow(/Unknown/)
    expect(() => api.run({ task: 'classify', input: 'x'.repeat(128 * 1024) })).toThrow(/budget/)
    expect(spawn).not.toHaveBeenCalled()
  })
  it('keeps cancellation pending until exit and never accepts a late successful result', async () => {
    const api = service()
    const { id } = api.run({ task: 'classify', input: {} })
    message(children[0], { type: 'started', session_id: 'task-fixture' })
    expect(api.cancel(id)).toBe(true)
    expect(api.cancel(id)).toBe(false)
    expect(api.status(id).status).toBe('cancelling')
    expect(() => api.release(id)).toThrow(/active/)
    expect(children[0].kill).toHaveBeenCalledWith('SIGTERM')
    finish(children[0])
    expect(await api.result(id)).toMatchObject({ outcome: 'cancelled', result: null })
    expect(api.cancel(id)).toBe(false)
  })
  it.each(['foreign', 'malformed', 'missing', 'nonzero'])('fails closed on %s bridge output', kind => {
    const api = service()
    const { id } = api.run({ task: 'classify', input: {} })
    const child = children[0]
    message(child, { type: 'started', session_id: 'task-fixture' })
    if (kind === 'foreign') message(child, { type: 'result', result: { session_id: 'other', outcome: 'validated' } })
    if (kind === 'malformed') child.stdout.write('not-json\n')
    child.emit('close', kind === 'nonzero' ? 1 : 0)
    return expect(api.result(id)).resolves.toMatchObject({ outcome: 'error', result: null })
  })
  it('registers the service and closes owned runs with the host lifecycle', async () => {
    let api: any, dispose: any
    const ctx = {
      provide: vi.fn((_name, value) => {
        api = value
      }),
      effect: vi.fn(install => {
        dispose = install()
      }),
    }
    apply(ctx, config)
    services.push(api)
    expect(ctx.provide).toHaveBeenCalledWith('multiAgent', api)
    const { id } = api.run({ task: 'classify', input: {} })
    const result = api.result(id)
    const disposed = dispose()
    expect(children[0].kill).toHaveBeenCalledWith('SIGTERM')
    children[0].emit('close', null)
    await disposed
    expect(await result).toMatchObject({ outcome: 'cancelled' })
    expect(() => api.run({ task: 'classify', input: {} })).toThrow(/closed/)
  })
  it('does not start anything from the default disabled patch', () => {
    const ctx = { provide: vi.fn(), effect: vi.fn() }
    apply(ctx, { enabled: false })
    expect(ctx.provide).not.toHaveBeenCalled()
    expect(spawn).not.toHaveBeenCalled()
  })
})
