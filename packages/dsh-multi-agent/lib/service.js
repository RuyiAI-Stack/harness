import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { isAbsolute, join } from 'node:path'

const MAX_INPUT = 128 * 1024
const MAX_OUTPUT = 16 * 1024 * 1024
const TERMINAL = new Set(['validated', 'invalid_output', 'incomplete', 'error', 'cancelled'])

function object(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function text(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${label} is required`)
  return value
}

function configuration(config) {
  const python = text(config.python, 'Absolute Python executable')
  if (!isAbsolute(python)) throw new TypeError('Python executable must be absolute')
  const model = config.model
  if (!object(model)) throw new TypeError('Trusted model configuration is required')
  for (const key of ['provider', 'model', 'apiKeyEnv']) text(model[key], key)
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(model.apiKeyEnv)) throw new TypeError('Invalid credential environment name')
  if (model.baseURL !== undefined && model.baseURL !== null) {
    const url = new URL(model.baseURL)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password)
      throw new TypeError('Model baseURL must be HTTP(S) without credentials')
  }
  const timeout = model.requestTimeoutSeconds ?? 900
  if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 86400) throw new TypeError('Invalid request timeout')
  if (!object(config.tasks)) throw new TypeError('Trusted task definitions are required')
  const tasks = new Map()
  for (const [name, task] of Object.entries(config.tasks)) {
    text(name, 'Task name')
    text(task.systemPrompt, 'Task system prompt')
    if (!object(task.resultSchema)) throw new TypeError('A task resultSchema is required')
    tasks.set(name, structuredClone({ systemPrompt: task.systemPrompt, resultSchema: task.resultSchema }))
  }
  return {
    python,
    tasks,
    model: {
      provider: model.provider,
      model: model.model,
      api_key_env: model.apiKeyEnv,
      base_url: model.baseURL ?? null,
      request_timeout_seconds: timeout,
    },
  }
}

function environment(config, home) {
  const result = { HOME: home, USERPROFILE: home, TMPDIR: home, PYTHONUNBUFFERED: '1' }
  // Runtime discovery and networking only; no parent DSH/PYTHON/NODE configuration.
  for (const key of [
    'PATH',
    'SystemRoot',
    'WINDIR',
    'LANG',
    'LC_ALL',
    'SSL_CERT_FILE',
    'SSL_CERT_DIR',
    'HTTP_PROXY',
    'HTTPS_PROXY',
    'ALL_PROXY',
    'NO_PROXY',
    'http_proxy',
    'https_proxy',
    'all_proxy',
    'no_proxy',
    config.model.api_key_env,
  ]) {
    if (process.env[key] !== undefined) result[key] = process.env[key]
  }
  return result
}

/** Host-owned one-shot executions, not a delegation scheduler or a model-facing tool. */
export function createMultiAgent(config) {
  const trusted = configuration(config)
  const runs = new Map()
  let closed = false

  function record(id) {
    const run = runs.get(id)
    if (!run) throw new Error('Unknown multiAgent run')
    return run
  }

  function run(request) {
    if (closed) throw new Error('multiAgent is closed')
    if (!object(request) || Object.keys(request).some(key => !['task', 'input'].includes(key)))
      throw new TypeError('A run accepts only task and JSON input')
    const definition = trusted.tasks.get(request.task)
    if (!definition) throw new Error('Unknown registered task')
    const input = JSON.stringify(request.input)
    if (input === undefined || Buffer.byteLength(input) > MAX_INPUT)
      throw new TypeError('Task input exceeds its JSON budget')
    const payload = JSON.stringify({ settings: trusted.model, task: definition, input: JSON.parse(input) })
    if (Buffer.byteLength(payload) > 2 * MAX_INPUT) throw new TypeError('Task definition exceeds bridge budget')
    const id = randomUUID()
    const directory = mkdtempSync(join(tmpdir(), 'dsh-multi-agent-'))
    const home = join(directory, 'home')
    mkdirSync(home, { mode: 0o700 })
    let child
    try {
      child = spawn(trusted.python, ['-I', '-B', '-m', 'dsh_multi_agent.bridge'], {
        cwd: directory,
        env: environment(trusted, home),
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: process.platform !== 'win32',
        windowsHide: true,
      })
    } catch (error) {
      rmSync(directory, { recursive: true, force: true })
      throw error
    }
    let resolveResult
    const state = {
      id,
      sessionId: null,
      status: 'running',
      usage: null,
      cancelled: false,
      child,
      result: new Promise(resolve => {
        resolveResult = resolve
      }),
    }
    runs.set(id, state)
    let buffer = '',
      bytes = 0,
      response,
      failure,
      killTimer

    function signal(value, group = false) {
      try {
        if (group && process.platform !== 'win32' && child.pid) process.kill(-child.pid, value)
        else child.kill(value)
      } catch (error) {
        if (error.code !== 'ESRCH') failure = 'Could not signal the isolated runtime'
      }
    }

    function stop() {
      signal('SIGTERM')
      killTimer ??= setTimeout(() => signal('SIGKILL', true), 5000)
      killTimer.unref()
    }
    state.stop = stop

    function receive(line) {
      const message = JSON.parse(line)
      if (message.type === 'started') {
        if (state.sessionId || typeof message.session_id !== 'string') throw new Error('Invalid session identity')
        state.sessionId = message.session_id
      } else if (message.type === 'usage') {
        if (message.usage?.session_id !== state.sessionId) throw new Error('Foreign usage')
        state.usage = message.usage
      } else if (message.type === 'result') {
        if (response || !TERMINAL.has(message.result?.outcome) || message.result.session_id !== state.sessionId)
          throw new Error('Invalid result envelope')
        response = message.result
      } else if (message.type === 'error') {
        failure = 'Isolated Python backend failed'
      } else if (message.type !== 'event') throw new Error('Invalid bridge message')
    }

    child.stdout.setEncoding('utf8')
    child.stdout.on('data', chunk => {
      bytes += Buffer.byteLength(chunk)
      if (bytes > MAX_OUTPUT) {
        failure = 'Bridge output exceeded its budget'
        stop()
        return
      }
      buffer += chunk
      let end
      while ((end = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, end)
        buffer = buffer.slice(end + 1)
        try {
          receive(line)
        } catch {
          failure = 'Invalid isolated backend protocol'
          stop()
          break
        }
      }
    })
    // Drain diagnostics but never expose credentials or unbounded exception text.
    child.stderr.resume()
    child.stdin.on('error', () => {
      failure ??= 'Could not send task to Python backend'
    })
    child.on('error', () => {
      failure = 'Could not launch installed Python backend'
    })
    const timeout = setTimeout(
      () => {
        failure = 'Isolated task timed out'
        stop()
      },
      (trusted.model.request_timeout_seconds + 35) * 1000,
    )
    timeout.unref()
    child.on('close', code => {
      clearTimeout(timeout)
      clearTimeout(killTimer)
      // Also reap descendants if Python died before its SDK finally block.
      signal('SIGKILL', true)
      rmSync(directory, { recursive: true, force: true })
      let result = response
      if (state.cancelled)
        result = {
          session_id: state.sessionId,
          outcome: 'cancelled',
          result: null,
          error: 'Host cancelled the isolated task',
          usage: state.usage ? { ...state.usage, outcome: 'cancelled' } : null,
        }
      else if (failure || code !== 0 || !response || buffer.trim())
        result = {
          session_id: state.sessionId,
          outcome: 'error',
          result: null,
          error: failure ?? 'Isolated backend exited without a complete result',
          usage: state.usage,
        }
      state.status = result.outcome
      state.usage = result.usage ?? state.usage
      state.child = null
      resolveResult(result)
    })
    child.stdin.end(payload + '\n')
    return Object.freeze({ id })
  }

  return Object.freeze({
    run,
    status(id) {
      const state = record(id)
      return { id, sessionId: state.sessionId, status: state.status }
    },
    async result(id) {
      return structuredClone(await record(id).result)
    },
    usage(id) {
      return structuredClone(record(id).usage)
    },
    cancel(id) {
      const state = record(id)
      if (TERMINAL.has(state.status) || state.cancelled) return false
      state.cancelled = true
      state.status = 'cancelling'
      state.stop()
      return true
    },
    release(id) {
      if (!TERMINAL.has(record(id).status)) throw new Error('Cannot release an active run')
      runs.delete(id)
    },
    async close() {
      closed = true
      for (const state of runs.values()) {
        if (!TERMINAL.has(state.status)) {
          state.cancelled = true
          state.status = 'cancelling'
          state.stop()
        }
      }
      await Promise.all([...runs.values()].map(state => state.result))
      runs.clear()
    },
  })
}
