import { spawn } from 'node:child_process'
import { isAbsolute } from 'node:path'
import { bridgeEnvironment, configFromEnvironment, resolveConfig } from './config.js'

// This host-only channel is intentionally not registered as an MCP tool.
export function callBridge(request, { signal, config, env = process.env, timeoutMs = 30_000 } = {}) {
  const settings = config || resolveConfig(configFromEnvironment(env))
  const root = settings.repoRoot
  if (!root || !isAbsolute(root)) throw new Error('Set an absolute TRITON_RISCV_REPO_ROOT for the native plugin')
  const python = settings.python
  signal?.throwIfAborted()
  return new Promise((accept, reject) => {
    const child = spawn(python, ['-I', '-m', 'codex_agent.harness.native_bridge'], {
      cwd: root,
      env: bridgeEnvironment(settings, env),
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let output = ''
    let size = 0
    let failure
    const stop = reason => {
      failure ||= reason
      child.kill('SIGKILL')
    }
    const abort = () => stop(new Error('Native bridge cancelled'))
    const timer = setTimeout(() => stop(new Error('Native bridge timed out')), timeoutMs)
    signal?.addEventListener('abort', abort, { once: true })
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', data => {
      size += Buffer.byteLength(data)
      if (size > 2_000_000) stop(new Error('Native bridge output exceeds limit'))
      else output += data
    })
    // Do not forward arbitrary stderr (it may contain environment/config values).
    child.stderr.resume()
    child.stdin.on('error', error => {
      failure ||= error
    })
    child.on('error', error => {
      failure ||= error
    })
    child.on('close', code => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
      if (failure) return reject(failure)
      try {
        const result = JSON.parse(output)
        if (code !== 0 || result.error) throw new Error(result.error || `Native bridge exited ${code}`)
        accept(result)
      } catch (error) {
        reject(error)
      }
    })
    child.stdin.end(JSON.stringify(request))
  })
}
