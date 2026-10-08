import { createHash } from 'node:crypto'
import { realpathSync, statSync } from 'node:fs'
import { basename, dirname, isAbsolute, join } from 'node:path'
import { mcpConfiguration, resolveConfig } from './config.js'
import { installNativeAdapter } from './native-adapter.js'
import { callBridge } from './native-bridge.js'
import { createStateStore } from './native-state.js'

function workspace(path) {
  if (typeof path !== 'string' || !isAbsolute(path) || /[\0\r\n]/.test(path))
    throw new Error('Harness session workspace must be an absolute directory')
  const root = realpathSync(path)
  if (!statSync(root).isDirectory()) throw new Error('Harness session workspace must be a directory')
  return root
}

export function resolveSessionConfig(input, session) {
  const base = resolveConfig(input)
  const selected = session?.header?.cwd
  if (selected === undefined && !base.repoRoot) return null
  // Only the host's persisted session metadata selects a repository, never tool arguments.
  const root = workspace(selected === undefined ? base.repoRoot : selected)
  let legacyRoot
  try {
    legacyRoot = base.repoRoot ? workspace(base.repoRoot) : undefined
  } catch {
    /* A stale fallback cannot override a selected workspace. */
  }
  const legacy = root === legacyRoot
  const id = createHash('sha256').update(root).digest('hex')
  const stateDir = input.stateDir
    ? legacy
      ? base.stateDir
      : join(base.stateDir, 'workspaces', id)
    : join(root, 'agent-results')
  const database = input.memory?.database
    ? legacy
      ? base.memory.database
      : join(dirname(base.memory.database), 'workspaces', id, basename(base.memory.database))
    : join(stateDir, 'memory.sqlite3')
  return resolveConfig({ ...input, repoRoot: root, stateDir, memory: { ...input.memory, database } })
}

export function installWorkspaceHost(ctx, input, { mcp, createScope }) {
  const entries = new Map()
  const disposed = new WeakSet()
  let stopped = false
  function check(agent, entry) {
    if (stopped || disposed.has(agent) || entry.closed) throw new Error('Triton-RISCV session is closed')
    if (resolveSessionConfig(input, agent.session)?.repoRoot !== entry.root)
      throw new Error('Session workspace changed; open a new session and review a new plan. No action executed.')
  }
  async function ensure(agent) {
    const current = entries.get(agent)
    if (current) {
      check(agent, current)
      await current.pending
      return current
    }
    if (stopped || disposed.has(agent)) throw new Error('Triton-RISCV session is closed')
    const config = resolveSessionConfig(input, agent.session)
    if (!config) return null
    const scope = createScope(ctx, agent)
    const entry = { root: config.repoRoot, scope, ready: false, closed: false, pending: null }
    entries.set(agent, entry)
    entry.pending = (async () => {
      try {
        installNativeAdapter(scope.ctx, {
          owner: agent,
          store: createStateStore(config),
          bridge: (request, options) => callBridge(request, { ...options, config }),
        })
        await scope.ctx.plugin(mcp, mcpConfiguration(config))
        check(agent, entry)
        entry.ready = true
      } catch (error) {
        await scope.dispose()
        throw error
      }
    })()
    await entry.pending
    return entry
  }
  function deny(exec) {
    const entry = entries.get(exec.agent)
    if (!entry)
      return exec.name.startsWith('mcp__triton_riscv__')
        ? 'Select a workspace and initialize this session before using Triton-RISCV tools'
        : undefined
    try {
      check(exec.agent, entry)
    } catch (error) {
      return error.message
    }
    if (!entry.ready) return 'Triton-RISCV workspace tools are not ready; no action executed'
    return undefined
  }
  // A child agent must not borrow an ancestor's MCP connection before its own initialization.
  ctx.effect(() => ctx.tools.guard(deny), 'triton-riscv.workspace-guard')
  ctx.on('tools/execute', async (exec, next) => {
    const reason = deny(exec)
    if (reason) throw new Error(reason)
    return next()
  })
  ctx.on('system-prompt/assemble', async (_assembly, context, next) => {
    if (!context.agent) return next()
    const ready = entries.get(context.agent)?.ready
    const entry = await ensure(context.agent)
    context.signal?.throwIfAborted()
    // Assembly snapshots tools before middleware. Rebuild once after mounting
    // so the first model request already contains this workspace's tool schemas.
    if (entry && !ready) return ctx.systemPrompt.assemble(context)
    return next()
  })
  async function close(agent, entry) {
    disposed.add(agent)
    entry.closed = true
    await entry.scope.dispose()
    await entry.pending?.catch(() => {})
    entries.delete(agent)
  }
  ctx.on('agent/disposed', async ({ agent }) => {
    disposed.add(agent)
    const entry = entries.get(agent)
    if (entry) await close(agent, entry)
  })
  ctx.effect(
    () => async () => {
      stopped = true
      await Promise.all([...entries].map(([agent, entry]) => close(agent, entry)))
    },
    'triton-riscv.workspace-connections',
  )
}
