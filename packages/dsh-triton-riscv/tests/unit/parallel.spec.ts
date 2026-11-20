import { expect, it, vi } from 'vitest'
import { apply } from '../../python/codex_agent/harness/replica-policy.mjs'
import { parallelCall } from '../../lib/parallel-adapter.js'
import { apply as parentPolicy } from '../../python/codex_agent/harness/parallel-policy.mjs'

it('a bound workbench turn cannot use shell or arbitrary delegates to forge approval', () => {
  let guard
  parentPolicy(
    {
      effect(fn) {
        fn()
      },
      tools: {
        guard(fn) {
          guard = fn
        },
      },
    },
    { typedOnly: true },
  )
  const agent = { session: { id: 'bound-session' } }
  expect(guard({ agent, name: 'bash' })).toMatch(/approval/)
  expect(guard({ agent, name: 'delegate_task' })).toMatch(/approval/)
  expect(guard({ agent, name: 'mcp__triton_riscv__execute_parallel_tasks' })).toBeUndefined()
})

it('blocks nested delegation/writes and caps read tool dispatches', async () => {
  let guard, dispatch
  const ctx = {
    tools: {
      guard(fn) {
        guard = fn
      },
    },
    effect(fn) {
      fn()
    },
    on(name, fn) {
      if (name === 'tools/execute') dispatch = fn
    },
    systemPrompt: { variable() {}, section() {} },
  }
  apply(ctx, { tools: ['discover_operator'], maxToolCalls: 2, systemPrompt: 'Read only' })
  const read = { name: 'mcp__triton_riscv__discover_operator' }
  expect(guard({ name: 'mcp__triton_riscv__prepare_parallel_tasks' })).toMatch(/denied/)
  expect(guard({ name: 'bash' })).toMatch(/denied/)
  expect(guard(read)).toBeUndefined()
  const next = vi.fn(async () => 'read')
  await dispatch(read, next)
  await dispatch(read, next)
  expect(guard(read)).toMatch(/budget/)
  await expect(dispatch(read, next)).rejects.toThrow(/denied/)
  expect(next).toHaveBeenCalledTimes(2)
})

it.each(['allowed-once', 'denied'])('native approval %s is persisted before any execution', async outcome => {
  const seen = []
  const plan = { plan_id: 'parallel-fixture', status: 'pending', digest: 'digest' }
  const bridge = vi.fn(async request => {
    seen.push(request.action)
    if (request.action === 'parallel-decide') plan.status = request.approved ? 'approved' : 'rejected'
    if (request.action === 'parallel-execute') plan.status = 'completed'
    return { ...plan }
  })
  const approval = vi.fn(async () => outcome)
  const ctx = { get: () => ({ request: approval }) }
  const state = {}
  const store = { save: vi.fn() }
  const exec = { agent: { session: { id: 's1' } }, arguments: { tasks: [] }, signal: new AbortController().signal }
  await parallelCall(
    ctx,
    exec,
    'prepare_parallel_tasks',
    state,
    store,
    bridge,
    { message_id: 'm', text: 'human' },
    'Review',
  )
  expect(seen).toEqual(['parallel-prepare', 'parallel-decide'])
  expect(approval).toHaveBeenCalledTimes(1)
  exec.arguments = { plan_id: 'parallel-fixture' }
  await parallelCall(ctx, exec, 'execute_parallel_tasks', state, store, bridge, null, '')
  expect(seen.includes('parallel-execute')).toBe(outcome === 'allowed-once')
  exec.arguments = { plan_id: 'other-plan' }
  await expect(parallelCall(ctx, exec, 'execute_parallel_tasks', state, store, bridge, null, '')).rejects.toThrow(
    /current parallel plan/,
  )
})
