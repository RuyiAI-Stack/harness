import { expect, it, vi } from 'vitest'
import { parallelCall } from '../../lib/parallel-adapter.js'

function execution(signal = new AbortController().signal) {
  return {
    agent: { session: { id: 'safety-session' }, options: { provider: 'fixture', model: 'startup' } },
    arguments: { plan_id: 'parallel-fixture' },
    signal,
  }
}

const approved = {
  plan_id: 'parallel-fixture',
  status: 'approved',
  digest: 'fixture-digest',
  timeout_seconds: 30,
  tasks: [{ task_id: 'child-1' }, { task_id: 'child-2' }],
}

it('keeps the execution bridge alive and waits for replica cleanup after cancellation', async () => {
  const controller = new AbortController()
  const started = Promise.withResolvers<void>()
  const drained = Promise.withResolvers<object>()
  const cancelled = Promise.withResolvers<void>()
  let executionSignal: AbortSignal | undefined
  let settled = false
  const bridge = vi.fn(async (request, options) => {
    if (request.action === 'parallel-execute') {
      executionSignal = options.signal
      started.resolve()
      return drained.promise
    }
    if (request.action === 'parallel-cancel') cancelled.resolve()
    return { ...approved }
  })
  const running = parallelCall(
    {},
    execution(controller.signal),
    'execute_parallel_tasks',
    { parallel_plan: approved.plan_id },
    {},
    bridge,
    null,
    '',
  )
  const observed = running.then(
    value => {
      settled = true
      return value
    },
    error => {
      settled = true
      return error
    },
  )
  await started.promise
  controller.abort(new Error('user cancellation'))
  await cancelled.promise
  await Promise.resolve()
  expect(executionSignal).toBeUndefined()
  expect(settled).toBe(false)
  expect(bridge.mock.calls.filter(([request]) => request.action === 'parallel-cancel')).toHaveLength(1)
  drained.resolve({ ...approved, status: 'cancelled' })
  expect(await observed).toMatchObject({ message: 'user cancellation' })
  expect(settled).toBe(true)
})

it('reports cancellation persistence failure only after the execution bridge settles', async () => {
  const controller = new AbortController()
  const started = Promise.withResolvers<void>()
  const drained = Promise.withResolvers<object>()
  const attempted = Promise.withResolvers<void>()
  const bridge = async request => {
    if (request.action === 'parallel-execute') {
      started.resolve()
      return drained.promise
    }
    if (request.action === 'parallel-cancel') {
      attempted.resolve()
      throw new Error('cancel persistence failed')
    }
    return { ...approved }
  }
  let settled = false
  const running = parallelCall(
    {},
    execution(controller.signal),
    'execute_parallel_tasks',
    { parallel_plan: approved.plan_id },
    {},
    bridge,
    null,
    '',
  ).catch(error => {
    settled = true
    return error
  })
  await started.promise
  controller.abort()
  await attempted.promise
  await Promise.resolve()
  expect(settled).toBe(false)
  drained.resolve({ ...approved, status: 'completed' })
  expect(await running).toMatchObject({ message: 'cancel persistence failed' })
})

it('uses startup options only before a host request header exists', async () => {
  const exec = execution()
  const bridge = vi.fn(async () => ({ ...approved }))
  await parallelCall({}, exec, 'get_parallel_tasks', { parallel_plan: approved.plan_id }, {}, bridge, null, '')
  expect(bridge.mock.calls[0][0].model_selection).toEqual({ provider: 'fixture', model: 'startup' })
})
