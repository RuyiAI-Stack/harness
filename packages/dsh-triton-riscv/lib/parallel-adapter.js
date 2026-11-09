// Only the host may supply session identity and record a human approval.
export async function parallelCall(ctx, exec, name, state, store, bridge, human, reason) {
  const selection = exec.agent.session.requestHeader?.()?.config ?? exec.agent.options ?? {}
  const request = {
    session_id: exec.agent.session.id,
    id: exec.arguments?.plan_id,
    model_selection: { provider: selection.provider, model: selection.model },
  }
  const invoke = (action, extra = {}, options = {}) =>
    bridge({ action: `parallel-${action}`, ...request, ...extra }, { signal: exec.signal, ...options })
  let plan
  if (name === 'prepare_parallel_tasks') {
    if (!human) throw new Error('Parallel preparation requires an actual user request')
    plan = await invoke('prepare', { request: exec.arguments, human })
    state.parallel_plan = plan.plan_id
    store.save(state)
  } else {
    if (!request.id || state.parallel_plan !== request.id)
      throw new Error('Not the current parallel plan in this session')
    plan = await invoke('get')
  }
  request.id = plan.plan_id
  if (name === 'get_parallel_tasks') return result(plan)
  if (plan.status === 'pending') {
    const approval = ctx.get('approval')
    if (!approval) throw new Error('Human approval service unavailable; replicas not started')
    const outcome = await approval.request({
      agent: exec.agent,
      toolName: exec.name,
      callId: exec.callId,
      reason: reason + '\nRead-only parallel replicas (additional model calls):\n' + JSON.stringify(plan, null, 2),
      signal: exec.signal,
    })
    exec.signal.throwIfAborted()
    plan = await invoke('decide', { digest: plan.digest, approved: outcome === 'allowed-once' })
  }
  if (name !== 'execute_parallel_tasks' || plan.status === 'rejected') return result(plan)
  let cancellation
  const cancel = () => {
    cancellation ||= Promise.resolve()
      .then(() => bridge({ action: 'parallel-cancel', ...request }, { timeoutMs: 15000 }))
      .then(
        () => undefined,
        error => error,
      )
  }
  exec.signal.addEventListener('abort', cancel, { once: true })
  try {
    exec.signal.throwIfAborted()
    const seconds = Math.min(300, Math.max(30, Number(plan.timeout_seconds) || 120))
    const batches = Math.ceil(Math.min(4, plan.tasks?.length || 4) / 2)
    // The coordinator owns child cleanup and quota leases. Abort requests cancellation,
    // but must not SIGKILL that coordinator before it has reaped its replicas.
    const completed = await invoke('execute', {}, { signal: undefined, timeoutMs: (seconds * batches + 30) * 1000 })
    exec.signal.throwIfAborted()
    return result(completed)
  } finally {
    exec.signal.removeEventListener('abort', cancel)
    const failure = await cancellation
    if (failure) throw failure
  }
}

function result(plan) {
  return { value: { structuredContent: plan, content: [{ type: 'text', text: JSON.stringify(plan) }] } }
}
