import assert from 'node:assert/strict'
import { test } from 'vitest'
import { installNativeAdapter, contextText } from '../../lib/native-adapter.js'

const prefix = 'mcp__triton_riscv__'
function session(id, events = []) {
  return {
    id,
    snapshotEvents: () => events,
    append(type, data) {
      events.push({ type, data: structuredClone(data) })
    },
  }
}
function harness(outcome = 'allowed-once', bridgeOverride) {
  const seen = [],
    questions = [],
    hooks = {}
  const states = new Map()
  const store = {
    load(id) {
      return (
        states.get(id) || {
          session_id: id,
          artifacts: {},
          task: {},
          contract: null,
          memory: null,
        }
      )
    },
    save(state) {
      states.set(state.session_id, structuredClone(state))
    },
  }
  let context
  let variable
  let guard
  const ctx = {
    tools: {
      guard(fn) {
        guard = fn
        return () => {
          guard = undefined
        }
      },
    },
    effect(fn) {
      return fn()
    },
    systemPrompt: {
      context(value) {
        context = value
        return () => {}
      },
      variable(_name, value) {
        variable = value
        return () => {}
      },
    },
    on(name, fn) {
      hooks[name] = fn
    },
    get() {
      return outcome === 'missing'
        ? undefined
        : {
            async request(req) {
              questions.push(req)
              return outcome
            },
          }
    },
  }
  installNativeAdapter(ctx, {
    store,
    bridge: async (req, options) => {
      seen.push(req)
      if (bridgeOverride) return bridgeOverride(req, options)
      return req.action === 'review'
        ? { reason: 'exact command and diff', fingerprint: 'abc' }
        : req.action === 'memory'
          ? {
              status: 'found',
              items: [{ memory_id: 7 }],
              context_excerpt: 'historical evidence #7',
            }
          : { status: 'approved' }
    },
  })
  let calls = 0
  async function call(s, name, args, data, signal = new AbortController().signal) {
    return hooks['tools/execute'](
      {
        name: prefix + name,
        arguments: args,
        agent: { session: s },
        callId: 'c',
        signal,
      },
      async () => {
        calls++
        return {
          isError: false,
          value: { structuredContent: data },
          content: [],
        }
      },
    )
  }
  return {
    call,
    hook: hooks['tools/execute'],
    seen,
    questions,
    guard: exec => guard(exec),
    store,
    context: s => context.text.replace('{{triton_task_evidence}}', variable({ agent: { session: s } })),
    calls: () => calls,
  }
}

test('operator protection starts before the first tool body and survives a task reset', async () => {
  const h = harness(),
    s = session('protected')
  const exec = name => ({ name, agent: { session: s } })
  assert.equal(h.guard(exec('bash')), undefined)
  assert.equal(h.guard(exec(prefix + 'discover_operator')), undefined)
  for (const name of ['bash', 'write', 'edit', 'delegate', 'run_code', 'renamed_shell', prefix + 'unknown'])
    assert.match(h.guard(exec(name)), /restricted/)
  h.store.save({ ...h.store.load(s.id), task: {}, active: {}, artifacts: {} })
  assert.match(h.guard(exec('bash')), /restricted/)
  assert.equal(h.guard({ name: 'bash', agent: { session: session('unrelated') } }), undefined)
})

test('dispatch rechecks a generic call that was prepared before protection started', async () => {
  const h = harness(),
    s = session('race')
  const exec = { name: 'bash', agent: { session: s } }
  assert.equal(h.guard(exec), undefined)
  await plan(h, s)
  let dispatched = false
  await assert.rejects(
    h.hook(exec, async () => {
      dispatched = true
    }),
    /restricted/,
  )
  assert.equal(dispatched, false)
})

test('old persisted artifact sessions are protected; unregistered prefixed tools are rejected', () => {
  const h = harness(),
    s = session('legacy')
  h.store.save({ ...h.store.load(s.id), artifacts: { 'old-run': 'run' } })
  assert.match(h.guard({ name: 'bash', agent: { session: s } }), /restricted/)
  assert.match(h.guard({ name: prefix + 'invented', agent: { session: session('empty') } }), /restricted/)
  assert.match(h.guard({ name: prefix + 'read_operator_file' }), /require a native/)
})

test('unreadable session policy fails closed rather than allowing a generic tool', async () => {
  const h = harness()
  h.store.load = () => {
    throw new Error('State is unreadable')
  }
  const exec = { name: 'bash', agent: { session: session('broken-state') } }
  assert.throws(() => h.guard(exec), /unreadable/)
  let called = false
  await assert.rejects(
    h.hook(exec, async () => {
      called = true
    }),
    /unreadable/,
  )
  assert.equal(called, false)
})
async function plan(h, s, id = 'run-1') {
  await h.call(
    s,
    'validate_operator',
    { execute: false },
    {
      run_id: id,
      operator: 'add',
      status: 'planned',
      command: 'pytest test_add.py',
    },
  )
}

test('status queries require an owned plan and never reactivate an old plan or request approval', async () => {
  const h = harness(),
    s = session('status-owner')
  await plan(h, s, 'old-plan')
  await plan(h, s, 'new-plan')
  const before = structuredClone(h.store.load(s.id))
  await h.call(
    s,
    'get_validation_status',
    { run_id: 'old-plan', remote: true },
    {
      run_id: 'old-plan',
      operator: 'other',
      execution_state: 'unknown',
      remote_state: 'queued',
      test_status: null,
    },
  )
  assert.deepEqual(h.store.load(s.id), before)
  assert.equal(h.questions.length, 0)
  const calls = h.calls()
  await assert.rejects(h.call(session('foreign'), 'get_validation_status', { run_id: 'old-plan' }, {}))
  await assert.rejects(h.call(s, 'get_validation_status', { run_id: 'invented' }, {}))
  assert.equal(h.calls(), calls)
})

test('approval quotes original human text, not model or synthetic user summaries', async () => {
  const h = harness(),
    s = session('human-proof')
  s.append('user/message', {
    id: 'original',
    source: { kind: 'user' },
    content: [{ type: 'text', text: 'Please validate add only' }],
  })
  await plan(h, s)
  s.append('assistant/message', { content: [{ type: 'text', text: 'MODEL-INVENTED-REQUEST' }] })
  s.append('user/message', {
    id: 'synthetic',
    source: { kind: 'tool' },
    content: [{ type: 'text', text: 'SYNTHETIC-REQUEST' }],
  })
  s.append('user/message', { id: 'latest', source: { kind: 'user' }, content: [{ type: 'text', text: 'Continue' }] })
  await h.call(
    s,
    'execute_approved_validation',
    { run_id: 'run-1' },
    { run_id: 'result', operator: 'add', status: 'failed' },
  )
  assert.match(h.questions[0].reason, /Please validate add only/)
  assert.match(h.questions[0].reason, /Continue/)
  assert.doesNotMatch(h.questions[0].reason, /MODEL-INVENTED|SYNTHETIC-REQUEST/)
  assert.equal(h.store.load(s.id).artifact_requests.result.message_id, 'original')
})

test('source refresh creates a new pending plan without executing the stale body', async () => {
  let changed = true
  const h = harness('allowed-once', async req => {
      if (req.action === 'review')
        return { reason: 'source changed', fingerprint: 'v2', source_change: changed ? { files: ['add.py'] } : null }
      if (req.action === 'refresh') {
        changed = false
        return {
          status: 'replanned',
          plan: { run_id: 'run-new', operator: 'add', status: 'planned' },
          tool_result: { run_id: 'run-new', operator: 'add', status: 'planned' },
          next_action: 'approve new plan',
        }
      }
      return { status: 'planned', approval: { status: 'approved' } }
    }),
    s = session('refresh')
  await plan(h, s)
  const output = await h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, { status: 'passed' })
  assert.equal(output.value.structuredContent.status, 'planned')
  assert.equal(JSON.parse(output.value.content[0].text).status, 'replanned')
  assert.equal(h.calls(), 1)
  assert.equal(h.store.load(s.id).active.plan, 'run-new')
  assert.equal(h.store.load(s.id).task.status, 'planned')
  await assert.rejects(h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, {}), /not the current/)
  await h.call(s, 'execute_approved_validation', { run_id: 'run-new' }, { run_id: 'result', status: 'passed' })
  assert.equal(h.questions.length, 2)
  assert.equal(h.calls(), 2)
})

test('rejecting a changed-source baseline neither refreshes nor executes', async () => {
  const h = harness('rejected', async () => ({
      reason: 'source changed',
      fingerprint: 'v2',
      source_change: { files: ['add.py'] },
    })),
    s = session('reject-refresh')
  await plan(h, s)
  await assert.rejects(h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, {}), /Source refresh rejected/)
  assert.equal(h.calls(), 1)
  assert.equal(
    h.seen.some(req => req.action === 'refresh'),
    false,
  )
})

test('changed approvals re-prompt and stop after three unstable reviews', async () => {
  const h = harness('allowed-once', async req =>
      req.action === 'review' ? { reason: 'current source', fingerprint: 'latest' } : { status: 'review_changed' },
    ),
    s = session('unstable')
  await plan(h, s)
  await assert.rejects(h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, {}), /three reviews/)
  assert.equal(h.questions.length, 3)
  assert.equal(h.calls(), 1)
})

test('native approval binds exact session and artifact; rejection has no execution', async () => {
  for (const outcome of ['rejected', 'cancelled', 'unavailable', 'missing']) {
    const h = harness(outcome),
      s = session('s1')
    await plan(h, s)
    await assert.rejects(h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, {}))
    assert.equal(h.calls(), 1)
    assert.equal(h.seen.filter(x => x.action === 'decide').length, outcome === 'rejected' ? 1 : 0)
  }
})

test('native grant commits trusted decision before calling the domain tool', async () => {
  const h = harness(),
    s = session('s1')
  await plan(h, s)
  await h.call(
    s,
    'execute_approved_validation',
    { run_id: 'run-1' },
    { run_id: 'run-result', operator: 'add', status: 'passed' },
  )
  assert.equal(h.calls(), 2)
  assert.deepEqual(
    h.seen.map(x => x.action),
    ['review', 'decide'],
  )
  assert.equal(h.seen[1].session_id, 's1')
  assert.equal(h.seen[1].fingerprint, 'abc')
  assert.match(h.questions[0].reason, /exact command and diff$/)
  assert.match(h.questions[0].reason, /Unavailable in host event history/)
  assert.match(h.context(s), /run-result/)
})

test('conversation text and another session cannot authorize an artifact', async () => {
  const h = harness(),
    a = session('a'),
    b = session('b')
  await plan(h, a)
  await assert.rejects(h.call(b, 'execute_approved_validation', { run_id: 'run-1' }, {}), /not created/)
  assert.equal(h.questions.length, 0)
  const fork = session('fork', a.snapshotEvents())
  await assert.rejects(h.call(fork, 'execute_approved_validation', { run_id: 'run-1' }, {}), /not created/)
})

test('same session restored with domain state retains ownership', async () => {
  const h = harness(),
    s = session('s1')
  await plan(h, s)
  const restored = session('s1', JSON.parse(JSON.stringify(s.snapshotEvents())))
  await h.call(restored, 'execute_approved_validation', { run_id: 'run-1' }, { status: 'passed' })
  assert.equal(h.questions.length, 1)
})

test('generation and failure retrieve evidence automatically, with current run excluded by bridge query', async () => {
  const h = harness(),
    s = session('s1')
  await h.call(
    s,
    'prepare_operator_development',
    {
      specification: { semantics: 'x*x', pytorch_reference: 'torch.square(x)' },
    },
    { operator: 'square', status: 'planned', development_id: 'dev-1' },
  )
  assert.equal(h.seen[0].query.semantics, 'x*x')
  assert.match(h.context(s), /historical evidence #7/)
  await plan(h, s)
  await h.call(
    s,
    'execute_approved_validation',
    { run_id: 'run-1' },
    { operator: 'add', status: 'failed', run_id: 'run-2' },
  )
  assert.equal(h.seen.at(-1).query.run_id, 'run-2')
  assert.doesNotMatch(h.context(s), /torch.square/)
})

test('missing historical DB does not discard a successful preparation', async () => {
  const h = harness('allowed-once', async () => {
      throw new Error('unavailable')
    }),
    s = session('s1')
  await h.call(
    s,
    'prepare_operator_development',
    { specification: { semantics: 'x*x' } },
    { operator: 'square', development_id: 'dev-1' },
  )
  await h.call(
    s,
    'propose_operator_implementation',
    { development_id: 'dev-1' },
    { proposal_id: 'proposal-1', operator: 'square' },
  )
  assert.match(h.context(s), /retrieval unavailable/)
  assert.equal(h.calls(), 2)
})

test('switching to a validation job clears unrelated operator contract and evidence', async () => {
  const h = harness(),
    s = session('s1')
  await h.call(
    s,
    'prepare_operator_development',
    {
      specification: { semantics: 'x*x', pytorch_reference: 'torch.square(x)' },
    },
    { operator: 'square', status: 'planned', development_id: 'dev-1' },
  )
  assert.match(h.context(s), /historical evidence #7/)
  await h.call(s, 'prepare_validation_job', {}, { job_id: 'job-1', status: 'planned' })
  assert.match(h.context(s), /job-1/)
  assert.doesNotMatch(h.context(s), /torch.square|historical evidence #7|dev-1/)
  await assert.rejects(
    h.call(
      s,
      'propose_operator_implementation',
      { development_id: 'dev-1' },
      { operator: 'square', proposal_id: 'p-1' },
    ),
    /not the current/,
  )
  assert.equal(h.calls(), 2)
})

test('changed review and cancellation cannot run the underlying operation', async () => {
  const h = harness('allowed-once', async req => {
      if (req.action === 'decide') throw new Error('Artifact changed during review')
      return { reason: 'cmd', fingerprint: 'old' }
    }),
    s = session('s1')
  await plan(h, s)
  await assert.rejects(h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, {}), /changed/)
  assert.equal(h.calls(), 1)
  const abort = new AbortController()
  abort.abort()
  await assert.rejects(h.call(s, 'discover_operator', {}, {}, abort.signal))
  assert.equal(h.calls(), 1)
})

test('bounded context retains IDs; no partial semantic contract presented as complete', () => {
  const text = contextText({
    task: { operator: 'new', proposal_id: 'p-1', task_file: 'task.md' },
    contract: { semantics: 'x'.repeat(20000) },
    memory: { text: 'm'.repeat(7000) },
  })
  assert.match(text, /p-1/)
  assert.match(text, /read the immutable contract/)
  assert.match(text, /exceeds 6000/)
  assert.ok(text.length < 19000)
})

test('applied source invalidates a stale passing receipt in task context', async () => {
  const h = harness(),
    s = session('s1')
  await plan(h, s)
  await h.call(
    s,
    'execute_approved_validation',
    { run_id: 'run-1' },
    { status: 'passed', operator: 'add', run_id: 'result-1', exit_code: 0 },
  )
  await h.call(s, 'propose_repair', { run_id: 'result-1' }, { operator: 'add', proposal_id: 'fix-1' })
  await h.call(s, 'apply_repair', { proposal_id: 'fix-1' }, { status: 'applied', operator: 'add' })
  assert.match(h.context(s), /not_validated_after_change/)
  assert.doesNotMatch(h.context(s), /result-1/)
})

test('a real but superseded plan ID in the same session cannot execute', async () => {
  const h = harness(),
    s = session('s1')
  await plan(h, s, 'run-24')
  await plan(h, s, 'run-42')
  await assert.rejects(h.call(s, 'execute_approved_validation', { run_id: 'run-24' }, {}), /not the current/)
  assert.equal(h.questions.length, 0)
  await h.call(s, 'execute_approved_validation', { run_id: 'run-42' }, { status: 'passed' })
  assert.equal(h.questions.length, 1)
})

test('queued validation is not a receipt and only its owning session can collect it', async () => {
  const h = harness(),
    s = session('queued-owner')
  await plan(h, s)
  await h.call(
    s,
    'execute_approved_validation',
    { run_id: 'run-1' },
    { status: 'queued', job_id: 'async-1', run_id: 'run-1' },
  )
  assert.match(h.context(s), /queued/)
  await assert.rejects(h.call(s, 'diagnose_failure', { run_id: 'run-1' }, {}), /not created|not the current/)
  await assert.rejects(h.call(session('other'), 'inspect_queued_task', { job_id: 'async-1' }, {}), /not created/)
  await h.call(
    s,
    'inspect_queued_task',
    { job_id: 'async-1' },
    {
      kind: 'validation',
      status: 'succeeded',
      result: { status: 'passed', run_id: 'receipt-1', receipt_path: 'receipt.json' },
    },
  )
  assert.match(h.context(s), /receipt-1/)
  assert.match(h.context(s), /passed/)
})

test('a committed replay uses business checks without another approval dialog', async () => {
  const h = harness('allowed-once', async () => ({ replay: true, status: 'approved' })),
    s = session('s1')
  await plan(h, s)
  await h.call(s, 'execute_approved_validation', { run_id: 'run-1' }, { status: 'passed', run_id: 'result-1' })
  assert.equal(h.questions.length, 0)
  assert.equal(h.calls(), 2)
  assert.match(h.context(s), /result-1/)
})

test('abort during execution sends a separate trusted cancellation request', async () => {
  const h = harness(),
    s = session('s1'),
    abort = new AbortController()
  await plan(h, s)
  await h.hook(
    {
      name: prefix + 'execute_approved_validation',
      arguments: { run_id: 'run-1' },
      agent: { session: s },
      callId: 'c',
      signal: abort.signal,
    },
    async () => {
      abort.abort()
      return { isError: true }
    },
  )
  assert.equal(h.seen.at(-1).action, 'cancel')
  assert.equal(h.seen.at(-1).id, 'run-1')
})

test('diagnosis can select an owned result within the currently active batch', async () => {
  const h = harness(),
    s = session('s1')
  await h.call(s, 'prepare_validation_job', {}, { job_id: 'job-1', status: 'planned' })
  await h.call(
    s,
    'execute_validation_job',
    { job_id: 'job-1' },
    {
      job_id: 'job-1',
      status: 'failed',
      results: [{ id: 'add', run_id: 'run-1', status: 'failed' }],
    },
  )
  await h.call(s, 'diagnose_failure', { run_id: 'run-1' }, { operator: 'add', run_id: 'run-1', status: 'failed' })
  await h.call(s, 'propose_repair', { run_id: 'run-1' }, { operator: 'add', proposal_id: 'repair-1' })
  assert.equal(h.calls(), 4)
})
