import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const roles = ['main', 'semantic', 'reviewer', 'implementer', 'reviewer']
const readTools = ['discover_operator', 'read_operator_file', 'inspect_project', 'retrieve_operator_memory']

function invoke(options: object) {
  expect(process.env.TRITON_MYSQL_URL, 'Parent run_checks must supply a disposable database').toBeTruthy()
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) =>
        !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) &&
        !/^(TRITON_RISCV_|TRITON_WORKBENCH_|DSH_|RISCV_|ISRC_)/.test(key),
    ),
  )
  const result = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python'),
    [
      '-I',
      '-B',
      '-c',
      'import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2], run_name="__main__")',
      join(root, 'python'),
      join(root, 'tests/integration/workflow-parallel.fixture.py'),
    ],
    { env, input: JSON.stringify(options), encoding: 'utf8', timeout: 60000, maxBuffer: 8 * 1024 * 1024 },
  )
  expect(result.error, result.stderr).toBeUndefined()
  expect(result.status, JSON.stringify(options) + '\n' + result.stdout + result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

function expectPaused(value: ReturnType<typeof invoke>) {
  expect(value.original_status).toBe('succeeded')
  expect(value.initial).toMatchObject({
    view: { state: 'waiting_parallel_approval', parallel_plan: { status: 'pending', results: [] } },
    roles: ['main'],
    clones: [],
    executions: [],
    files_exist: [false, false],
    counts: { jobs: 1, outbox: 1, runs: 1 },
  })
}

function expectQueuedOnce(value: ReturnType<typeof invoke>, approved: boolean) {
  expectPaused(value)
  for (const decision of value.decisions) {
    expect(decision.status, JSON.stringify(decision)).toBe(200)
    expect(decision.body).toMatchObject({
      id: value.initial.view.id,
      state: 'preparing',
      parallel_plan: { plan_id: value.initial.view.parallel_plan.plan_id, status: approved ? 'approved' : 'rejected' },
      job: { id: value.continuation.id, status: 'queued' },
    })
  }
  expect(value.continuation.id).not.toBe(value.original_id)
  expect(value.continuation.payload).toMatchObject({ workflow_id: value.initial.view.id })
  expect(value.continuation.payload.fingerprint).toBe(value.initial.view.job.payload.fingerprint)
  expect(value.continuation.run_id).not.toBe(value.initial.view.job.run_id)
  expect(value.queued).toMatchObject({
    roles: ['main'],
    clones: [],
    executions: [],
    counts: { jobs: 2, outbox: 2, runs: 2 },
  })
}

function expectSequentialFinish(value: ReturnType<typeof invoke>) {
  expect(value.execution.status).toBe('succeeded')
  expect(value.execution.result).toMatchObject({ validation: 'not_started', files_applied: false })
  expect(value.final).toMatchObject({
    roles,
    phases: ['preparing', 'preparing', 'planning_tests', 'implementing', 'reviewing'],
    files_exist: [false, false],
    view: { state: 'waiting_approval', artifact_error: null, proposal: { status: 'pending_approval' } },
    counts: { jobs: 2, outbox: 2, runs: 2 },
  })
  expect(value.final.view.steps.map((step: { job_id: string }) => step.job_id)).toEqual([
    value.original_id,
    ...Array(4).fill(value.continuation.id),
  ])
  expect(value.old_job_status_after_resume).toBe('succeeded')
  expect(value.replay.status).toBe(200)
  expect(value.replay.body.job.id).toBe(value.continuation.id)
  expect(value.opposite_decision.status).toBe(409)
  expect(value.redelivery_claims).toEqual([null, null])
  expect(value.after_replay).toEqual(value.final)
  expect(value.raw_preserved_after_replay).toBe(true)
}

function parallelEvidence(value: ReturnType<typeof invoke>) {
  return value.calls
    .slice(1)
    .map((call: { evidence: { id: string; text: string; verified: boolean }[] }) =>
      call.evidence.filter(evidence => evidence.id === 'workflow_parallel'),
    ) as { id: string; text: string; verified: boolean }[][]
}

function expectParallelStatus(
  value: ReturnType<typeof invoke>,
  status: 'declined' | 'completed' | 'partial' | 'unavailable',
) {
  expect(value.calls[0].parallel_status).toBe('not_requested')
  expect(value.calls.slice(1).map((call: { parallel_status: string }) => call.parallel_status)).toEqual(
    Array(4).fill(status),
  )
  for (const audit of value.audits.slice(1)) {
    expect(audit.request.parallel_status).toBe(status)
    const policy = audit.system_prompt.replace(/\s+/g, ' ')
    expect(policy).toContain('parallel_status is host-supplied workflow metadata')
    expect(policy).toContain(
      'If declined, the user declined OPTIONAL parallel research; proceed with the normal sequential role',
    )
    expect(policy).toContain('If unavailable or partial, likewise use the evidence actually available.')
    expect(policy).toContain('Missing optional investigation of reference operators is NOT a blocking requirement')
    expect(policy).toContain('must not appear in unknowns or justify insufficient_evidence')
    expect(policy).toContain(
      "Real gaps in the operator's required semantics, frozen acceptance tests or necessary source remain blockers",
    )
  }
}

describe('optional workflow parallel research', () => {
  it.each(['absent', 'null'])(
    'leaves the sequential workflow unchanged for a %s request',
    request => {
      const value = invoke({ scenario: 'default', request })
      expect(value.original_status).toBe('succeeded')
      expect(value.initial).toMatchObject({
        roles,
        clones: [],
        executions: [],
        files_exist: [false, false],
        view: {
          state: 'waiting_approval',
          parallel_plan: null,
          artifact_error: null,
          proposal: { status: 'pending_approval' },
        },
        counts: { jobs: 1, outbox: 1, runs: 1 },
      })
      expect(value.plan_files).toBe(0)
    },
    65000,
  )

  it('persists the main request and pauses before semantics, implementation or replicas', () => {
    const value = invoke({ scenario: 'pause' })
    expectPaused(value)
    const plan = value.initial.view.parallel_plan
    expect(plan).toMatchObject({
      session_id: value.initial.view.session_id,
      workflow_id: value.initial.view.id,
      max_parallel: 2,
      max_tool_calls: 8,
      model: 'fixture-model',
      tools: readTools,
    })
    expect(value.initial_raw_scope).toBe('workflow-' + value.initial.view.id)
    expect(plan.tasks).toHaveLength(2)
    expect(plan.timeout_seconds).toBeGreaterThan(0)
    expect(plan.timeout_seconds).toBeLessThanOrEqual(300)
    expect(value.plan_files).toBe(1)
    expect(value.initial.view.steps).toHaveLength(1)
    expect(value.audits).toHaveLength(1)
    const output = JSON.parse(value.audits[0].raw.final_response)
    expect(output.parallel_request).toMatchObject({ reason: plan.reason, context: plan.context })
    expect(output.parallel_request.tasks.map((task: { goal: string }) => task.goal)).toEqual(
      plan.tasks.map((task: { goal: string }) => task.goal),
    )
  }, 65000)

  it('denies once, queues one continuation and runs the remaining roles once without clones', () => {
    const value = invoke({ approved: false })
    expectQueuedOnce(value, false)
    expectSequentialFinish(value)
    expectParallelStatus(value, 'declined')
    expect(value.final.clones).toEqual([])
    expect(value.final.executions).toEqual([])
    expect(value.final.view.parallel_plan).toMatchObject({ status: 'rejected', results: [] })
    expect(parallelEvidence(value)).toEqual([[], [], [], []])
  }, 65000)

  it('runs approved clones once and supplies unverified findings to every subsequent role', () => {
    const value = invoke({})
    expectQueuedOnce(value, true)
    expectSequentialFinish(value)
    expectParallelStatus(value, 'completed')
    expect(value.final.clones).toEqual(['child-1', 'child-2'])
    expect(value.final.executions).toEqual([
      {
        plan_id: value.initial.view.parallel_plan.plan_id,
        session_id: 'workflow-' + value.initial.view.id,
        has_stop: true,
      },
    ])
    expect(value.final.view.parallel_plan.status).toBe('completed')
    for (const evidence of parallelEvidence(value)) {
      expect(evidence).toHaveLength(1)
      expect(evidence[0].verified).toBe(false)
      expect(evidence[0].text).toContain('synthetic-clone-child-1')
      expect(evidence[0].text).toContain('synthetic-clone-child-2')
    }
    for (const audit of value.audits.slice(1)) expect(audit.task_prompt).toContain('synthetic-clone-child-1')
  }, 65000)

  it('rejects stale revisions, mismatched digests and another workflow plan in the same session', () => {
    const value = invoke({ scenario: 'invalid_decision' })
    expectPaused(value)
    expect(value.rejected.map((response: { status: number }) => response.status)).toEqual([409, 409, 409])
    expect(value.after).toEqual(value.before)
    expect(value.foreign.session_id).toBe(value.initial.view.session_id)
    expect(value.foreign.parallel_plan).toMatchObject({
      status: 'pending',
      session_id: value.foreign.session_id,
      workflow_id: value.foreign.id,
    })
    expect(value.foreign_raw_scope).toBe('workflow-' + value.foreign.id)
  }, 65000)

  it('keeps the publication job running and heartbeatable while rejecting manual or incorrectly owned publication', () => {
    const value = invoke({ publication_probe: true })
    expectQueuedOnce(value, true)
    expectSequentialFinish(value)
    expect(value.publication.checks).toHaveLength(1)
    expect(value.publication.checks[0]).toMatchObject({
      job_id: value.continuation.id,
      job_status: 'running',
      step_statuses: Array(5).fill('completed'),
      step_job_statuses: ['succeeded', ...Array(4).fill('running')],
      manual_status: 409,
      manual_body: { detail: 'proposal job is not completed or belongs to another workflow' },
      wrong_owner: 'LeaseLost',
      wrong_job: 'Conflict',
      task_unchanged: true,
      counts_unchanged: true,
    })
    expect(value.publication.preparations).toHaveLength(1)
    expect(value.publication.preparations[0]).toMatchObject({
      job_id: value.continuation.id,
      before_status: 'running',
      after_status: 'running',
      heartbeat_cancelled: false,
      lease_renewed: true,
      workflow_state: 'waiting_approval',
      operation: { action: 'publish', status: 'started' },
    })
    expect(value.final.view.job.status).toBe('succeeded')
    expect(value.final.view.proposal.status).toBe('pending_approval')
  }, 65000)

  it('rolls back a failed decision submission without orphaning approval and retries the same approval', () => {
    const value = invoke({ scenario: 'decision_rollback' })
    expectPaused(value)
    expect(value.rollback.response).toMatchObject({
      status: 409,
      body: { detail: 'synthetic continuation submit failure' },
    })
    expect(value.rollback.submit_calls).toBe(1)
    expect(value.rollback.after).toEqual(value.initial)
    expect(value.rollback.plan_bytes_unchanged).toBe(true)
    expect(value.rollback.raw_plan).toMatchObject({
      status: 'pending',
      plan_id: value.initial.view.parallel_plan.plan_id,
      digest: value.initial.view.parallel_plan.digest,
      results: [],
    })
    expect(value.rollback.raw_plan).not.toHaveProperty('approval')
    expectQueuedOnce(value, true)
    expect(value.queued_raw_plan.status).toBe('pending')
    expect(value.queued_raw_plan).not.toHaveProperty('approval')
    expectSequentialFinish(value)
    expect(value.final.clones).toEqual(['child-1', 'child-2'])
    expect(value.raw_plan.status).toBe('completed')
    expect(value.raw_plan.approval.digest).toBe(value.initial.view.parallel_plan.digest)
  }, 65000)

  it.each([false, true])(
    'serializes concurrent approved=%s replay into one job and Outbox message',
    approved => {
      const value = invoke({ scenario: 'concurrent', approved })
      expect(value.decisions).toHaveLength(4)
      expectQueuedOnce(value, approved)
      expectSequentialFinish(value)
      expect(value.final.clones).toEqual(approved ? ['child-1', 'child-2'] : [])
    },
    65000,
  )

  it('rejects approval after source changes without scheduling a continuation', () => {
    const value = invoke({ scenario: 'source_before_decision' })
    expectPaused(value)
    expect(value.decision.status).toBe(409)
    expect(value.after).toMatchObject({
      counts: value.initial.counts,
      clones: [],
      executions: [],
      roles: ['main'],
      view: { state: 'waiting_parallel_approval', parallel_plan: { status: 'pending' } },
    })
  }, 65000)

  it('blocks a queued continuation when source changes after approval', () => {
    const value = invoke({ scenario: 'source_after_decision' })
    expectQueuedOnce(value, true)
    expect(value.execution.status).toBe('failed')
    expect(value.final).toMatchObject({ roles: ['main'], clones: [], executions: [], view: { state: 'blocked' } })
    expect(value.execution.result.reason).toMatch(/changed/i)
    expect(value.source_content).toBe('# external source edit while approval was pending\n')
    expect(value.raw_plan.results).toEqual([])
    expect(value.redelivery_claims).toEqual([null, null])
  }, 65000)

  it.each(['partial', 'all', 'execution'])(
    'continues sequentially after %s research failure without verified claims',
    failure => {
      const value = invoke({ failure })
      expectQueuedOnce(value, true)
      expectSequentialFinish(value)
      expectParallelStatus(value, failure === 'execution' ? 'unavailable' : 'partial')
      expect(value.final.view.parallel_plan.workflow_status).toBe(failure === 'execution' ? 'fallback' : 'partial')
      expect(value.final.view.parallel_plan.fallback_reason).toBeTruthy()
      expect(value.final.clones).toEqual(failure === 'execution' ? [] : ['child-1', 'child-2'])
      for (const evidence of parallelEvidence(value)) {
        if (failure === 'partial') {
          expect(evidence).toHaveLength(1)
          expect(evidence[0].verified).toBe(false)
          expect(evidence[0].text).toContain('synthetic-clone-child-1')
          expect(evidence[0].text).not.toContain('synthetic-clone-child-2')
          expect(JSON.parse(evidence[0].text).status).toBe('partial')
        } else expect(evidence).toEqual([])
      }
      expect(JSON.stringify(value)).not.toContain('private-provider-error-must-not-leak')
    },
    65000,
  )

  it.each([{ model: 'unapproved-model' }, { provider: 'unapproved-provider' }, { base_url: 'http://127.0.0.1:2/v1' }])(
    'does not launch clones with changed model configuration %j',
    settings_change => {
      const value = invoke({ scenario: 'model_changed', settings_change })
      expectQueuedOnce(value, true)
      expectSequentialFinish(value)
      expect(value.final.clones).toEqual([])
      expect(value.final.view.parallel_plan).toMatchObject({ workflow_status: 'fallback', results: [] })
      expect(value.final.view.parallel_plan.fallback_reason).toContain('PermissionError')
      expect(parallelEvidence(value)).toEqual([[], [], [], []])
    },
    65000,
  )

  it('preserves complete raw plan results while bounding evidence supplied to roles', () => {
    const value = invoke({ long_result: true })
    expectQueuedOnce(value, true)
    expectSequentialFinish(value)
    const sorted = (rows: { task_id: string }[]) => [...rows].sort((a, b) => a.task_id.localeCompare(b.task_id))
    expect(sorted(value.raw_plan.results)).toEqual(value.expected_results)
    expect(sorted(value.final.view.parallel_plan.results)).toEqual(value.expected_results)
    for (const evidence of parallelEvidence(value)) {
      expect(evidence).toHaveLength(1)
      expect(evidence[0].verified).toBe(false)
      expect(evidence[0].text.length).toBeLessThanOrEqual(8000)
      expect(evidence[0].text).not.toContain('raw-tail-')
      expect(JSON.parse(evidence[0].text).findings.map((finding: { truncated: boolean }) => finding.truncated)).toEqual(
        [true, true],
      )
    }
  }, 65000)

  it('includes SDK-shaped replica counters in workflow usage once across continuation and replay', () => {
    const value = invoke({ usage: 'sdk' })
    expectQueuedOnce(value, true)
    expectSequentialFinish(value)
    expect(value.initial_usage.totals).toMatchObject({ role_invocations: 1, total_tokens: 120 })
    expect(value.queued_usage.totals).toEqual(value.initial_usage.totals)
    expect(value.usage.totals).toMatchObject({
      role_invocations: 7,
      recorded_invocations: 7,
      observed_model_attempts: 7,
      model_attempt_count_complete: true,
      usage_status: 'complete',
      unknown_usage_count: 0,
      input_tokens: 522,
      output_tokens: 114,
      reasoning_tokens: 31,
      total_tokens: 636,
      known_tokens: 636,
      unsuccessful_invocations: 0,
    })
    expect(value.usage.by_role.replica).toMatchObject({
      role_invocations: 2,
      observed_model_attempts: 2,
      input_tokens: 22,
      output_tokens: 14,
      total_tokens: 36,
      usage_status: 'complete',
      unsuccessful_invocations: 0,
    })
    const replicas = value.usage.calls.filter((call: { role: string }) => call.role === 'replica')
    expect(replicas.map((call: { step_id: string }) => call.step_id).sort()).toEqual(['child-1', 'child-2'])
    for (const call of replicas)
      expect(call).toMatchObject({
        workflow_id: value.initial.view.id,
        job_id: value.continuation.id,
        phase: 'parallel_research',
        error: null,
        usage: { schema_version: 1, outcome: 'completed' },
      })
    expect(value.usage_after_replay).toEqual(value.usage)
  }, 65000)

  it('adds no replica usage for denial, approval replay or queue redelivery', () => {
    const value = invoke({ approved: false, usage: 'sdk' })
    expectQueuedOnce(value, false)
    expectSequentialFinish(value)
    expect(value.queued_usage.totals).toEqual(value.initial_usage.totals)
    expect(value.queued_usage.calls).toEqual(value.initial_usage.calls)
    expect(value.usage.totals).toMatchObject({
      role_invocations: 5,
      recorded_invocations: 5,
      observed_model_attempts: 5,
      usage_status: 'complete',
      unknown_usage_count: 0,
      input_tokens: 500,
      output_tokens: 100,
      total_tokens: 600,
    })
    expect(value.usage.by_role).not.toHaveProperty('replica')
    expect(value.usage.calls.map((call: { role: string }) => call.role)).toEqual(roles)
    expect(value.final.clones).toEqual([])
    expect(value.usage_after_replay).toEqual(value.usage)
  }, 65000)

  it('marks replica usage without schema_version unknown instead of counting malformed fake totals', () => {
    const value = invoke({ usage: 'malformed' })
    expectSequentialFinish(value)
    expect(value.usage.totals).toMatchObject({
      role_invocations: 7,
      recorded_invocations: 5,
      model_attempt_count_complete: false,
      usage_status: 'partial',
      unknown_usage_count: 2,
      total_tokens: null,
      input_tokens: null,
      known_tokens: 600,
    })
    expect(value.usage.by_role.replica).toMatchObject({
      role_invocations: 2,
      recorded_invocations: 0,
      usage_status: 'unknown',
      unknown_usage_count: 2,
      total_tokens: null,
      input_tokens: null,
      output_tokens: null,
      known_tokens: null,
    })
    const replicas = value.usage.calls.filter((call: { role: string }) => call.role === 'replica')
    expect(replicas).toHaveLength(2)
    for (const call of replicas) expect(call).toMatchObject({ usage: null, error: 'usage_not_recorded' })
    expect(value.usage_after_replay).toEqual(value.usage)
  }, 65000)

  it.each(['queued', 'running'])(
    'honors cancellation of a %s continuation without further roles or clones',
    at => {
      const value = invoke({ scenario: 'cancel', at })
      expectQueuedOnce(value, true)
      expect(value.final).toMatchObject({
        roles: ['main'],
        clones: [],
        files_exist: [false, false],
        counts: { jobs: 2, outbox: 2, runs: 2 },
        view: { job: { cancel_requested: 1 } },
      })
      if (at === 'queued') {
        expect(value.cancelled_claim).toBeNull()
        expect(value.final.view.job.status).toBe('cancelled')
        expect(value.final.executions).toEqual([])
        expect(value.raw_plan.status).toBe('pending')
      } else {
        expect(value.execution_error).toBe('LeaseLost')
        expect(value.final.view.job.status).toBe('running')
        expect(value.final.executions).toHaveLength(1)
        expect(value.final.executions[0].has_stop).toBe(true)
        expect(value.raw_plan.status).toBe('cancelled')
        expect(value.raw_plan.results).toHaveLength(2)
        for (const row of value.raw_plan.results) expect(row.status).toBe('cancelled')
      }
      expect(value.redelivery_claims).toEqual([null, null])
      expect(value.replay.status).toBe(200)
      expect(value.replay.body.job.id).toBe(value.continuation.id)
      expect(value.after_replay).toEqual(value.final)
    },
    65000,
  )
})
