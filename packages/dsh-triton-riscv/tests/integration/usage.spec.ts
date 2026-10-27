import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python')
// Vitest owns all assertions. Python is only a bridge to the production backend.
const driver = `
import dataclasses, json, sys
from types import SimpleNamespace
from unittest.mock import patch
from codex_agent.orchestration.usage import RoleUsage, normalize_usage, summarize
from codex_agent.orchestration.runtime import RoleAgent
from codex_agent.harness.runtime import HarnessRunResult
from codex_agent.orchestration.usage_report import workflow_usage, aggregate_calls
data = json.load(sys.stdin)
operation = data.pop("operation")
if operation == "normalize":
    result = normalize_usage(data["usage"])
elif operation == "summarize":
    result = summarize([normalize_usage(v) for v in data["usages"]])
elif operation == "meter":
    meter = RoleUsage("owned", provider="fixture", model="fixture-model")
    for event in data["events"]:
        meter.observe(event)
    result = meter.finish("validated")
elif operation == "runtime":
    records, received = [], []
    class Backend:
        settings = SimpleNamespace(provider="fixture", model="fixture-model")
        def run(self, policy, prompt, *, session_id, on_event=None):
            for event in data["events"]:
                on_event({"method": "session.event", "payload": {"sessionId": session_id, "event": event}})
            if data.get("raises"):
                raise TimeoutError("fixture timeout, never a model API call")
            return HarnessRunResult(session_id, data.get("response", json.dumps(dict(
                status="completed", summary="fixture only", decision="plan_tests"))),
                "completed", data["events"])
    request = dict(task_id="task", step_id="step", input_epoch=1, role="main", phase="preparing",
                   kind="create", operator="fixture", requirement="fixture", versions={})
    try:
        run = RoleAgent(backend=Backend()).run(request, on_event=received.append, on_usage=records.append)
        result = dict(outcome=run.outcome, usage=run.usage)
    except TimeoutError:
        result = dict(outcome="timeout_preserved")
    result.update(records=records, received=len(received))
elif operation == "sdk":
    from codex_agent.tests.test_agent_roles import ProposalBackendTests
    records = []
    original = RoleAgent.run
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        samples = [e["data"]["usage"] for e in value.harness.events
                   if e.get("type") == "assistant/message" and e.get("data", {}).get("usage")]
        records.append(dict(measured=value.usage, sdk_samples=samples))
        return value
    fixture = ProposalBackendTests("test_real_sdk_roundtrip_to_loopback_responses_fixture")
    with patch.object(RoleAgent, "run", capture):
        fixture.test_real_sdk_roundtrip_to_loopback_responses_fixture()
    result = records
elif operation == "durable":
    from codex_agent.tests.test_orchestration_worker import WorkflowWorkerTests
    from codex_agent.artifacts import ArtifactStore
    from codex_agent.tasks.schema import artifacts
    from codex_agent.orchestration.api import router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    fixture = WorkflowWorkerTests("test_create_full_worker_route_stops_at_approval_without_writes_or_validation")
    fixture.setUp()
    try:
        original = fixture.backend.run
        def run(*args, **kwargs):
            for event in data["events"]:
                kwargs["on_event"]({"method": "session.event", "payload": {"sessionId": kwargs["session_id"], "event": event}})
            if data.get("raises"):
                raise TimeoutError("fixture")
            value = original(*args, **kwargs)
            return dataclasses.replace(value, events=data["events"])
        fixture.backend.run = run
        fixture.backend.settings = SimpleNamespace(provider="fixture", model="fixture-model")
        job = fixture.submit()
        executed = fixture.execute(job)
        task = fixture.task(job)
        report = workflow_usage(fixture.jobs.db, task["id"])
        again = workflow_usage(fixture.jobs.db, task["id"])
        app = FastAPI()
        app.include_router(router(fixture.root))
        with TestClient(app) as client:
            response = client.get("/api/operator-workflows/" + task["id"] + "/usage")
            missing_status = client.get("/api/operator-workflows/nonexistent/usage").status_code
        result = dict(report=report, repeat_equal=again == report, state=task["state"], job_state=executed["status"],
                      endpoint_status=response.status_code, endpoint=response.json(),
                      missing_status=missing_status,
                      files_written=(fixture.root / "python").exists(),
                      model_roles=[call["role"] for call in fixture.backend.calls])
        records = fixture.jobs.db.rows(artifacts, artifacts.c.name != "result.json")
        audit = json.loads(ArtifactStore(fixture.root).read(records[0]))
        result["artifact_usage"] = audit.get("usage")
        with patch.object(ArtifactStore, "read", side_effect=OSError("fixture corrupt artifact")):
            result["corrupt"] = workflow_usage(fixture.jobs.db, task["id"])
        with patch.object(ArtifactStore, "read", return_value=json.dumps({**audit, "usage": None}).encode()):
            result["legacy"] = workflow_usage(fixture.jobs.db, task["id"])
        with patch.object(ArtifactStore, "read", return_value=json.dumps({**audit, "usage": {"schema_version": 1}}).encode()):
            result["malformed"] = workflow_usage(fixture.jobs.db, task["id"])
    finally:
        fixture.doCleanups()
elif operation == "chain":
    from codex_agent.orchestration.store import WorkflowStore
    from codex_agent.tasks.store import Conflict
    tasks = data["tasks"]
    with patch.object(WorkflowStore, "get", side_effect=lambda identifier: tasks[identifier]), \\
         patch("codex_agent.orchestration.usage_report.workflow_calls", side_effect=lambda db, task: task["calls"]):
        try:
            result = workflow_usage(SimpleNamespace(root="/unused"), data["id"])
        except Conflict:
            result = {"rejected": True}
print(json.dumps(result))
`

function invoke(operation: string, input: object) {
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) => !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) && !/^(TRITON_RISCV_CONFIG|DSH_|RISCV_)/.test(key),
    ),
  )
  const result = spawnSync(
    python,
    [
      '-I',
      '-B',
      '-c',
      'import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))',
      join(root, 'python'),
      driver,
    ],
    {
      input: JSON.stringify({ operation, ...input }),
      encoding: 'utf8',
      timeout: 30000,
      maxBuffer: 8 * 1024 * 1024,
      env,
    },
  )
  expect(result.error).toBeUndefined()
  expect(result.status, result.stdout + result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

const usage = { inputTokens: 100, cacheReadTokens: 40, cacheWriteTokens: 10, outputTokens: 20, reasoningTokens: 5 }
const event = (seq: number, type: string, extra = {}, step = 1) => ({ seq, type, data: { turn: 1, step, ...extra } })
const start = event(1, 'step/start')
const chunk = event(2, 'assistant/chunk', { chunk: { type: 'usage', usage } })
const message = event(3, 'assistant/message', { usage })
const end = event(4, 'step/end')
const events = [start, chunk, message, end]

describe('role usage accounting', () => {
  it('adds disjoint cached input once and never adds reasoning twice', () => {
    expect(invoke('normalize', { usage })).toMatchObject({
      input_tokens: 150,
      total_tokens: 170,
      output_tokens: 20,
      reasoning_tokens: 5,
      usage_status: 'complete',
    })
  })
  it('keeps absent cache counts and exact totals unknown rather than zero', () => {
    expect(invoke('normalize', { usage: { inputTokens: 3, outputTokens: 20 } })).toMatchObject({
      cache_read_tokens: null,
      cache_write_tokens: null,
      total_tokens: null,
      known_tokens: 23,
      usage_status: 'partial',
    })
  })
  it('uses an authoritative total without inventing its cache breakdown', () => {
    expect(invoke('normalize', { usage: { inputTokens: 3, outputTokens: 20, totalTokens: 170 } })).toMatchObject({
      input_tokens: 150,
      total_tokens: 170,
      cache_read_tokens: null,
      usage_status: 'complete',
    })
  })
  it.each([-1, '30', true, 1.5])('rejects invalid usage counter %s', outputTokens => {
    expect(invoke('normalize', { usage: { ...usage, outputTokens } })).toMatchObject({
      total_tokens: null,
      usage_status: 'invalid',
    })
  })
  it('rejects contradictory totals and reasoning larger than output', () => {
    for (const value of [
      { ...usage, totalTokens: 999 },
      { ...usage, reasoningTokens: 30 },
    ])
      expect(invoke('normalize', { usage: value }).usage_status).toBe('invalid')
  })
  it('does not infer missing usage from prompt length', () => {
    const result = invoke('runtime', { events: [] })
    expect(result.usage.prompt_characters).toBeGreaterThan(100)
    expect(result.usage.totals).toMatchObject({ total_tokens: null, known_tokens: null, usage_status: 'unknown' })
    expect(result.usage.cost).toBeNull()
  })
  it('deduplicates streamed usage, final messages and replayed result events', () => {
    const result = invoke('meter', { events: [...events, ...events] })
    expect(result.attempt_count).toBe(1)
    expect(result.totals.total_tokens).toBe(170)
  })
  it('uses the final cumulative snapshot, not the sum of intermediate snapshots', () => {
    const result = invoke('meter', {
      events: [
        start,
        event(2, 'assistant/chunk', { chunk: { type: 'usage', usage: { ...usage, outputTokens: 1 } } }),
        message,
        end,
      ],
    })
    expect(result.totals.total_tokens).toBe(170)
  })
  it('counts both billed failure and successful retry within the same step', () => {
    const result = invoke('meter', {
      events: [
        start,
        chunk,
        event(3, 'llm/retry'),
        event(4, 'llm/retry-started'),
        event(5, 'assistant/message', { usage }),
        event(6, 'step/end'),
      ],
    })
    expect(result.attempt_count).toBe(2)
    expect(result.attempts[0].outcome).toBe('failed_attempt')
    expect(result.totals.total_tokens).toBe(340)
  })
  it('counts multiple steps and supports durable packed stream usage', () => {
    const result = invoke('meter', {
      events: [
        ...events,
        event(5, 'step/start', {}, 2),
        event(6, 'assistant/message', { stream: [{ type: 'chunk', time: 0, chunk: { type: 'usage', usage } }] }, 2),
        event(7, 'step/end', {}, 2),
      ],
    })
    expect(result.totals.total_tokens).toBe(340)
  })
  it('preserves known costs but makes the total unknown when a failed attempt lacks usage', () => {
    const result = invoke('meter', {
      events: [
        start,
        event(2, 'llm/retry'),
        event(3, 'llm/retry-started'),
        event(4, 'assistant/message', { usage }),
        event(5, 'step/end'),
      ],
    })
    expect(result.totals).toMatchObject({ total_tokens: null, known_tokens: 170, unknown_usage_count: 1 })
  })
  it('ignores foreign session events and unrelated fixture counters', () => {
    const result = invoke('meter', {
      events: [
        ...events,
        { method: 'session.event', payload: { sessionId: 'other', event: event(10, 'assistant/message', { usage }) } },
        { type: 'fixture_usage', input_tokens: 999 },
      ],
    })
    expect(result.attempt_count).toBe(1)
    expect(result.totals.total_tokens).toBe(170)
  })
  it('discloses conflicting identities and missing attempt boundaries', () => {
    for (const value of [
      [message],
      [...events, event(3, 'assistant/message', { usage: { ...usage, outputTokens: 99 } })],
    ]) {
      const result = invoke('meter', { events: value })
      expect(result.issues.length).toBeGreaterThan(0)
      expect(result.totals.total_tokens).toBeNull()
    }
  })
  it('does not certify totals when a retry or second response lacks a boundary', () => {
    for (const value of [
      [event(1, 'llm/retry-started'), message, end],
      [...events, event(5, 'assistant/message', { usage })],
    ]) {
      const result = invoke('meter', { events: value })
      expect(result.issues.length).toBeGreaterThan(0)
      expect(result.totals.total_tokens).toBeNull()
    }
  })
  it('retains usage for invalid model JSON without retrying or accepting it', () => {
    const result = invoke('runtime', { events, response: 'not JSON' })
    expect(result.outcome).toBe('invalid_output')
    expect(result.records).toHaveLength(1)
    expect(result.usage.totals.total_tokens).toBe(170)
    expect(result.received).toBe(4)
  })
  it('reports partial usage on timeout and preserves the original exception', () => {
    const result = invoke('runtime', { events: [start, chunk], raises: true })
    expect(result.outcome).toBe('timeout_preserved')
    expect(result.records).toHaveLength(1)
    expect(result.records[0]).toMatchObject({ outcome: 'error', totals: { total_tokens: null, known_tokens: 170 } })
    expect(result.records[0].elapsed_ms).toBeGreaterThanOrEqual(0)
  })
  it('records provider, model and timings without changing existing notifications', () => {
    const result = invoke('runtime', { events })
    expect(result.usage).toMatchObject({
      model: 'fixture-model',
      provider: 'fixture',
      role_status: 'completed',
      totals: { total_tokens: 170 },
    })
    expect(result.received).toBe(4)
    expect(result.records).toHaveLength(1)
  })
  it('makes aggregate totals unknown if any call is unknown', () => {
    expect(invoke('summarize', { usages: [usage, {}] })).toMatchObject({
      total_tokens: null,
      known_tokens: 170,
      unknown_usage_count: 1,
    })
  })
})

describe('repair chain usage', () => {
  function fixture() {
    const measured = invoke('meter', { events })
    measured.role_status = 'completed'
    const task = (id: string, result: object, state: string) => ({
      id,
      session_id: 'session',
      operator: 'fixture',
      state,
      created_ms: 1,
      updated_ms: 2,
      result,
      calls: [{ role: 'implementer', usage: measured }],
    })
    return {
      root: task('root', { lifecycle: { child_workflow_id: 'repair', validation_plan_id: 'failed-test' } }, 'failed'),
      repair: task(
        'repair',
        { parent_workflow_id: 'root', lifecycle: { validation_plan_id: 'passing-test' } },
        'succeeded',
      ),
    }
  }
  it('includes the failed parent when queried from a successful repair', () => {
    const result = invoke('chain', { tasks: fixture(), id: 'repair' })
    expect(result.root_workflow_id).toBe('root')
    expect(result.totals).toMatchObject({ total_tokens: 340, role_invocations: 2, observed_model_attempts: 2 })
    expect(result.workflows.map((task: { state: string }) => task.state)).toEqual(['failed', 'succeeded'])
    expect(result.workflows.map((task: { validation_plan_id: string }) => task.validation_plan_id)).toEqual([
      'failed-test',
      'passing-test',
    ])
  })
  it('returns the same totals from either end without duplicated calls', () => {
    const tasks = fixture()
    expect(invoke('chain', { tasks, id: 'root' }).totals).toEqual(invoke('chain', { tasks, id: 'repair' }).totals)
  })
  it.each(['session_id', 'operator'])('rejects a repair bound to another %s', field => {
    const tasks = fixture()
    Object.assign(tasks.repair, { [field]: 'foreign' })
    expect(invoke('chain', { tasks, id: 'root' })).toEqual({ rejected: true })
  })
  it('rejects non-reciprocal links and cycles', () => {
    const tasks = fixture()
    tasks.repair.result = {}
    expect(invoke('chain', { tasks, id: 'root' })).toEqual({ rejected: true })
    tasks.root.result = { parent_workflow_id: 'repair', lifecycle: { child_workflow_id: 'repair' } }
    tasks.repair.result = { parent_workflow_id: 'root', lifecycle: { child_workflow_id: 'root' } }
    expect(invoke('chain', { tasks, id: 'root' })).toEqual({ rejected: true })
  })
})

describe('durable workflow usage', () => {
  it('reads real installed Harness SDK events through a loopback-only Responses fixture', () => {
    const result = invoke('sdk', {})
    console.info('Loopback fixture usage (not business token savings):', JSON.stringify(result))
    expect(result).toHaveLength(1)
    expect(result[0].measured).toMatchObject({
      outcome: 'validated',
      attempt_count: 1,
      issues: [],
      totals: {
        uncached_input_tokens: 32,
        output_tokens: 16,
        known_tokens: 48,
      },
    })
    expect(result[0].sdk_samples).toHaveLength(1)
    const sample = result[0].sdk_samples[0]
    // Older bundled runtimes discard total/cache fields even when the endpoint
    // returned them. Test the observable SDK contract, never fill missing fields.
    const complete =
      sample.totalTokens !== undefined ||
      (sample.cacheReadTokens !== undefined && sample.cacheWriteTokens !== undefined)
    expect(result[0].measured.totals.total_tokens).toBe(complete ? 48 : null)
    expect(result[0].measured.totals.usage_status).toBe(complete ? 'complete' : 'partial')
  }, 40000)
  it('persists five role calls and serves an idempotent read-only API without granting execution', () => {
    expect(process.env.TRITON_MYSQL_URL, 'Use a disposable MySQL database').toBeTruthy()
    const result = invoke('durable', { events })
    expect(result.state).toBe('waiting_approval')
    expect(result.model_roles).toEqual(['main', 'semantic', 'reviewer', 'implementer', 'reviewer'])
    expect(result.files_written).toBe(false)
    expect(result.repeat_equal).toBe(true)
    expect(result.endpoint_status).toBe(200)
    expect(result.missing_status).toBe(404)
    expect(result.endpoint).toEqual(result.report)
    expect(result.report.totals).toMatchObject({
      role_invocations: 5,
      observed_model_attempts: 5,
      total_tokens: 850,
      cost: null,
    })
    expect(result.report.by_role.reviewer.total_tokens).toBe(340)
    expect(result.artifact_usage.totals.total_tokens).toBe(170)
    expect(result.corrupt.totals.total_tokens).toBeNull()
    expect(result.legacy.totals.total_tokens).toBeNull()
    expect(result.malformed.totals.total_tokens).toBeNull()
  }, 40000)
  it('includes failed calls whose steps remain fenced for reconciliation', () => {
    expect(process.env.TRITON_MYSQL_URL, 'Use a disposable MySQL database').toBeTruthy()
    const result = invoke('durable', { events: [start, chunk], raises: true })
    expect(result.state).not.toBe('succeeded')
    expect(result.report.totals).toMatchObject({
      role_invocations: 1,
      unsuccessful_invocations: 1,
      total_tokens: null,
      known_tokens: 170,
    })
    expect(result.artifact_usage.outcome).toBe('error')
    expect(result.report.calls[0].step_status).toBe('running')
  }, 40000)
})
