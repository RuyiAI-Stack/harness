import { describe, expect, it } from 'vitest'
import { probe } from '../probe'

const driver = `
import json, sys
from dataclasses import asdict
from types import SimpleNamespace
from dsh_multi_agent import HarnessRunResult, TaskDefinition, TaskRunner, json_result, RunCancelled
data = json.load(sys.stdin)
calls, reports, received = [], [], []
schema = {"type":"object", "required":["category"], "additionalProperties":False,
          "properties":{"category":{"enum":["question", "feedback"]}}}
class Backend:
    settings = SimpleNamespace(provider="fixture", model="classification-model")
    def run(self, policy, prompt, *, session_id, on_event=None):
        calls.append(dict(policy=policy, prompt=prompt, session_id=session_id))
        for event in data.get("events", []):
            on_event(dict(method="session.event", payload=dict(sessionId=session_id, event=event)))
        if data.get("failure"):
            raise RunCancelled() if data["failure"] == "cancelled" else TimeoutError("fixture")
        return HarnessRunResult("foreign" if data.get("foreign") else session_id,
            "x" * data["response_size"] if data.get("response_size") else data.get("response", '{"category":"question"}'),
            data.get("finish", "completed"), data.get("events", []))
runner = TaskRunner(Backend())
task = TaskDefinition("Classify mail into a category.", '{"message":"When will it arrive?"}', json_result(schema))
try:
    result = asdict(runner.run(task, on_usage=reports.append, on_event=received.append))
    if data.get("twice"):
        runner.run(task)
except Exception as error:
    result = dict(error_type=type(error).__name__)
if data.get("response_size"):
    result["raw_response_bytes"] = len(result["harness"].pop("final_response").encode())
print(json.dumps(dict(result=result, calls=calls, reports=reports, received=len(received),
    domain_imported=any(name == "codex_agent" or name.startswith("codex_agent.") for name in sys.modules))))
`

const usage = { inputTokens: 100, cacheReadTokens: 40, cacheWriteTokens: 10, outputTokens: 20, reasoningTokens: 5 }
const event = (seq: number, type: string, extra = {}) => ({ seq, type, data: { turn: 1, step: 1, ...extra } })
const events = [
  event(1, 'step/start'),
  event(2, 'assistant/chunk', { chunk: { type: 'usage', usage } }),
  event(3, 'assistant/message', { usage }),
  event(4, 'step/end'),
]

describe('generic typed execution', () => {
  it('runs non-operator mail classification without any domain package import', () => {
    const result = probe(driver, { twice: true, events })
    expect(result.domain_imported).toBe(false)
    expect(result.result).toMatchObject({ outcome: 'validated', result: { category: 'question' }, error: null })
    expect(result.calls).toHaveLength(2)
    expect(result.calls[0].session_id).not.toBe(result.calls[1].session_id)
    expect(result.result.usage).toMatchObject({
      provider: 'fixture',
      model: 'classification-model',
      attempt_count: 1,
      totals: { total_tokens: 170, input_tokens: 150, reasoning_tokens: 5 },
      cost: null,
    })
    expect(result.received).toBe(events.length)
    expect(result.reports).toHaveLength(1)
  })
  it.each([
    '',
    '[]',
    'null',
    '```json\n{}\n```',
    '{"category":"unknown"}',
    '{"category":"question","approved":true}',
    '{"category":"feedback","category":"question"}',
    '{"category":NaN}',
  ])('rejects untyped or malformed output %# without retry', response => {
    const result = probe(driver, { response, events })
    expect(result.result).toMatchObject({
      outcome: 'invalid_output',
      result: null,
      harness: { final_response: response },
    })
    expect(result.calls).toHaveLength(1)
    expect(result.result.usage.totals.total_tokens).toBe(170)
  })
  it('rejects oversized output and retains its complete raw response for audit', () => {
    const value = probe(driver, { response_size: 65537, events })
    expect(value.result).toMatchObject({ outcome: 'invalid_output', result: null, raw_response_bytes: 65537 })
    expect(value.calls).toHaveLength(1)
    expect(value.result.usage.totals.total_tokens).toBe(170)
  })
  it.each([{ foreign: true }, { finish: 'max-tokens' }, { finish: null }])(
    'rejects incomplete/foreign results: %j',
    data => {
      const result = probe(driver, { ...data, events })
      expect(result.result).toMatchObject({ outcome: 'incomplete', result: null })
      if (data.foreign) {
        expect(result.result.usage.issues).toContain('foreign_result_session')
        expect(result.result.usage.totals.total_tokens).toBeNull()
      }
    },
  )
  it.each(['error', 'cancelled'])('preserves observed usage and propagates %s without retries', failure => {
    const result = probe(driver, { failure, events })
    expect(result.result.error_type).toBe(failure === 'error' ? 'TimeoutError' : 'RunCancelled')
    expect(result.reports).toHaveLength(1)
    expect(result.reports[0]).toMatchObject({ outcome: failure, totals: { total_tokens: 170 } })
    expect(result.calls).toHaveLength(1)
  })
  it('never invents absent usage', () => {
    const result = probe(driver)
    expect(result.result.usage.totals).toMatchObject({
      total_tokens: null,
      known_tokens: null,
      usage_status: 'unknown',
    })
  })
  it('counts billed retries once and keeps unknown attempts unknown', () => {
    const retry = [
      event(1, 'step/start'),
      event(2, 'assistant/attempt', { usage }),
      event(3, 'llm/retry'),
      event(4, 'llm/retry-started'),
      event(5, 'assistant/message', { usage }),
      event(6, 'step/end'),
    ]
    expect(probe(driver, { events: retry }).result.usage.totals.total_tokens).toBe(340)
    const unknown = retry.map(item => (item.type === 'assistant/attempt' ? event(2, 'assistant/attempt') : item))
    expect(probe(driver, { events: unknown }).result.usage.totals).toMatchObject({
      total_tokens: null,
      known_tokens: 170,
    })
  })
  it('disables external schema retrieval and supports local schema references', () => {
    const value = probe(`
import json
from dsh_multi_agent import json_result
local = json_result({"$defs":{"category":{"enum":["question"]}}, "$ref":"#/$defs/category"})('"question"')
try:
    json_result({"$ref":"http://127.0.0.1:1/never-fetch"})('{}')
    rejected = False
except Exception:
    rejected = True
print(json.dumps(dict(local=local, rejected=rejected)))
`)
    expect(value).toEqual({ local: 'question', rejected: true })
  })
  it('rejects non-finite exponent overflow even when the schema accepts numbers', () => {
    const value = probe(`
import json
from dsh_multi_agent import json_result
try:
    json_result({"type":"number"})('1e999')
    rejected = False
except ValueError:
    rejected = True
print(json.dumps(dict(rejected=rejected)))
`)
    expect(value.rejected).toBe(true)
  })
})
