import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

it('distinguishes provider rate limits from invalid role output without exposing provider text or retrying', () => {
  const driver = `
import json
from types import SimpleNamespace
from codex_agent.orchestration.coordinator import ProposalCoordinator as C
value = SimpleNamespace(outcome="incomplete", harness=SimpleNamespace(events=[{
    "type":"turn/end", "data":{"reason":{"kind":"error","error":{
        "code":"RATE_LIMIT", "message":"private credential and provider request details"}}}}]))
summary = C._rejected_call_summary(value)
task = {"steps":[{"result":{"summary":summary}}]}
value.harness.events[0]["data"]["reason"]["error"]["code"] = "PRIVATE_UNKNOWN_CODE"
unknown = C._rejected_call_summary(value)
value.harness.events = [None, {"type":"turn/end", "data":{"reason":"incomplete"}},
    {"type":"turn/end", "data":{"reason":{"error":{"code":[]}}}}]
malformed = C._rejected_call_summary(value)
value.outcome = "invalid_output"
print(json.dumps({"summary":summary,"blocked":C._blocked_reason(None,"Semantic output rejected.",task),
    "unknown":unknown,"malformed":malformed,"invalid":C._rejected_call_summary(value)}))
`
  const result = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url)),
    ['-I', '-B', '-c', driver],
    { encoding: 'utf8', timeout: 5000 },
  )
  expect(result.status, result.stderr).toBe(0)
  const value = JSON.parse(result.stdout)
  expect(value.summary).toContain('RATE_LIMIT')
  expect(value.summary).toContain('no automatic workflow replay')
  expect(value.blocked).toBe(value.summary)
  expect(value.unknown).toContain('did not complete')
  expect(value.malformed).toContain('did not complete')
  expect(value.invalid).toBe('Role output rejected')
  expect(result.stdout).not.toContain('private credential')
  expect(result.stdout).not.toContain('PRIVATE_UNKNOWN_CODE')
})
