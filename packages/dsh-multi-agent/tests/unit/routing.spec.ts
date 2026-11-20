import { expect, it } from 'vitest'
import { probe } from '../probe'

const driver = `
import json, sys
from dsh_multi_agent.routing import RoutingPolicy
data = json.load(sys.stdin)
try:
    result = RoutingPolicy(**data.get("policy", {})).select(provider="fixture", default_model="current",
        input_bytes=data.get("bytes", 100), low_risk=data.get("low", False), reason="host_facts")
except (ValueError, TypeError):
    result = {"rejected":True}
print(json.dumps(result))
`

it('keeps the current model when disabled, even with other models configured', () => {
  expect(probe(driver, { policy: { fastModel: 'fast', strongModel: 'strong' }, low: true })).toMatchObject({
    tier: 'default',
    model: 'current',
    reason: 'routing_disabled',
  })
})
it('routes bounded low-risk facts, unknown tasks and oversized inputs deterministically', () => {
  const policy = { enabled: true, fastModel: 'fast', strongModel: 'strong' }
  expect(probe(driver, { policy, low: true })).toMatchObject({ tier: 'fast', model: 'fast', timeout_seconds: 120 })
  expect(probe(driver, { policy })).toMatchObject({ tier: 'strong', model: 'strong', timeout_seconds: 300 })
  expect(probe(driver, { policy, low: true, bytes: 12001 })).toMatchObject({
    tier: 'strong',
    reason: 'input_exceeds_fast_budget',
  })
  expect(probe(driver, { policy: { enabled: true }, low: true })).toMatchObject({
    model: 'current',
    reason: 'fast_model_not_configured',
  })
})
it('fingerprints model IDs, time limits and input policy without including credentials', () => {
  const one = probe(driver, { policy: { enabled: true } })
  const two = probe(driver, { policy: { enabled: true, strongTimeoutSeconds: 250 } })
  expect(one.policy_fingerprint).toMatch(/^[a-f0-9]{64}$/)
  expect(one.policy_fingerprint).not.toBe(two.policy_fingerprint)
})
it.each([
  { enabled: 'yes' },
  { fastTimeoutSeconds: 301 },
  { strongTimeoutSeconds: 29 },
  { fastMaxInputBytes: true },
  { fastModel: 'bad\nmodel' },
  { strongModel: ' spaced ' },
  { unknown: 1 },
])('rejects invalid host configuration: %j', policy => {
  expect(probe(driver, { policy })).toEqual({ rejected: true })
})
