import { describe, expect, it } from 'vitest'
import { probe } from '../probe'

const driver = `
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dsh_multi_agent import IsolatedHarnessBackend, Settings
data = json.load(sys.stdin)
calls, notifications = [], []
class Runtime:
    def __init__(self, **kwargs):
        self.record = dict(kwargs=kwargs, empty=list(Path(kwargs["cwd"]).iterdir()) == [], closed=False,
            profile=json.loads((Path(kwargs["dsh_home"]) / "profiles" / kwargs["profile"] / "cordis.patch.yml").read_text()),
            manifest=json.loads((Path(kwargs["dsh_home"]) / "profiles" / kwargs["profile"] / "package.json").read_text()))
        calls.append(self.record)
        if data.get("constructor_failure"):
            raise RuntimeError("fixture startup")
    def run(self, prompt, *, session_id, on_notification):
        on_notification(SimpleNamespace(method="session.event", payload={"sessionId":session_id,"event":{}}))
        if data.get("failure"):
            raise TimeoutError("fixture")
        return SimpleNamespace(session_id=session_id, final_response='{"category":"question"}', finish_reason="completed", events=[])
    def close(self):
        self.record["closed"] = True
settings = Settings("fixture", "exact-model", "http://127.0.0.1:1/v1", "FIXTURE_API_KEY", "fixture", 123)
failure = None
with patch("deepseek_harness.DeepSeekHarness", Runtime):
    try:
        backend = IsolatedHarnessBackend(settings)
        backend.run("Classify mail", "First message", session_id="a", on_event=notifications.append)
        backend.run("Classify mail", "Second message", session_id="b")
    except Exception as error:
        failure = type(error).__name__
for call in calls:
    call["removed"] = not Path(call["kwargs"]["cwd"]).exists() and not Path(call["kwargs"]["dsh_home"]).exists()
print(json.dumps(dict(calls=calls, failure=failure, notifications=notifications)))
`

describe('isolated SDK backend', () => {
  it('creates fresh cwd/profile/session storage without parent tools and preserves model settings', () => {
    const result = probe(driver)
    expect(result.calls).toHaveLength(2)
    expect(result.calls[0].kwargs.cwd).not.toBe(result.calls[1].kwargs.cwd)
    expect(result.notifications).toHaveLength(1)
    for (const call of result.calls) {
      expect(call).toMatchObject({ empty: true, closed: true, removed: true })
      expect(call.kwargs).toMatchObject({ provider: 'fixture', model: 'exact-model', request_timeout_seconds: 123 })
      expect(call.kwargs.runtime_cwd).toBe(call.kwargs.cwd)
      expect(call.kwargs).not.toHaveProperty('temperature')
      expect(call.kwargs).not.toHaveProperty('max_tokens')
      expect(call.manifest.dsh.profile).toEqual({ bundles: [], patchReload: 'startup' })
      const entries = call.profile[0].insert
      expect(entries.some((entry: any) => /mcp|bash|fs-local|subprocess|subagent|tool-subagent/.test(entry.name))).toBe(
        false,
      )
      expect(entries.find((entry: any) => entry.id === 'system-prompt').config).toEqual({
        includeHarnessIdentity: false,
        includeRuntimeContext: false,
      })
      expect(entries.find((entry: any) => entry.id === 'agent-loop').config.agents).toEqual([])
      expect(JSON.stringify(entries)).not.toContain('FIXTURE_API_KEY":"fixture')
    }
  })
  it('closes the SDK and removes all temporary storage after failure', () => {
    const result = probe(driver, { failure: true })
    expect(result.failure).toBe('TimeoutError')
    expect(result.calls).toHaveLength(1)
    expect(result.calls[0]).toMatchObject({ closed: true, removed: true })
  })
  it('removes the home even when SDK construction fails', () => {
    const result = probe(driver, { constructor_failure: true })
    expect(result.failure).toBe('RuntimeError')
    expect(result.calls[0].removed).toBe(true)
  })
})
