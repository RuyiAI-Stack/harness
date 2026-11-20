import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { resolveConfig, workerEnvironment } from '../../lib/config.js'

const python =
  process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url))
const driver = `
import json, os, sys, tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.delegation import Plans, host_scope, base_settings
from codex_agent.harness.model_routing import select_route
from codex_agent.harness.runtime import HarnessAgent, HarnessRunResult
from codex_agent.orchestration.runtime import RoleAgent
data = json.load(sys.stdin)
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    config = dict(schemaVersion=1,repoRoot=folder,stateDir=folder,modelRouting=data.get("policy",{}))
    os.environ["TRITON_RISCV_CONFIG"] = json.dumps(config)
    settings = HarnessSettings.from_env(root, {"ISRC_API_KEY":"fixture", "DSH_MODEL":"current"})
    out = {"routes":[select_route(settings, value["request"], value.get("prompt", value["request"]), role=value.get("role")) for value in data.get("cases",[])]}
    calls, events = [], []
    class Backend:
        def __init__(self, settings): self.settings = settings
        def run(self, *args, session_id=None, on_event=None):
            calls.append({"model":self.settings.model,"provider":self.settings.provider,"timeout":self.settings.request_timeout_seconds})
            if data.get("timeout"): raise TimeoutError("fixture")
            return HarnessRunResult(session_id or "fixture-s", "{}", "completed")
        def close(self): pass
    with patch("codex_agent.harness.runtime.DeepSeekHarnessBackend", Backend):
        agent = HarnessAgent(settings)
        for text in ["List available tests", "fix the kernel"]:
            try:
                with host_scope("s", "r", text): agent.run("Assembled context", on_event=events.append)
            except TimeoutError: pass
        agent.close()
    with patch("codex_agent.orchestration.runtime.ProposalHarnessBackend", Backend):
        role = RoleAgent(settings)
        reports = []
        try:
            role.run(dict(task_id="t",step_id="s",input_epoch=1,role="semantic",phase="preparing",
                kind="create",operator="fixture",requirement="Extract the supplied contract fields",versions={}),
                on_event=events.append,on_usage=reports.append)
        except TimeoutError: out["role_timeout"] = True
        out["role_usage"] = reports[-1]
    out.update(calls=calls,events=events)
    plans = Plans(root)
    scope = dict(session_id="s",run_id="r",user_request="List facts from sources")
    plan = plans.prepare(scope,dict(reason="Independent investigations of source facts",tasks=[
        {"goal":"List public function names"},{"goal":"Review numerical edge behavior"}]),settings)
    out["plan"] = plan
    plans.decide(plan["plan_id"],"s",plan["digest"],True,"human")
    config["modelRouting"] = {**config["modelRouting"],"strongTimeoutSeconds":250}
    os.environ["TRITON_RISCV_CONFIG"] = json.dumps(config)
    out["executions"] = []
    try:
        plans.execute(plan["plan_id"],"s",settings,lambda task, stop: out["executions"].append(task))
        out["changed_blocked"] = False
    except PermissionError: out["changed_blocked"] = True
    out["base_model"] = base_settings(root,{"base_model":"current"}).model
print(json.dumps(out))
`
function probe(input: object) {
  const result = spawnSync(python, ['-I', '-B', '-c', driver], {
    input: JSON.stringify(input),
    encoding: 'utf8',
    timeout: 15000,
    env: Object.fromEntries(
      Object.entries(process.env).filter(
        ([key]) => !/^(TRITON_|RISCV_|DSH_)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
      ),
    ),
  })
  expect(result.status, result.stderr + result.stdout).toBe(0)
  return JSON.parse(result.stdout)
}
const policy = { enabled: true, fastModel: 'fast-fixture', strongModel: 'strong-fixture', fastMaxInputBytes: 64000 }

it('routes actual chat and typed role backend settings; exposes decisions and unknown usage', () => {
  const value = probe({ policy })
  expect(value.calls.map(x => x.model)).toEqual(['fast-fixture', 'strong-fixture', 'fast-fixture'])
  expect(value.events.map(x => x.payload.scope)).toEqual(['chat', 'chat', 'semantic'])
  expect(value.role_usage).toMatchObject({
    model: 'fast-fixture',
    outcome: 'invalid_output',
    route: { tier: 'fast' },
    totals: { usage_status: 'unknown' },
  })
  expect(value.plan.tasks.map(x => x.route.tier)).toEqual(['fast', 'strong'])
  expect(value.changed_blocked).toBe(true)
  expect(value.executions).toEqual([])
  expect(value.base_model).toBe('current')
})
it('keeps disabled and missing-fast-model configurations usable without inventing model IDs', () => {
  expect(probe({}).calls.map(x => x.model)).toEqual(['current', 'current', 'current'])
  const value = probe({ policy: { enabled: true } })
  expect(value.calls.map(x => x.model)).toEqual(['current', 'current', 'current'])
  expect(value.events[0].payload.reason).toBe('fast_model_not_configured')
})
it('does not add routing retries after ambiguous timeouts, and retains unknown usage', () => {
  const value = probe({ policy, timeout: true })
  expect(value.calls).toHaveLength(3)
  expect(value.role_timeout).toBe(true)
  expect(value.role_usage).toMatchObject({
    outcome: 'error',
    route: { model: 'fast-fixture' },
    totals: { usage_status: 'unknown' },
  })
})
it('does not downgrade short repairs/reviews, ambiguous requests or oversized Chinese context', () => {
  const value = probe({
    policy,
    cases: [
      { request: '修复log' },
      { request: '整理后执行测试' },
      { request: 'hello' },
      { request: 'List functions', role: 'reviewer' },
      { request: 'List functions', role: 'main' },
      { request: '列出字段', prompt: '中'.repeat(22000) },
    ],
  })
  expect(value.routes.every(x => x.tier === 'strong')).toBe(true)
  expect(value.routes[5]).toMatchObject({ input_bytes: 66000, reason: 'input_exceeds_fast_budget' })
})
it('distinguishes implementation-file inventories from implementation actions', () => {
  const value = probe({
    policy,
    cases: [
      { request: '列出 relu_and_mul 的实现文件、测试文件和测试节点' },
      { request: '提取 relu_and_mul 的实现路径。' },
      { request: '列出需求然后实现文件上传功能' },
      { request: '列出实现文件、修改源码并执行测试' },
      { request: '列出实现文件、审查正确性' },
    ],
  })
  expect(value.routes.map(x => x.tier)).toEqual(['fast', 'fast', 'strong', 'strong', 'strong'])
})
it('serializes routing in one credential-free host document', () => {
  const config = resolveConfig({ repoRoot: '/tmp/fixture', modelRouting: policy })
  const env = workerEnvironment(config, {})
  expect(JSON.parse(env.TRITON_RISCV_CONFIG).modelRouting).toMatchObject(policy)
  expect(() => resolveConfig({ modelRouting: { fastTimeoutSeconds: 301 } })).toThrow()
  expect(() => resolveConfig({ modelRouting: { fastModel: 'x\n' } })).toThrow()
  expect(() => resolveConfig({ modelRouting: { arbitraryModel: 'x' } })).toThrow()
})
