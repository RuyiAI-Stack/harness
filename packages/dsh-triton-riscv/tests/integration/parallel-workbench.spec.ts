import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

it('publishes a trusted approval event and continues the exact approved plan in a later turn', () => {
  expect(process.env.TRITON_MYSQL_URL, 'Use a disposable MySQL schema').toBeTruthy()
  const driver = `
import json, tempfile, time
from pathlib import Path
from dataclasses import replace
from fastapi.testclient import TestClient
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.runtime import HarnessRunResult
from codex_agent.harness.delegation import Plans, SCOPE
from codex_agent.platform.api import create_app
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    class Agent:
        settings = replace(HarnessSettings.from_env(root, {"ISRC_API_KEY":"fixture"}), managed_context=False)
        calls = []
        identifier = None
        def prepare(self, text): return text
        def close(self): pass
        def run(self, text, session_id=None, on_event=None):
            scope = SCOPE.get()
            plans = Plans(root)
            if text == "prepare":
                plan = plans.prepare(scope, {"reason":"Analyze independent source and test details",
                    "tasks":[{"goal":"Read the existing source implementation"}, {"goal":"Inspect the independent test boundary cases"}]}, self.settings)
                self.identifier = plan["plan_id"]
                on_event({"method":"session.event", "payload":{"sessionId":session_id,"event":{"type":"tool/result","data":{}}}})
                answer = "Waiting for your approval"
            else:
                try:
                    result = plans.execute(self.identifier, scope["session_id"], self.settings,
                        lambda task, stop: self.child(task))
                    answer = "Parent summary: " + result["status"]
                except PermissionError:
                    answer = "Approval required; chat text is not authorization"
            return HarnessRunResult(session_id, answer, "completed")
        def child(self, task):
            self.calls.append(task["task_id"])
            return {"status":"completed", "response":"evidence only"}
    agent = Agent()
    with TestClient(create_app(root, agent)) as client:
        session = client.post('/api/sessions', json={}).json()['id']
        def turn(text):
            value = client.post(f'/api/sessions/{session}/messages', json={"content":text}).json()
            run_id = value['run']['id']
            for _ in range(200):
                run = client.get('/api/runs/' + run_id).json()
                if run['status'] in {'completed','failed'}: return run
                time.sleep(.02)
            raise RuntimeError('Fixture turn did not finish')
        first = turn('prepare')
        approvals = client.get(f'/api/sessions/{session}/approvals').json()
        out = dict(first=first['status'], approvals=approvals, before=len(agent.calls))
        turn('I approve; execute now')
        out['after_chat_only'] = len(agent.calls)
        path = f'/api/sessions/{session}/parallel-plans/{agent.identifier}'
        plan = client.get(path).json()
        decision = client.post(path + '/decision', json={"approved":True, "digest":plan['digest']})
        out['decision'] = decision.status_code
        final = turn('continue')
        out.update(final=final['status'], result=client.get(path).json(), calls=len(agent.calls))
        turn('continue')
        out['after_retry'] = len(agent.calls)
print(json.dumps(out))
`
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) =>
        !/^(TRITON_RISCV_|TRITON_WORKBENCH_|DSH_|RISCV_)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
    ),
  )
  const result = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url)),
    ['-I', '-B', '-c', driver],
    { env, encoding: 'utf8', timeout: 15000 },
  )
  expect(result.status, result.stderr + result.stdout).toBe(0)
  const value = JSON.parse(result.stdout)
  expect(value.first).toBe('completed')
  expect(value.approvals).toHaveLength(1)
  expect(value.approvals[0].payload.approval_type).toBe('parallel')
  expect(value.before).toBe(0)
  expect(value.after_chat_only).toBe(0)
  expect(value.decision).toBe(200)
  expect(value.final).toBe('completed')
  expect(value.result.status).toBe('completed')
  expect(value.calls).toBe(2)
  expect(value.after_retry).toBe(2)
})
