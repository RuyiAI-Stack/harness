import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const python =
  process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url))
const driver = `
import asyncio, json, os, tempfile, threading, time
from pathlib import Path
from dataclasses import replace
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.delegation import Plans
from codex_agent.execution_guard import atomic_json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from codex_agent.harness.delegation_api import router
out = {}
def failure(fn):
    try: fn(); return None
    except Exception as e: return type(e).__name__
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    settings = HarnessSettings.from_env(root, {"ISRC_API_KEY":"fixture", "DSH_MODEL":"fixture-model"})
    plans = Plans(root)
    scope = {"session_id":"session-A", "run_id":"run-1", "user_request":"Compare the implementations without changing files"}
    request = {"reason":"Independent source analysis can be performed concurrently", "tasks":[{"goal":"Inspect the forward implementation"}, {"goal":"Inspect independent backward tests"}]}
    plan = plans.prepare(scope, request, settings)
    identifier = plan["plan_id"]
    out["pending"] = plan["status"]
    calls, active, peak = [], 0, 0
    lock = threading.Lock()
    def runner(task, stop):
        global active, peak
        with lock: active += 1; peak = max(peak, active); calls.append(task["task_id"])
        time.sleep(.08)
        with lock: active -= 1
        return {"status":"completed", "response":"source evidence, not a test pass"}
    out["unapproved"] = failure(lambda: plans.execute(identifier, "session-A", settings, runner))
    out["calls_before_approval"] = len(calls)
    out["wrong_session"] = failure(lambda: plans.load(identifier, "session-B"))
    out["wrong_digest"] = failure(lambda: plans.decide(identifier, "session-A", "wrong", True, "human"))
    out["duplicate_plan"] = plans.prepare(scope, request, settings)["plan_id"] == identifier
    class Store:
        def get_session(self, session):
            if session not in {"session-A", "session-B"}: raise KeyError(session)
    app = FastAPI(); app.include_router(router(root, Store()))
    with TestClient(app) as client:
        base = "/api/sessions/session-A/parallel-plans/" + identifier
        out["api_wrong_digest"] = client.post(base + "/decision", json={"approved":True,"digest":"bad"}).status_code
        out["api_cross_session"] = client.get(base.replace("session-A", "session-B")).status_code
        out["api_approve"] = client.post(base + "/decision", json={"approved":True,"digest":plan["digest"]}).status_code
        out["model_changed"] = failure(lambda: plans.execute(identifier, "session-A", replace(settings, model="other"), runner))
        out["result"] = plans.execute(identifier, "session-A", settings, runner)
        plans.execute(identifier, "session-A", settings, runner)
        out.update(calls=len(calls), peak=peak)
        second = plans.prepare(scope, request, settings)
        rejected = client.post(base.replace(identifier,second["plan_id"]) + "/decision", json={"approved":False,"digest":second["digest"]}).json()
        out["rejected"] = plans.execute(second["plan_id"], "session-A", settings, runner)["status"]
        third = plans.prepare(scope, request, settings)
        plans.decide(third["plan_id"], "session-A", third["digest"], True, "human")
        def partial(task, stop):
            if task["task_id"] == "child-2": raise ValueError("secret-provider-error")
            return {"status":"completed", "response":"one useful finding"}
        out["partial"] = plans.execute(third["plan_id"], "session-A", settings, partial)
        fourth = plans.prepare(scope, request, settings)
        plans.decide(fourth["plan_id"], "session-A", fourth["digest"], True, "human")
        raw = plans.load(fourth["plan_id"], "session-A"); raw["status"] = "running"
        atomic_json(plans.path(fourth["plan_id"]), raw)
        out["crashed"] = plans.execute(fourth["plan_id"], "session-A", settings, runner)["status"]
        fifth = plans.prepare(scope, request, settings)
        plans.cancel(fifth["plan_id"], "session-A")
        out["cancelled"] = plans.execute(fifth["plan_id"], "session-A", settings, runner)["status"]
        out["calls_final"] = len(calls)
    os.environ["TRITON_REPLICA_READ_ONLY"] = "1"
    from codex_agent.harness.mcp_server import server
    out["child_tools"] = [tool.name for tool in asyncio.run(server.list_tools())]
    os.environ.pop("TRITON_REPLICA_READ_ONLY")
    import importlib
    import codex_agent.harness.mcp_server as module
    out["parent_tools"] = [tool.name for tool in asyncio.run(importlib.reload(module).server.list_tools())]
    from codex_agent.harness.replica import run_replica
    stopped_spec = {**plan, "timeout_seconds":0, "base_url":"http://127.0.0.1:1/v1"}
    stopped_task = {k:v for k,v in plan["tasks"][0].items() if k != "route"}
    out["process_timeout"] = run_replica(settings, stopped_spec, stopped_task, lambda: False)
    out["process_cancel"] = run_replica(settings, stopped_spec, stopped_task, lambda: True)
print(json.dumps(out))
`

it('enforces human/session/digest gates, bounded fan-out, replay and partial results', () => {
  const result = spawnSync(python, ['-I', '-B', '-c', driver], {
    encoding: 'utf8',
    timeout: 15000,
    env: Object.fromEntries(
      Object.entries(process.env).filter(
        ([key]) => !/^(TRITON_|RISCV_|DSH_)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
      ),
    ),
  })
  expect(result.status, result.stdout + result.stderr).toBe(0)
  const data = JSON.parse(result.stdout)
  expect(data.pending).toBe('pending')
  expect(data.unapproved).toBe('PermissionError')
  expect(data.calls_before_approval).toBe(0)
  expect(data.wrong_session).toBe('PermissionError')
  expect(data.wrong_digest).toBe('PermissionError')
  expect(data.api_wrong_digest).toBe(409)
  expect(data.api_cross_session).toBe(409)
  expect(data.api_approve).toBe(200)
  expect(data.duplicate_plan).toBe(true)
  expect(data.model_changed).toBe('PermissionError')
  expect(data.result.status).toBe('completed')
  expect(data.result.results).toHaveLength(2)
  expect(data.peak).toBe(2)
  expect(data.calls).toBe(2)
  expect(data.calls_final).toBe(2)
  expect(data.rejected).toBe('rejected')
  expect(data.partial.status).toBe('partial')
  expect(JSON.stringify(data.partial)).not.toContain('secret-provider-error')
  expect(data.crashed).toBe('unknown')
  expect(data.process_timeout.status).toBe('timed_out')
  expect(data.process_cancel.status).toBe('cancelled')
  expect(data.process_timeout.usage).toBeNull()
  expect(data.process_timeout.usage_complete).toBe(false)
  expect(data.cancelled).toBe('cancelled')
  expect(data.child_tools.sort()).toEqual([
    'discover_operator',
    'inspect_project',
    'read_operator_file',
    'retrieve_operator_memory',
  ])
  expect(data.parent_tools).toEqual(
    expect.arrayContaining(['prepare_parallel_tasks', 'execute_parallel_tasks', 'get_parallel_tasks']),
  )
})
