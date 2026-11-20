import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { delimiter, dirname, resolve } from 'node:path'

const root = fileURLToPath(new URL('../..', import.meta.url))
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || resolve(root, '.venv/bin/python')
const bootstrap = `
import sys
sys.path[:0] = sys.argv[1:3]
exec(sys.argv[3])
`
function probe(driver: string) {
  const result = spawnSync(
    python,
    ['-I', '-B', '-c', bootstrap, resolve(root, 'python'), resolve(root, '../dsh-multi-agent/python'), driver],
    {
      encoding: 'utf8',
      timeout: 25000,
      env: { PATH: [dirname(process.execPath), '/usr/bin', '/bin'].join(delimiter) },
    },
  )
  expect(result.error).toBeUndefined()
  expect(result.status, result.stdout + result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

const prelude = `
import json, sys, tempfile, time
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.delegation import Plans
from codex_agent.execution_guard import atomic_json
scope = dict(session_id="safety-session", run_id="safety-run", user_request="Review independent sources")
request = dict(reason="Independent source and test analysis", tasks=[
    {"goal":"Inspect source implementation"}, {"goal":"Inspect independent test coverage"}])
def failure(fn):
    try: fn(); return None
    except Exception as e: return type(e).__name__
def settings_for(root):
    return HarnessSettings(repo_root=root, session_root=root, cordis_path=root / "fixture.yml",
        plugin=None, mcp_python=sys.executable, provider="fixture", model="fixture-model",
        api_key_env="FIXTURE_KEY", api_key="fixture-only", base_url="http://127.0.0.1:1")
`

it('retires expired plans without transferring approval and permits explicit revocation', () => {
  const data = probe(
    prelude +
      `
out = {}
with tempfile.TemporaryDirectory() as folder, patch.object(HarnessSettings, "validate_live_run", lambda self: None):
    root = Path(folder); plans = Plans(root); settings = settings_for(root)
    first = plans.prepare(scope, request, settings)
    plans.decide(first["plan_id"], scope["session_id"], first["digest"], True, "human")
    raw = plans.load(first["plan_id"], scope["session_id"])
    raw["approval"]["at"] = time.time() - 3601
    atomic_json(plans.path(first["plan_id"]), raw)
    calls = []
    def run(task, stop): calls.append(task["task_id"]); return {"status":"completed"}
    out["expired_execute"] = failure(lambda: plans.execute(first["plan_id"], scope["session_id"], settings, run))
    second = plans.prepare(scope, request, settings)
    out["new_id"] = first["plan_id"] != second["plan_id"]
    out["old_status"] = plans.load(first["plan_id"], scope["session_id"])["status"]
    out["new_status"] = second["status"]
    out["old_reapprove"] = failure(lambda: plans.decide(first["plan_id"], scope["session_id"], first["digest"], True, "human"))
    out["unapproved_execute"] = failure(lambda: plans.execute(second["plan_id"], scope["session_id"], settings, run))
    out["wrong_session"] = failure(lambda: plans.decide(second["plan_id"], "other", second["digest"], True, "human"))
    out["wrong_digest"] = failure(lambda: plans.decide(second["plan_id"], scope["session_id"], "other", True, "human"))
    out["calls_before_approval"] = len(calls)
    plans.decide(second["plan_id"], scope["session_id"], second["digest"], True, "human")
    out["revoked"] = plans.decide(second["plan_id"], scope["session_id"], second["digest"], False, "human")["status"]
    out["revoked_execute"] = plans.execute(second["plan_id"], scope["session_id"], settings, run)["status"]
    third = plans.prepare(scope, request, settings)
    plans.decide(third["plan_id"], scope["session_id"], third["digest"], True, "human")
    out["completed"] = plans.execute(third["plan_id"], scope["session_id"], settings, run)["status"]
    plans.execute(third["plan_id"], scope["session_id"], settings, run)
    out["calls"] = len(calls)
    pending = plans.prepare(scope, request, settings)
    raw = plans.load(pending["plan_id"], scope["session_id"]); raw["created_at"] -= 3601
    atomic_json(plans.path(pending["plan_id"]), raw)
    refreshed = plans.prepare(scope, request, settings)
    out["expired_pending_replaced"] = refreshed["plan_id"] != pending["plan_id"] and refreshed["status"] == "pending"
print(json.dumps(out))
`,
  )
  expect(data).toMatchObject({
    expired_execute: 'PermissionError',
    new_id: true,
    old_status: 'cancelled',
    new_status: 'pending',
    old_reapprove: 'PermissionError',
    unapproved_execute: 'PermissionError',
    wrong_session: 'PermissionError',
    wrong_digest: 'PermissionError',
    calls_before_approval: 0,
    revoked: 'rejected',
    revoked_execute: 'rejected',
    completed: 'completed',
    calls: 2,
    expired_pending_replaced: true,
  })
})

it.each(['model', 'routing'])('requires a new approval after %s configuration changes', mode => {
  const data = probe(
    prelude +
      `
import os
out = {}
with tempfile.TemporaryDirectory() as folder, patch.object(HarnessSettings, "validate_live_run", lambda self: None):
    root = Path(folder); plans = Plans(root); settings = settings_for(root)
    old = plans.prepare(scope, request, settings)
    plans.decide(old["plan_id"], scope["session_id"], old["digest"], True, "human")
    if ${JSON.stringify(mode)} == "model": settings = replace(settings, model="changed-model")
    else:
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(dict(schemaVersion=1,repoRoot=folder,stateDir=folder,
            modelRouting={"strongTimeoutSeconds":250}))
    calls = []
    out["changed_execute"] = failure(lambda: plans.execute(old["plan_id"], scope["session_id"], settings,
        lambda task, stop: calls.append(task)))
    fresh = plans.prepare(scope, request, settings)
    out.update(new_id=fresh["plan_id"] != old["plan_id"], status=fresh["status"],
        old_status=plans.load(old["plan_id"], scope["session_id"])["status"], calls=len(calls))
    out["unapproved"] = failure(lambda: plans.execute(fresh["plan_id"], scope["session_id"], settings,
        lambda task, stop: calls.append(task)))
print(json.dumps(out))
`,
  )
  expect(data).toEqual({
    changed_execute: 'PermissionError',
    new_id: true,
    status: 'pending',
    old_status: 'cancelled',
    calls: 0,
    unapproved: 'PermissionError',
  })
})

it('kills real sleeping replicas, retains quota until reaping, and never starts queued children after cancel', () => {
  const data = probe(
    prelude +
      `
import os, signal, subprocess, threading
from contextlib import ExitStack
import codex_agent.harness.replica as replica
from dsh_multi_agent.parallel import local_replica_slot
out = {}; processes = []; errors = []; lock = threading.Lock()
started = threading.Event(); waiting = threading.Event(); release = threading.Event()
waits = []; original_popen = subprocess.Popen
def available_slots():
    count = 0
    with ExitStack() as slots:
        for _ in range(4):
            try: slots.enter_context(local_replica_slot()); count += 1
            except RuntimeError: break
    return count
def sleeper(command, **kwargs):
    process = original_popen([sys.executable, "-I", "-B", "-c", "import time; time.sleep(30)"], **kwargs)
    actual_wait = process.wait
    def wait(*args, **kw):
        with lock:
            waits.append(process.pid)
            if len(waits) == 2: waiting.set()
        if not release.wait(10): raise RuntimeError("fixture reap gate timed out")
        return actual_wait(*args, **kw)
    process.wait = wait
    with lock:
        processes.append(process)
        if len(processes) == 2: started.set()
    return process
with tempfile.TemporaryDirectory() as folder, patch.object(HarnessSettings, "validate_live_run", lambda self: None):
    root = Path(folder); plans = Plans(root); settings = settings_for(root)
    quota = root / "quota"; quota.mkdir()
    request["tasks"] += [{"goal":"Inspect independent numerical edge cases"}, {"goal":"Inspect source documentation"}]
    plan = plans.prepare(scope, request, settings)
    plans.decide(plan["plan_id"], scope["session_id"], plan["digest"], True, "human")
    def execute():
        try: out["result"] = plans.execute(plan["plan_id"], scope["session_id"], settings)
        except Exception as error: errors.append(type(error).__name__ + ":" + str(error))
    with patch.object(tempfile, "tempdir", str(quota)), patch.object(replica.subprocess, "Popen", sleeper):
        thread = threading.Thread(target=execute)
        thread.start()
        try:
            if not started.wait(10): raise RuntimeError("fixture replicas did not start")
            out["slots_running"] = available_slots()
            out["cancel"] = plans.cancel(plan["plan_id"], scope["session_id"])
            if not waiting.wait(10): raise RuntimeError("fixture replicas were not stopped")
            out["slots_before_reap"] = available_slots()
            out["coordinator_waiting"] = thread.is_alive()
            out["replan_while_draining"] = failure(lambda: plans.prepare(scope,
                {**request, "context":"new task"}, replace(settings, model="changed-model")))
            release.set(); thread.join(10)
            out["coordinator_exited"] = not thread.is_alive()
            out["slots_after_reap"] = available_slots()
            out["started_count"] = len(processes)
            out["returncodes"] = [process.returncode for process in processes]
            out["errors"] = errors
        finally:
            release.set()
            plans.cancel(plan["plan_id"], scope["session_id"])
            for process in processes:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=5)
            thread.join(10)
print(json.dumps(out))
`,
  )
  expect(data.errors).toEqual([])
  expect(data.cancel).toMatchObject({ status: 'running', cancel_requested: true })
  expect(data.slots_running).toBe(2)
  expect(data.slots_before_reap).toBe(2)
  expect(data.coordinator_waiting).toBe(true)
  expect(data.replan_while_draining).toBe('PermissionError')
  expect(data.coordinator_exited).toBe(true)
  expect(data.slots_after_reap).toBe(4)
  expect(data.started_count).toBe(2)
  expect(data.returncodes).toEqual([-9, -9])
  expect(data.result.status).toBe('cancelled')
  expect(data.result.results).toHaveLength(4)
  expect(data.result.results.every(row => row.status === 'cancelled')).toBe(true)
}, 30000)
