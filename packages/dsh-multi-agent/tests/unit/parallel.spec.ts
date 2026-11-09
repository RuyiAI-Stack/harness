import { expect, it } from 'vitest'
import { probe } from '../probe'

it('caps fan-out and host-local admission; cancellation does not start pending children', () => {
  const data = probe(`
import json, tempfile, threading, time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from dsh_multi_agent.parallel import run_parallel, local_replica_slot
out = {}
tasks = [{"task_id":str(i)} for i in range(4)]
def fail(fn):
    try: fn(); return None
    except Exception as e: return type(e).__name__
out["too_many"] = fail(lambda: run_parallel(tasks * 2, lambda task, stop: {}))
out["parallel_limit"] = fail(lambda: run_parallel(tasks, lambda task, stop: {}, max_parallel=3))
with tempfile.TemporaryDirectory(prefix="dsh-quota-test-") as folder:
    with patch("dsh_multi_agent.parallel.tempfile.gettempdir", return_value=folder):
        with ExitStack() as slots:
            for _ in range(4): slots.enter_context(local_replica_slot())
            def fifth():
                with local_replica_slot(): pass
            out["overload"] = fail(fifth)
        with local_replica_slot(): out["released"] = True
out["quota_cleaned"] = not Path(folder).exists()
stop = threading.Event()
calls = []
def run(task, cancelled):
    calls.append(task["task_id"])
    stop.set()
    return {"status":"cancelled"}
out["rows"] = run_parallel(tasks, run, cancelled=stop.is_set)
out["calls"] = len(calls)
print(json.dumps(out))
`)
  expect(data.too_many).toBe('ValueError')
  expect(data.parallel_limit).toBe('ValueError')
  expect(data.overload).toBe('RuntimeError')
  expect(data.released).toBe(true)
  expect(data.quota_cleaned).toBe(true)
  expect(data.calls).toBeLessThanOrEqual(2)
  expect(data.rows).toHaveLength(4)
  expect(data.rows.every(row => row.status === 'cancelled')).toBe(true)
})
