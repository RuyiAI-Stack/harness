import { beforeAll, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
let result: Record<string, unknown>
beforeAll(() => {
  expect(process.env.TRITON_MYSQL_URL, 'Use a disposable database').toBeTruthy()
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) =>
        !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) && !/^(TRITON_RISCV_|TRITON_WORKBENCH_|DSH_|RISCV_)/.test(key),
    ),
  )
  const driver = `
import json
from concurrent.futures import ThreadPoolExecutor
from codex_agent.tests.test_orchestration_integration import WorkflowIntegrationTests
from codex_agent.tests.test_operator_development import valid_spec
from codex_agent.orchestration.roles import operator_paths
from codex_agent.tasks.schema import jobs, outbox
from codex_agent.tasks.store import Conflict

f = WorkflowIntegrationTests('test_interactive_text_extracts_spec_and_publishes_without_writes')
f.setUp()
result = {}
try:
    spec = valid_spec()
    f.proposal()
    original = f.submit(specification=spec)
    f.approve()
    f.action('apply')
    paths = [f.root / p for p in operator_paths('square_new')]
    before = [p.read_bytes() for p in paths]
    replay = f.submit(specification=spec)
    result['after_apply_same_job'] = replay['id'] == original['id']
    result['files_unchanged'] = before == [p.read_bytes() for p in paths]
    paths[0].write_bytes(before[0] + b'\\n# later local revision\\n')
    result['after_local_edit_same_job'] = f.submit(specification=spec)['id'] == original['id']
    result['local_edit_preserved'] = paths[0].read_bytes() == before[0] + b'\\n# later local revision\\n'
    with ThreadPoolExecutor(max_workers=4) as pool:
        replays = list(pool.map(lambda _: f.submit(specification=spec)['id'], range(4)))
    result['concurrent_replay'] = replays
    result['original_id'] = original['id']
    result['job_count'] = len(f.jobs.db.rows(jobs))
    result['outbox_count'] = len(f.jobs.db.rows(outbox))
    blocked = []
    for change in [dict(requirement='different requirement'), dict(operator='different_operator'),
                   dict(session_id='foreign'), dict(auto_repair=True),
                   dict(versions={'source_sha256': 'f' * 64})]:
        try:
            f.service.submit({**f.request, **change}, 'one', specification=spec)
            blocked.append(False)
        except Conflict:
            blocked.append(True)
    for fields in [dict(specification={**spec, 'semantics': 'Different operation: negate input values'}),
                   dict(specification=spec, evidence=[dict(id='new', kind='requirement', source='test', text='new evidence')]),
                   dict(specification=spec, exclude_source_runs=['different-run']),
                   dict(specification=spec, failure_run_id='different-failure')]:
        try:
            f.service.submit(f.request, 'one', **fields)
            blocked.append(False)
        except Conflict:
            blocked.append(True)
    result['changed_requests_blocked'] = blocked
    try:
        f.submit(key='different-key', specification=spec)
        result['new_key_cannot_overwrite'] = False
    except Conflict:
        result['new_key_cannot_overwrite'] = True
finally:
    f.doCleanups()

g = WorkflowIntegrationTests('test_interactive_repair_can_derive_contract_and_display_it_for_review')
g.setUp()
try:
    failed = g.seed_failure()
    g.request.update(kind='repair', auto_publish=True)
    original = g.submit(failure_run_id=failed.run_id)
    path = g.root / operator_paths('square_new')[0]
    path.write_text(path.read_text() + '\\n# changed after submission\\n')
    result['repair_replay_same_job'] = g.submit(failure_run_id=failed.run_id)['id'] == original['id']
finally:
    g.doCleanups()
print(json.dumps(result))
`
  const output = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python'),
    ['-I', '-B', '-c', driver],
    {
      env,
      encoding: 'utf8',
      timeout: 90000,
      maxBuffer: 4 * 1024 * 1024,
    },
  )
  expect(output.error, output.stderr).toBeUndefined()
  expect(output.status, output.stdout + output.stderr).toBe(0)
  result = JSON.parse(output.stdout)
}, 100000)

it('returns the original job after apply without overwriting files or later edits', () => {
  for (const key of ['after_apply_same_job', 'files_unchanged', 'after_local_edit_same_job', 'local_edit_preserved'])
    expect(result[key], key).toBe(true)
})
it('serializes concurrent replay without additional jobs or Outbox messages', () => {
  expect(result.concurrent_replay).toEqual(Array(4).fill(result.original_id))
  expect(result.job_count).toBe(1)
  expect(result.outbox_count).toBe(1)
})
it('rejects changed requests and a new key attempting to overwrite existing code', () => {
  expect(result.changed_requests_blocked).toEqual(Array(9).fill(true))
  expect(result.new_key_cannot_overwrite).toBe(true)
})
it('returns an existing repair job without re-auditing a now-obsolete source snapshot', () => {
  expect(result.repair_replay_same_job).toBe(true)
})
