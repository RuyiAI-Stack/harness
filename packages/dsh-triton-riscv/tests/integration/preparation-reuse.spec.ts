import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python')
// The existing fixture supplies disposable state. Vitest owns the new assertions.
const driver = `
import json, sys
from types import SimpleNamespace
from unittest.mock import patch
from codex_agent.tests.test_orchestration_integration import WorkflowIntegrationTests
from codex_agent.orchestration.runtime import RoleAgent
from codex_agent.orchestration.roles import operator_paths
from codex_agent.orchestration.service import WorkflowSeed
from codex_agent.orchestration.reuse import _plan, preparation_scope, load_reused_tests, reuse_preparation
from codex_agent.orchestration.usage_report import workflow_usage
from codex_agent.tasks.store import Conflict, digest
data = json.load(sys.stdin)
f = WorkflowIntegrationTests('test_failure_creates_one_child_then_success_preserves_failed_attempt')
f.setUp()
try:
    f.backend.settings = SimpleNamespace(provider='fixture', model='fixture', base_url='http://127.0.0.1')
    f.request.update(auto_repair=True, max_repairs=2, reuse_preparation=not data.get('disabled', False))
    parent_preapproval_exists = None
    if data.get('frozen_create'):
        f.backend.decision = 'request_semantics'
        def parent_reply(request, result):
            global parent_preapproval_exists
            f.reply(request, result)
            if request['phase'] == 'reviewing':
                parent_preapproval_exists = (f.root / operator_paths('square_new')[0]).exists()
        f.backend.change = parent_reply
    f.validation(f.queued(), passed=False)
    parent = f.current()
    child = f.store.get(parent['result']['lifecycle']['child_workflow_id'])
    seed = WorkflowSeed.model_validate(f.store.history(child['id'], limit=1)[0]['payload']['context'])
    agent = RoleAgent(backend=f.backend)
    if data.get('probe'):
        scenario = data['probe']
        scope = preparation_scope(f.root, child, seed, seed.contract, agent)
        modified_parent = json.loads(json.dumps(parent))
        if scenario == 'foreign_session': modified_parent['session_id'] = 'foreign'
        if scenario == 'foreign_operator': modified_parent['operator'] = 'foreign'
        if scenario == 'new_requirement': modified_parent['request']['requirement'] += ' different'
        if scenario == 'new_contract': modified_parent['versions']['contract_sha256'] = '0' * 64
        if scenario == 'new_tests': modified_parent['versions']['tests_sha256'] = '0' * 64
        if scenario == 'wrong_run': modified_parent['result']['lifecycle']['validation_run_id'] = 'wrong'
        if scenario == 'not_reciprocal': modified_parent['result']['lifecycle']['child_workflow_id'] = 'wrong'
        if scenario == 'unproven_parent': modified_parent['state'] = 'waiting_validation'
        if scenario == 'missing_failure': seed = seed.model_copy(update={'failure_receipt_sha256': None})
        if scenario == 'changed_scope': scope = '0' * 64
        original = f.store.get
        with patch('codex_agent.orchestration.reuse.WorkflowStore.get', side_effect=lambda key: modified_parent if key == parent['id'] else original(key)):
            result = _plan(f.jobs.db, child, seed, scope)
    else:
        checkpoint_replayed = None
        if data.get('resume'):
            checkpoint = reuse_preparation(f.store, child, seed, agent)
            checkpoint_replayed = reuse_preparation(f.store, checkpoint, seed, agent)['revision'] == checkpoint['revision']
            if data.get('settings_after_checkpoint'):
                f.backend.settings.model = 'changed-after-checkpoint'
        if data.get('changed_model'): f.backend.settings.model = 'different-model'
        if data.get('legacy'): del f.backend.settings
        if data.get('unreadable_environment'):
            (f.root / 'scripts/triton-riscv-env.sh').mkdir(parents=True)
        if data.get('redundant_semantics'): f.backend.decision = 'request_semantics'
        if data.get('corrupt_plan'):
            from codex_agent.artifacts import ArtifactStore
            from codex_agent.tasks.schema import artifacts
            step = next(s for s in parent['steps'] if s['state'] == 'planning_tests')
            record = f.jobs.db.rows(artifacts, artifacts.c.id == step['result']['artifact_id'])[0]
            (ArtifactStore(f.root).folder / record['storage_key']).write_bytes(b'corrupt fixture')
        before = len(f.backend.calls)
        original_tests = (f.root / operator_paths('square_new')[1]).read_bytes()
        original_source = (f.root / operator_paths('square_new')[0]).read_bytes()
        f.workflow_id = child['id']
        def response(request, result):
            f.reply(request, result)
            if data.get('ask_user') and request['role'] == 'main':
                result.update(decision='ask_user', questions=['Confirm changed requirement'])
            if request['role'] == 'implementer': result['candidate']['content'] += '\\n# corrected fixture child\\n'
            if data.get('review_reject') and request['phase'] == 'reviewing':
                result.update(verdict='changes_requested', findings=['fixture: boundary not fixed'])
        f.backend.change = response
        job = f.jobs.get(child['result']['agent_job_id'])
        if data.get('timeout'):
            f.backend.run = lambda *a, **k: (_ for _ in ()).throw(TimeoutError('synthetic timeout'))
        executed = f.execute(job)
        current = f.current()
        result = dict(state=current['state'], job_status=executed['status'],
            checkpoint_replayed=checkpoint_replayed,
            parent_preapproval_exists=parent_preapproval_exists,
            parent_contract_unchanged=parent['versions']['contract_sha256'] == digest(f.store.history(parent['id'], limit=1)[0]['payload']['context']['contract']),
            decision=current['result'].get('preparation_reuse'),
            phases=[r['phase'] for r in f.backend.calls[before:]],
            parent_phases=[r['phase'] for r in f.backend.calls[:before]],
            files_unchanged=(f.root / operator_paths('square_new')[0]).read_bytes() == original_source,
            tests_unchanged=(f.root / operator_paths('square_new')[1]).read_bytes() == original_tests,
            usage=workflow_usage(f.jobs.db, child['id']))
        if current['state'] == 'waiting_approval':
            result['proposal'] = f.lifecycle.view(child['id'])['proposal']['status']
            result['artifact_error'] = f.lifecycle.view(child['id'])['artifact_error']
            try:
                f.action('apply')
                result['unapproved_blocked'] = False
            except PermissionError:
                result['unapproved_blocked'] = True
            if data.get('tamper'):
                forged = json.loads(json.dumps(current))
                forged['result']['preparation_reuse']['source_artifact_sha256'] = '0' * 64
                try:
                    load_reused_tests(f.jobs.db, forged, seed)
                    result['tamper_blocked'] = False
                except Conflict:
                    result['tamper_blocked'] = True
            if data.get('finish'):
                f.approve()
                f.action('apply')
                f.action('plan-validation')
                from codex_agent import operator_lifecycle as domain
                domain.decide_validation_plan(f.root, f.current()['result']['lifecycle']['validation_plan_id'],
                    approve=True, reviewer='synthetic-test-human')
                f.action('enqueue-validation')
                validation = f.jobs.get(f.current()['result']['lifecycle']['validation_job_id'])
                f.validation(validation, passed=not data.get('second_failure', False))
                result['after_validation'] = f.current()['state']
                if data.get('second_failure'):
                    again_before = len(f.backend.calls)
                    next_job = f.next_repair()
                    result['grandchild_decision'] = f.current()['result'].get('preparation_reuse')
                    result['grandchild_phases'] = [c['phase'] for c in f.backend.calls[again_before:]]
                    f.validation(next_job, passed=False)
                    result['stop_reason'] = f.current()['result']['lifecycle'].get('stop_reason')
                result['tests_still_unchanged'] = (f.root / operator_paths('square_new')[1]).read_bytes() == original_tests
                result['repeat_collection_state'] = f.action('collect-validation')['state']
finally:
    f.doCleanups()
print(json.dumps(result))
`

function invoke(input: object) {
  expect(process.env.TRITON_MYSQL_URL, 'Use a disposable database').toBeTruthy()
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) => !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) && !/^(TRITON_RISCV_CONFIG|DSH_|RISCV_)/.test(key),
    ),
  )
  const result = spawnSync(
    python,
    [
      '-I',
      '-B',
      '-c',
      'import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))',
      join(root, 'python'),
      driver,
    ],
    {
      input: JSON.stringify(input),
      encoding: 'utf8',
      timeout: 60000,
      maxBuffer: 8 * 1024 * 1024,
      env,
    },
  )
  expect(result.error, result.stderr).toBeUndefined()
  expect(result.status, result.stdout + result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

describe('host-verified frozen preparation reuse', () => {
  it('skips two calls only in a proven repair chain and still requires approval and validation', () => {
    const value = invoke({ finish: true, tamper: true })
    expect(value.parent_phases).toEqual(['preparing', 'planning_tests', 'implementing', 'reviewing'])
    expect(value.phases).toEqual(['implementing', 'reviewing'])
    expect(value.decision).toMatchObject({ status: 'reused', avoided_role_calls_minimum: 2 })
    expect(value).toMatchObject({
      state: 'waiting_approval',
      proposal: 'pending_approval',
      files_unchanged: true,
      tests_unchanged: true,
      unapproved_blocked: true,
      tamper_blocked: true,
      artifact_error: null,
      after_validation: 'succeeded',
      tests_still_unchanged: true,
      repeat_collection_state: 'succeeded',
    })
    expect(value.usage.totals.role_invocations).toBe(6)
    expect(value.usage.totals.total_tokens).toBeNull()
    expect(value.usage.workflows[1].preparation_reuse.status).toBe('reused')
  }, 65000)

  it.each(['disabled', 'changed_model', 'legacy', 'corrupt_plan', 'unreadable_environment'])(
    'uses the full route for %s',
    key => {
      const value = invoke({ [key]: true })
      expect(value.decision.status).toBe('not_reused')
      expect(value.phases).toEqual(['preparing', 'planning_tests', 'implementing', 'reviewing'])
      expect(value.artifact_error).toBeNull()
      expect(value.unapproved_blocked).toBe(true)
    },
    65000,
  )

  it.each([
    'foreign_session',
    'foreign_operator',
    'new_requirement',
    'new_contract',
    'new_tests',
    'wrong_run',
    'not_reciprocal',
    'unproven_parent',
    'missing_failure',
    'changed_scope',
  ])(
    'rejects reuse when evidence changes: %s',
    probe => {
      expect(invoke({ probe }).status).toBe('not_reused')
    },
    65000,
  )

  it('does not suppress independent review or allow endless repeated patches', () => {
    const value = invoke({ review_reject: true })
    expect(value.decision.status).toBe('reused')
    expect(value.state).toBe('blocked')
    expect(value.phases).toEqual(['implementing', 'reviewing', 'implementing'])
    expect(value.files_unchanged).toBe(true)
  }, 65000)

  it('keeps timed-out calls unknown and never claims that reuse completed the task', () => {
    const value = invoke({ timeout: true })
    expect(value.decision.status).toBe('reused')
    expect(value.state).not.toBe('succeeded')
    expect(value.files_unchanged).toBe(true)
    expect(value.usage.totals.total_tokens).toBeNull()
    expect(value.usage.totals.unsuccessful_invocations).toBe(1)
  }, 65000)

  it('resumes a saved reuse checkpoint idempotently without repeating planning', () => {
    const value = invoke({ resume: true })
    expect(value.checkpoint_replayed).toBe(true)
    expect(value.phases).toEqual(['implementing', 'reviewing'])
    expect(value.artifact_error).toBeNull()
  }, 65000)

  it('does not re-extract an authoritative contract even on the full route', () => {
    const value = invoke({ disabled: true, redundant_semantics: true })
    expect(value.phases).toEqual(['preparing', 'planning_tests', 'implementing', 'reviewing'])
    expect(value.state).toBe('waiting_approval')
    expect(value.artifact_error).toBeNull()
  }, 65000)

  it('preserves the frozen creation contract without calling semantics or writing before approval', () => {
    const value = invoke({ frozen_create: true })
    expect(value.parent_phases).toEqual(['preparing', 'planning_tests', 'implementing', 'reviewing'])
    expect(value.parent_contract_unchanged).toBe(true)
    expect(value.parent_preapproval_exists).toBe(false)
  }, 65000)

  it('still stops for explicit user clarification instead of treating a contract as permission', () => {
    const value = invoke({ disabled: true, ask_user: true })
    expect(value.phases).toEqual(['preparing'])
    expect(value.state).toBe('blocked')
    expect(value.files_unchanged).toBe(true)
  }, 65000)

  it('blocks a resumed shortcut if model settings changed after the checkpoint', () => {
    const value = invoke({ resume: true, settings_after_checkpoint: true })
    expect(value.state).toBe('blocked')
    expect(value.phases).toEqual([])
    expect(value.files_unchanged).toBe(true)
  }, 65000)

  it('follows the original test plan across retries without resetting the global repair limit', () => {
    const value = invoke({ finish: true, second_failure: true })
    expect(value.grandchild_decision.status).toBe('reused')
    expect(value.grandchild_decision.source_workflow_id).toBe(value.decision.source_workflow_id)
    expect(value.grandchild_phases).toEqual(['implementing', 'reviewing'])
    expect(value.stop_reason).toBe('repair_budget_exhausted')
    expect(value.tests_still_unchanged).toBe(true)
  }, 65000)
})
