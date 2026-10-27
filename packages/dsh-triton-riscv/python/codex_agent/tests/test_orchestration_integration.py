"""Stage-five integration with synthetic roles/execution and disposable services."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from codex_agent import operator_lifecycle as domain
from codex_agent.orchestration.api import router
from codex_agent.orchestration.retrieval import role_evidence
from codex_agent.orchestration.schema import tasks
from codex_agent.orchestration.service import WorkflowSeed
from codex_agent.orchestration.roles import operator_paths
from codex_agent.tasks.schema import jobs, artifacts
from codex_agent.tests import test_orchestration_lifecycle as fixture
from codex_agent.tests.test_operator_development import valid_spec, IMPLEMENTATION, TEST_SOURCE


class WorkflowIntegrationTests(unittest.TestCase):
    submit = fixture.WorkflowLifecycleTests.submit
    execute = fixture.WorkflowLifecycleTests.execute
    task = fixture.WorkflowLifecycleTests.task
    claim = fixture.WorkflowLifecycleTests.claim
    proposal = fixture.WorkflowLifecycleTests.proposal
    current = fixture.WorkflowLifecycleTests.current
    action = fixture.WorkflowLifecycleTests.action
    approve = fixture.WorkflowLifecycleTests.approve
    queued = fixture.WorkflowLifecycleTests.queued
    synthetic_execution = fixture.WorkflowLifecycleTests.synthetic_execution
    validation = fixture.WorkflowLifecycleTests.validation
    seed_failure = fixture.WorkflowLifecycleTests.seed_failure
    reply = staticmethod(fixture.WorkflowLifecycleTests.reply)

    def setUp(self):
        fixture.WorkflowLifecycleTests.setUp(self)
        self.memory = patch('codex_agent.orchestration.retrieval.retrieve_operator_memory',
            return_value=SimpleNamespace(status='not_found', context_excerpt='', items=[]))
        self.retrieval = self.memory.start()
        self.addCleanup(self.memory.stop)
        app = FastAPI()
        app.include_router(router(self.root))
        self.client = TestClient(app)

    def test_interactive_text_extracts_spec_and_publishes_without_writes(self):
        self.backend.decision = 'request_semantics'
        def response(request, result):
            self.reply(request, result)
            if request['role'] == 'semantic':
                result['specification'] = valid_spec()
        self.backend.change = response
        body = {'request': {**self.request, 'auto_publish': True}, 'request_id': 'natural'}
        response = self.client.post('/api/operator-workflows', json=body)
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(self.execute(job)['status'], 'succeeded')
        task = self.lifecycle.view(self.task(job)['id'])
        self.assertEqual(task['state'], 'waiting_approval')
        self.assertEqual(task['proposal']['status'], 'pending_approval')
        self.assertEqual(task['acceptance'], valid_spec())
        self.assertFalse((self.root / operator_paths('square_new')[0]).exists())
        self.assertEqual(self.client.get('/api/operator-workflows', params={'session_id': self.session['id']}).json()[0]['id'], task['id'])
        self.assertEqual(self.client.post('/api/operator-workflows', json=body).json()['id'], job['id'])

    def test_missing_semantics_blocks_without_inventing_contract(self):
        self.backend.decision = 'request_semantics'
        def response(request, result):
            if request['role'] == 'semantic':
                result.update(status='insufficient_evidence', unknowns=['dtype missing'], contract=None)
        self.backend.change = response
        job = self.submit()
        self.execute(job)
        task = self.task(job)
        self.assertEqual(task['state'], 'blocked')
        self.assertEqual(task['result']['agent_job_id'], job['id'])
        self.assertNotIn('lifecycle', task['result'])
        self.assertEqual(len(self.backend.calls), 2)

    def test_interactive_repair_can_derive_contract_and_display_it_for_review(self):
        failure = self.seed_failure()
        self.request.update(kind='repair', auto_publish=True)
        job = self.submit(failure_run_id=failure.run_id)
        self.assertEqual(self.execute(job)['status'], 'succeeded')
        task = self.lifecycle.view(self.task(job)['id'])
        self.assertIsNone(task['artifact_error'])
        self.assertEqual(task['proposal']['status'], 'pending_approval')
        self.assertEqual(task['acceptance']['formula'], 'y=x*x')
        self.assertEqual((self.root / operator_paths('square_new')[1]).read_text(), TEST_SOURCE)
        semantic = next(c for c in self.backend.calls if c['role'] == 'semantic')
        self.assertEqual([f['path'] for f in semantic['files']], [operator_paths('square_new')[1]])

    def review_reply(self, request, result, *, repeated=False, always=False):
        self.reply(request, result)
        if request['role'] == 'implementer' and not repeated:
            result['candidate']['content'] += '\n# synthetic revision %s\n' % request['input_epoch']
        if request['phase'] == 'reviewing':
            count = sum(c['phase'] == 'reviewing' for c in self.backend.calls)
            if always or count == 1:
                result.update(verdict='changes_requested', findings=['Check tail mask against the frozen tests.'])

    def test_review_rework_retains_fixed_tests_and_feedback(self):
        self.backend.change = self.review_reply
        self.proposal()
        task = self.current()
        self.assertEqual(task['result']['rework_count'], 1)
        self.assertEqual(len(self.backend.calls), 6)
        implementers = [c for c in self.backend.calls if c['role'] == 'implementer']
        self.assertTrue(any(e['id'] == 'workflow_review' for e in implementers[1]['evidence']))
        self.assertEqual(implementers[0]['versions']['tests_sha256'], implementers[1]['versions']['tests_sha256'])
        self.assertEqual(self.lifecycle.view(task['id'])['artifact_error'], None)

    def test_same_patch_stops_without_infinite_rework(self):
        self.backend.change = lambda request, result: self.review_reply(request, result, repeated=True)
        job = self.submit(specification=valid_spec())
        self.execute(job)
        task = self.task(job)
        self.assertEqual(task['state'], 'blocked')
        self.assertIn('same implementation', task['result']['reason'])
        self.assertEqual(task['result']['rework_count'], 1)
        self.assertEqual(len(self.backend.calls), 5)

    def test_review_rework_budget_is_enforced(self):
        self.request['max_repairs'] = 1
        self.backend.change = lambda request, result: self.review_reply(request, result, always=True)
        job = self.submit(specification=valid_spec())
        self.execute(job)
        self.assertEqual(self.task(job)['state'], 'blocked')
        self.assertEqual(self.task(job)['result']['rework_count'], 1)
        self.assertEqual(len(self.backend.calls), 6)

    def test_stage_rag_enters_role_context_and_saved_audit(self):
        self.retrieval.return_value = SimpleNamespace(status='found', context_excerpt='Historical mask advice, not validated for this task.', items=[SimpleNamespace(memory_id='older-case')])
        job = self.submit(specification=valid_spec())
        self.execute(job)
        calls = self.backend.calls
        self.assertFalse(any(e['id'] == 'workflow_history' for e in calls[0]['evidence']))
        for call in calls[1:]:
            history = next(e for e in call['evidence'] if e['id'] == 'workflow_history')
            self.assertFalse(history['verified'])
            self.assertIn('not validated', history['text'])
        self.assertEqual([c.kwargs['task_stage'] for c in self.retrieval.call_args_list], ['validation', 'generation', 'validation'])
        rows = self.jobs.db.rows(artifacts, artifacts.c.job_id == job['id'])
        from codex_agent.artifacts import ArtifactStore
        saved = [json.loads(ArtifactStore(self.root).read(row)) for row in rows if row['name'] != 'result.json']
        self.assertTrue(any(a['retrieval'].get('case_ids') == ['older-case'] and 'Historical mask advice' in a['task_prompt'] for a in saved))

    def test_rag_failure_and_oversize_fall_back_without_claiming_evidence(self):
        task = {'kind': 'repair', 'operator': 'square_new', 'request': self.request}
        seed = WorkflowSeed(exclude_source_runs=['old-run', 'current-run'])
        self.retrieval.side_effect = RuntimeError('secret must not be echoed')
        evidence, trace = role_evidence(self.root, task, seed, 'implementer', 'implementing')
        self.assertEqual(evidence, [])
        self.assertEqual(trace['error_type'], 'RuntimeError')
        self.assertNotIn('secret', json.dumps(trace))
        self.retrieval.side_effect = None
        self.retrieval.return_value = SimpleNamespace(status='found', context_excerpt='x' * 6001, items=[])
        evidence, trace = role_evidence(self.root, task, seed, 'implementer', 'implementing')
        self.assertEqual(evidence, [])
        self.assertEqual(trace['status'], 'omitted')
        self.assertEqual(self.retrieval.call_args.kwargs['exclude_source_runs'], ('old-run', 'current-run'))

    def next_repair(self):
        parent = self.current()
        child_id = parent['result']['lifecycle']['child_workflow_id']
        child = self.store.get(child_id)
        self.workflow_id = child_id
        def response(request, result):
            self.reply(request, result)
            if request['role'] == 'implementer':
                result['candidate']['content'] += '\n# synthetic child %s\n' % child_id
        self.backend.change = response
        job = self.jobs.get(child['result']['agent_job_id'])
        self.assertEqual(self.execute(job)['status'], 'succeeded')
        self.assertEqual(self.lifecycle.view(child_id)['proposal']['status'], 'pending_approval')
        self.approve()
        self.action('apply')
        self.action('plan-validation')
        plan = self.current()['result']['lifecycle']['validation_plan_id']
        domain.decide_validation_plan(self.root, plan, approve=True, reviewer='synthetic-test-human')
        self.action('enqueue-validation')
        return self.jobs.get(self.current()['result']['lifecycle']['validation_job_id'])

    def test_failure_creates_one_child_then_success_preserves_failed_attempt(self):
        self.request.update(auto_repair=True, max_repairs=2)
        failed = self.validation(self.queued(), passed=False)
        parent = self.current()
        self.assertEqual(parent['state'], 'failed')
        child = self.store.get(parent['result']['lifecycle']['child_workflow_id'])
        self.assertEqual(child['repair_count'], 1)
        self.assertEqual(child['result']['parent_workflow_id'], parent['id'])
        seed = self.store.history(child['id'], limit=1)[0]['payload']['context']
        self.assertIn('operator-lifecycle:' + failed['result']['run_id'], seed['exclude_source_runs'])
        count = len(self.jobs.db.rows(tasks))
        self.action('collect-validation')
        self.assertEqual(len(self.jobs.db.rows(tasks)), count)
        accepted_tests = (self.root / operator_paths('square_new')[1]).read_text()
        self.validation(self.next_repair())
        self.assertEqual(self.current()['state'], 'succeeded')
        self.assertEqual(self.store.get(parent['id'])['state'], 'failed')
        self.assertEqual((self.root / operator_paths('square_new')[1]).read_text(), accepted_tests)
        self.assertEqual(self.lifecycle.view(self.workflow_id)['artifact_error'], None)

    def test_failed_child_stops_at_global_repair_limit(self):
        self.request.update(auto_repair=True, max_repairs=1)
        self.validation(self.queued(), passed=False)
        self.validation(self.next_repair(), passed=False)
        task = self.current()
        self.assertEqual(task['state'], 'failed')
        self.assertEqual(task['result']['lifecycle']['stop_reason'], 'repair_budget_exhausted')
        self.assertNotIn('child_workflow_id', task['result']['lifecycle'])
        self.assertEqual(len(self.jobs.db.rows(tasks)), 2)

    def test_environment_failure_does_not_trigger_source_repair(self):
        self.request['auto_repair'] = True
        self.validation(self.queued(), passed=False, collect=False)
        diagnosis = SimpleNamespace(source_repair_allowed=False, recommended_action='Ask the operator to restore the environment.')
        with patch.object(domain, 'diagnose_failure_run', return_value=diagnosis):
            task = self.action('collect-validation')
        self.assertEqual(task['result']['lifecycle']['stop_reason'], 'not_source_repairable')
        self.assertEqual(len(self.jobs.db.rows(tasks)), 1)

    def test_child_transaction_rollback_then_collection_retry_never_retests(self):
        self.request['auto_repair'] = True
        self.validation(self.queued(), passed=False, collect=False)
        original = self.lifecycle._checkpoint
        def fail_after_child(task, action, data, **kwargs):
            if action == 'repair_queued':
                raise RuntimeError('synthetic transaction failure')
            return original(task, action, data, **kwargs)
        with patch.object(self.lifecycle, '_checkpoint', side_effect=fail_after_child), self.assertRaises(RuntimeError):
            self.action('collect-validation')
        self.assertEqual(self.current()['state'], 'waiting_validation')
        self.assertEqual(len(self.jobs.db.rows(tasks)), 1)
        with patch.object(domain, 'run_operator', side_effect=AssertionError('do not test twice')):
            self.action('collect-validation')
        self.assertEqual(len(self.jobs.db.rows(tasks)), 2)

    def test_config_scope_and_default_off(self):
        self.assertTrue(self.client.get('/api/operator-workflows/config').json()['enabled'])
        self.assertEqual(self.client.get('/api/operator-workflows', params={'session_id': 'unknown'}).status_code, 404)
        self.cfg['queue']['proposalWorkflows'] = False
        with patch.dict(os.environ, {'TRITON_RISCV_CONFIG': json.dumps(self.cfg)}):
            self.assertFalse(self.client.get('/api/operator-workflows/config').json()['enabled'])
            response = self.client.post('/api/operator-workflows', json={'request': {**self.request, 'auto_publish': True}, 'request_id': 'disabled'})
            self.assertEqual(response.status_code, 403)
        self.assertFalse(self.backend.calls)

    def test_missing_review_artifact_keeps_state_visible_and_blocks_controls(self):
        self.proposal()
        with patch.object(self.lifecycle, '_bundle', side_effect=ValueError('bad artifact')):
            task = self.lifecycle.view(self.workflow_id)
        self.assertEqual(task['state'], 'waiting_approval')
        self.assertIn('ValueError', task['artifact_error'])


if __name__ == '__main__':
    unittest.main()
