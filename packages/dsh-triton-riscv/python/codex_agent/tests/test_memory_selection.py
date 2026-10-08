"""Synthetic functional fixtures, not business evaluation samples."""
import asyncio
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_agent.memory import MemoryStore, MemoryQuery
from codex_agent.memory_selection import diversity_bucket, select_candidates
from codex_agent.tests.test_memory import record


def candidate(index, **overrides):
    row = {"id":index, "operator":"alpha", "memory_type":"failure-diagnosis",
           "outcome":"failed", "failure_stage":"llvm-ir", "error_signature":"sig-a",
           "environment":{"architecture":"riscv64"}, "source_run":f"run-{index}",
           "evidence":{"chain":{"items":[{"kind":"error", "state":"observed-error",
                 "source":f"log-{index}", "run_id":f"run-{index}", "attempt":index, "text":"error"}]}},
           "retrieval":{"score":1-index/100}}
    row.update(overrides)
    return row


def choose(rows, limit=5, **kwargs):
    return select_candidates(rows, limit, strategy="evidence-diverse-v1", **kwargs)


def stored(index, **kwargs):
    return record(source_run=f'run-{index}', evidence={'chain':{
        'schema':'evidence-chain-v1', 'identity':f'run-{index}', 'items':[]}}, **kwargs)


class SelectionTests(unittest.TestCase):
    def test_default_preserves_order_objects_and_inputs(self):
        values = [candidate(i) for i in range(7)]
        before = deepcopy(values)
        result = select_candidates(values)
        self.assertEqual(result, values[:5])
        self.assertIs(result[0], values[0])
        self.assertEqual(values, before)

    def test_repeated_roles_deferred_without_merging_sources(self):
        values = [candidate(i) for i in range(6)] + [candidate(7, operator="beta")]
        trace = {}
        result = choose(values, trace=trace)
        self.assertEqual([r['id'] for r in result], [0, 1, 2, 3, 7])
        self.assertFalse(trace['source_records_merged'])
        self.assertEqual(result[-1]['evidence']['chain']['items'][0]['attempt'], 7)
        self.assertEqual(values[4]['source_run'], 'run-4')

    def test_complementary_actions_kept_and_recommendation_not_applied(self):
        values = [candidate(i) for i in range(8)]
        values[1]['evidence']['chain']['items'].append({'kind':'action','state':'recommended'})
        values[2]['evidence']['chain']['items'].append({'kind':'action','state':'applied'})
        values[-1]['operator'] = 'beta'
        result = choose(values)
        self.assertEqual([r['id'] for r in result], [0, 1, 2, 3, 7])
        self.assertEqual(result[1]['evidence']['chain']['items'][-1]['state'], 'recommended')

    def test_same_operator_different_fault_or_environment_not_same_bucket(self):
        a = candidate(1)
        for overrides in ({'error_signature':'sig-b'}, {'failure_stage':'runtime'},
                          {'environment':{'architecture':'riscv64','triton':'new'}},
                          {'outcome':'passed'}, {'memory_type':'failed-repair'}):
            b = candidate(2, **overrides)
            self.assertNotEqual(diversity_bucket(a), diversity_bucket(b))
        self.assertNotIn('family', a)

    def test_missing_fields_and_small_or_empty_pool_fall_back(self):
        for field, value in [('operator',''), ('environment',{}), ('error_signature',None), ('failure_stage',None)]:
            values = [candidate(i, **{field:value}) for i in range(7)]
            self.assertEqual(choose(values), values[:5])
        for n in range(6):
            values = [candidate(i) for i in range(n)]
            self.assertEqual(choose(values), values)
        self.assertEqual(choose([candidate(1)], 0), [])
        with self.assertRaises(ValueError): select_candidates([], strategy='automatic')

    def test_no_unconditional_success_priority(self):
        values = [candidate(i, error_signature=f'sig-{i}') for i in range(6)]
        values.append(candidate(9, outcome='passed'))
        self.assertEqual([r['id'] for r in choose(values)], list(range(5)))

    def test_stable_score_ties_filters_and_unchanged_last_used_on_ranking(self):
        with tempfile.TemporaryDirectory() as temp, MemoryStore(Path(temp)/'db') as store:
            ids = [store.add(stored(i, environment={'architecture':'riscv64'}))[0] for i in range(8)]
            store.archive(ids[1], 'inactive')
            q = MemoryQuery(operator='tanh_and_mul', semantics='', pytorch_reference='')
            ranked = store.rank_candidates(q, exclude_source_runs=('run-2',))
            self.assertEqual([r['id'] for r in ranked], [ids[i] for i in (0,3,4,5,6,7)])
            self.assertTrue(all(r['last_used_at'] is None for r in store.list()))
            for strategy in ('record-top5','evidence-diverse-v1'):
                selected = store.retrieve(q, exclude_source_runs=('run-2',), candidate_strategy=strategy)
                self.assertEqual([r['id'] for r in selected], [ids[i] for i in (0,3,4,5,6)])

    def test_all_below_threshold_and_empty_database(self):
        with tempfile.TemporaryDirectory() as temp, MemoryStore(Path(temp)/'db') as store:
            q = MemoryQuery(operator='unrelated', semantics='', pytorch_reference='', environment={'architecture':'different'})
            self.assertEqual(store.retrieve(q, candidate_strategy='evidence-diverse-v1'), [])
            store.add(record(operator='abc', semantics='', pytorch_reference='', summary='',
                             tl_ops=[], confidence_grade='D', outcome='failed', environment={'architecture':'riscv64'}))
            self.assertEqual(store.rank_candidates(q), [])
            self.assertEqual(store.retrieve(q, candidate_strategy='evidence-diverse-v1'), [])

    def test_actual_mcp_selection_parameter_default_and_opt_in(self):
        from mcp import Client
        from codex_agent.harness.mcp_server import server
        async def check():
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); db = root/'memory.sqlite3'
                with MemoryStore(db) as store:
                    for i in range(6): store.add(stored(i, operator='alpha'))
                    beta, _ = store.add(stored('beta', operator='beta'))
                env = {'TRITON_RISCV_REPO_ROOT':str(root), 'TRITON_RISCV_MEMORY_DB':str(db),
                       'TRITON_RISCV_EMBEDDING_PROVIDER':'none', 'TRITON_RISCV_MEMORY_RETRIEVAL_MODE':'legacy'}
                with patch.dict(os.environ, env):
                    os.environ.pop('TRITON_RISCV_MEMORY_CONTEXT_FORMAT', None)
                    async with Client(server) as client:
                        listed = await client.list_tools()
                        tool = next(t for t in listed.tools if t.name == 'retrieve_operator_memory')
                        prop = tool.input_schema['properties']['candidate_strategy']
                        self.assertEqual(prop['default'], 'record-top5')
                        a = await client.call_tool('retrieve_operator_memory', {'operator_name':'alpha'})
                        b = await client.call_tool('retrieve_operator_memory', {'operator_name':'alpha', 'candidate_strategy':'evidence-diverse-v1'})
                        invalid = await client.call_tool('retrieve_operator_memory', {'operator_name':'alpha', 'candidate_strategy':'invalid'})
                self.assertFalse(a.is_error)
                self.assertFalse(b.is_error)
                self.assertTrue(invalid.is_error)
                self.assertNotIn(beta, [r['memory_id'] for r in a.structured_content['items']])
                self.assertIn(beta, [r['memory_id'] for r in b.structured_content['items']])
                self.assertLessEqual(len(b.structured_content['items']), 5)
                self.assertFalse(b.structured_content['context_excerpt'].startswith('{'))
                self.assertLessEqual(len(b.structured_content['context_excerpt']), 6000)
        asyncio.run(check())


if __name__ == '__main__': unittest.main()
