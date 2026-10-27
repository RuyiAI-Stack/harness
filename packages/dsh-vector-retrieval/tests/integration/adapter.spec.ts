import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'
import { probe } from '../probe'

const driver = readFileSync(new URL('../adapter-probe.py', import.meta.url), 'utf8')
const run = (input = {}) => probe(driver, input, true)

it('hydrates original SQL evidence in its workspace and uses the extracted transport', () => {
  const value = run()
  expect(value.error).toBeUndefined()
  expect(value.items).toHaveLength(1)
  expect(value.items[0].matched_evidence).toMatchObject({
    text: 'exact original SQL evidence',
    chunk_id: 'existing-chunk',
    provenance: { source: { state: 'reported-not-proven' } },
  })
  expect(JSON.stringify(value.items)).not.toMatch(/foreign SQL|FORGED VECTOR/)
  expect(value.calls[0].workspace).toBe('workspace-a')
  expect(value.transport_module).toBe('dsh_vector_retrieval.qdrant')
})

it.each([
  ['workspace_id', 'workspace-b'],
  ['generation', 'wrong'],
  ['index_fingerprint', 'wrong'],
  ['case_version', 'wrong'],
  ['memory_id', 999],
  ['revision', 999],
  ['chunk_id', 'missing'],
])('rejects forged legacy payload %s', (field, value) => {
  const result = run({ scenario: 'forged', field, value })
  expect(result.items).toEqual([])
  expect(result.trace.rejected).toBeGreaterThan(0)
})

it.each(['pending', 'archived', 'changed-parent', 'changed-chunk', 'archived-between-pages'])(
  'retains source/version validation: %s',
  scenario => {
    expect(run({ scenario }).items).toEqual([])
  },
)

it('retains exclusion, candidate budget and index-switch behavior', () => {
  expect(run({ exclude: ['original-run'] }).items).toEqual([])
  const budget = run({ scenario: 'budget' })
  expect(budget.items).toHaveLength(1)
  expect(budget.trace).toMatchObject({ scanned: 200, candidate_budget: 200, budget_exhausted: true })
  expect(budget.calls.map((call: any) => call.offset)).toEqual([0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
  expect(run({ scenario: 'generation-switch' }).error).toBe('RetrievalUnavailable')
})

it.each([{ vector: [0, 0, 0, 0] }, { vector: [1, 2] }])(
  'never searches with invalid query embeddings %j',
  ({ vector }) => {
    const value = run({ vector })
    expect(value.error).toBe('RetrievalUnavailable')
    expect(value.calls).toEqual([])
  },
)

it('keeps legacy deletion filters and import compatibility without duplicate providers', () => {
  const value = probe(
    `
import json
from unittest.mock import Mock
from codex_agent.indexing.qdrant import Qdrant
import codex_agent.embeddings as old
import dsh_vector_retrieval.embeddings as new
q = Qdrant(dict(qdrantUrl='http://127.0.0.1:1', timeoutSeconds=1, dimensions=4, apiKeyEnv='FIXTURE_KEY'), 'existing_collection')
q.request = Mock(return_value={'status': 'completed'})
q.delete_revision('existing-workspace', 42, 3, older=True, include=True)
print(json.dumps(dict(body=q.request.call_args.args[2], same=old.build_embedding_provider is new.build_embedding_provider,
    cached=old._sentence_encoder is new._sentence_encoder)))
`,
    {},
    true,
  )
  expect(value.same).toBe(true)
  expect(value.cached).toBe(true)
  expect(value.body.filter.must).toEqual([
    { key: 'workspace_id', match: { value: 'existing-workspace' } },
    { key: 'memory_id', match: { value: 42 } },
    { key: 'revision', range: { lte: 3 } },
  ])
})
