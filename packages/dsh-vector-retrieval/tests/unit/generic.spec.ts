import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync } from 'node:fs'
import { probe } from '../probe'
import { bridgeEnvironment, resolveConfig } from '../../lib/config.js'
import { createVectorRetrieval } from '../../lib/service.js'

const config = {
  python: '/tmp/python',
  workspaceId: 'support-a',
  generation: 'faq-v1',
  fingerprint: 'fp',
  collection: 'faq',
  dimensions: 4,
}

describe('independent document retrieval', () => {
  it('indexes non-operator documents with deterministic scope-bound IDs and rejects foreign results', () => {
    const value = probe(`
import json, sys
from dsh_vector_retrieval import DocumentChunk, Scope, VectorIndex, Qdrant
from unittest.mock import Mock
q = Qdrant(dict(qdrantUrl='http://127.0.0.1:1', timeoutSeconds=1, dimensions=4, apiKeyEnv='FIXTURE_KEY'), 'faq')
q.ensure = Mock()
q.upsert = Mock()
a = VectorIndex(q, Scope('support-a', 'faq-v1', 'fp'))
b = VectorIndex(q, Scope('support-b', 'faq-v1', 'fp'))
chunk = DocumentChunk('returns-policy', 1, 'paragraph-1', 'source-sha', 'Returns within 30 days.', {'workspace_id': 'support-b'})
first = a.index_chunks([chunk], vectors=[[1, 0, 0, 0]])
point = q.upsert.call_args.args[0][0]
again = a.index_chunks([chunk], vectors=[[1, 0, 0, 0]])
other = b.index_chunks([chunk], vectors=[[1, 0, 0, 0]])
foreign = q.upsert.call_args.args[0][0]
q.query = Mock(return_value=[{**point, 'score': 0.8}, {**foreign, 'score': 1.0}])
hits = a.search([1, 0, 0, 0])
q.delete_version = Mock()
a.delete_version('returns-policy', 1)
print(json.dumps(dict(first=first, again=again, other=other, hits=hits,
    deletion=list(q.delete_version.call_args.args), scope=q.delete_version.call_args.kwargs,
    imports=[name for name in sys.modules if name.startswith('codex_agent')])) )
`)
    expect(value.first).toEqual(value.again)
    expect(value.first).not.toEqual(value.other)
    expect(value.hits).toHaveLength(1)
    expect(value.hits[0].payload).toMatchObject({
      workspace_id: 'support-a',
      document_id: 'returns-policy',
      version: 1,
      text: 'Returns within 30 days.',
      source_version: 'source-sha',
    })
    expect(value.deletion).toEqual(['support-a', 'returns-policy', 1])
    expect(value.scope).toEqual({ older: false, include: false, generation: 'faq-v1', fingerprint: 'fp' })
    expect(value.imports).toEqual([])
  })

  it('preserves provider ordering, native token counting and pinned local models', () => {
    const result = probe(`
import json, os
from unittest.mock import MagicMock, patch
from dsh_vector_retrieval.embeddings import *
from dsh_vector_retrieval.embeddings import _sentence_encoder
response = MagicMock()
response.__enter__.return_value = response
response.read.return_value = json.dumps({'data': [{'index': 1, 'embedding': [0, 1]}, {'index': 0, 'embedding': [1, 0]}]}).encode()
os.environ['FIXTURE_KEY'] = 'synthetic-secret'
os.environ['AGENT_EMBEDDING_MODEL'] = 'ambient-model'
p = build_embedding_provider('openai-compatible', model='configured-model', base_url='https://example.invalid/v1', api_key_env='FIXTURE_KEY', use_environment=False)
with patch('urllib.request.urlopen', return_value=response) as call:
    vectors = p.embed(['first', 'second'])
    request = json.loads(call.call_args.args[0].data)
response.read.return_value = json.dumps({'embeddings': [[1, 0]], 'prompt_eval_count': 7}).encode()
o = OllamaEmbeddingProvider(model='fixture')
with patch('urllib.request.urlopen', return_value=response) as call:
    counts = [o.count_tokens('same'), o.count_tokens('same')]
    calls = call.call_count
    body = json.loads(call.call_args.args[0].data)
fake = MagicMock()
with patch.dict('sys.modules', {'sentence_transformers': fake}):
    build_embedding_provider('sentence-transformers', model='fixture-model', revision='pinned-revision')
print(json.dumps(dict(vectors=vectors, request=request, counts=counts, calls=calls, body=body,
    revision=fake.SentenceTransformer.call_args.kwargs,
    cosine=[cosine_similarity([1,0],[1,0]), cosine_similarity([1,0],[0,1]), cosine_similarity([],[])])))
`)
    expect(result.vectors).toEqual([
      [1, 0],
      [0, 1],
    ])
    expect(result.request).toEqual({ model: 'configured-model', input: ['first', 'second'] })
    expect(result.counts).toEqual([7, 7])
    expect(result.calls).toBe(1)
    expect(result.body.truncate).toBe(true)
    expect(result.revision).toEqual({
      revision: 'pinned-revision',
      trust_remote_code: false,
      model_kwargs: { weights_only: true },
    })
    expect(result.cosine).toEqual([1, 0, 0])
  })

  it('scopes Qdrant requests, caps candidates and fails closed before unscoped reads/deletes', () => {
    const result = probe(`
import json
from unittest.mock import Mock
from dsh_vector_retrieval import Qdrant
q = Qdrant(dict(qdrantUrl='http://127.0.0.1:1', timeoutSeconds=1, dimensions=4, apiKeyEnv='FIXTURE_KEY'), 'faq')
q.request = Mock(return_value={'points': []})
q.query([1, 0, 0, 0], 'ws', 'gen', 'fp', limit=100, offset=100)
query = q.request.call_args.args[2]
q.request = Mock(return_value={'status': 'completed'})
q.delete_version('ws', 'faq-42', 2, older=True, include=True)
deletion = q.request.call_args.args[2]
q.request.reset_mock()
errors = []
for call in [lambda: q.query([1,0,0,0], '', 'gen', 'fp', limit=10),
             lambda: q.query([1,0,0,0], 'ws', '', 'fp', limit=10),
             lambda: q.query([1,0,0,0], 'ws', 'gen', 'fp', limit=100, offset=101),
             lambda: q.query([1,0,0,0], 'ws', 'gen', 'fp', limit=True),
             lambda: q.delete_version(None, 'faq-42', 2)]:
    try: call(); errors.append(None)
    except Exception as e: errors.append(type(e).__name__)
print(json.dumps(dict(query=query, deletion=deletion, errors=errors, calls=q.request.call_count)))
`)
    expect(result.query).toMatchObject({ limit: 100, offset: 100, with_vector: false, with_payload: true })
    expect(result.query.filter.must).toEqual([
      { key: 'workspace_id', match: { value: 'ws' } },
      { key: 'generation', match: { value: 'gen' } },
      { key: 'index_fingerprint', match: { value: 'fp' } },
    ])
    expect(result.deletion.filter.must).toEqual([
      { key: 'workspace_id', match: { value: 'ws' } },
      { key: 'document_id', match: { value: 'faq-42' } },
      { key: 'version', range: { lte: 2 } },
    ])
    expect(result.errors).toEqual(Array(5).fill('ValueError'))
    expect(result.calls).toBe(0)
  })

  it('contains no product imports or business rules in its Python distribution', () => {
    const directory = new URL('../../python/dsh_vector_retrieval/', import.meta.url)
    for (const file of readdirSync(directory).filter(name => name.endsWith('.py'))) {
      expect(readFileSync(new URL(file, directory), 'utf8')).not.toMatch(/codex_agent|memory_id|case_version|operator/)
    }
  })
})

describe('trusted service configuration', () => {
  it('accepts secrets only via named environment variables and ignores ambient model config', () => {
    const value = resolveConfig(config)
    const env = bridgeEnvironment(value, {
      DSH_VECTOR_API_KEY: 'fixture-key',
      AGENT_EMBEDDING_MODEL: 'ambient',
      TRITON_MYSQL_URL: 'mysql://private',
      PYTHONPATH: '/untrusted',
      OTHER_SECRET: 'hidden',
    })
    expect(env.DSH_VECTOR_API_KEY).toBe('fixture-key')
    expect(JSON.stringify(env)).not.toMatch(/mysql|ambient|untrusted|hidden/)
    expect(JSON.parse(env.DSH_VECTOR_RETRIEVAL_CONFIG).workspaceId).toBe('support-a')
  })

  it.each([
    { workspaceId: '' },
    { python: 'python3' },
    { apiKey: 'secret' },
    { collection: '../other' },
    { qdrantUrl: 'https://user:secret@example.invalid' },
    { qdrantUrl: 'https://example.invalid?key=secret' },
    { embedding: { apiKey: 'secret' } },
    { timeoutMs: -1 },
    { dimensions: 0 },
  ])('rejects unsafe trusted settings without echoing them: %j', override => {
    expect(() => resolveConfig({ ...config, ...override })).toThrow()
    try {
      resolveConfig({ ...config, ...override })
    } catch (error) {
      expect(String(error)).not.toContain('secret')
    }
  })

  it('refuses model-supplied scope/config and disposes without launching a process', async () => {
    const client = createVectorRetrieval(config)
    expect(() => client.search({ vector: [1, 0, 0, 0], workspaceId: 'other' })).toThrow()
    expect(() => client.search({ vector: [1, 0, 0, 0] }, { generation: 'other' })).toThrow()
    expect(() => client.deleteVersion('faq-42', 1, { workspaceId: 'other' })).toThrow()
    expect(() => client.index([], { apiKeyEnv: 'OTHER_SECRET' })).toThrow()
    client.dispose()
    await expect(client.search({ vector: [1, 0, 0, 0] })).rejects.toThrow('disposed')
  })
})
