# dsh-vector-retrieval

Independent Harness service and Python distribution (`dsh_vector_retrieval`).
Provides embedding providers and scoped Qdrant document/chunk/version operations.
It does not own chunking, source authorization, business ranking, or a source database.

## Install

Install this npm package with the host's Cordis and Schemastery peers. Run
`npm run build` and `npm run setup:backend`, or install `./python` into your own
Python 3.10+ environment. Optional extras: `[embedding-client]` for tokenizers,
`[embedding]` for sentence-transformers. No model is downloaded at installation.

Enable the patch entry only after setting trusted host configuration:

```yaml
- id: vector-retrieval
  name: dsh-vector-retrieval
  config:
    python: /absolute/venv/bin/python
    workspaceId: support-team-a
    generation: articles-v1
    fingerprint: articles-embedding-v1
    collection: articles
    qdrantUrl: http://127.0.0.1:6333
    dimensions: 384
    apiKeyEnv: DSH_VECTOR_API_KEY
```

Inject `vectorRetrieval` into another Cordis plugin. The service is bound to that
trusted configuration; never construct it from model/tool arguments. For multiple
workspaces use separately scoped host contexts, each with its own service instance.

```js
await ctx.vectorRetrieval.index(
  [
    {
      document_id: 'faq-42',
      version: 1,
      chunk_id: 'paragraph-1',
      source_version: 'sha256-of-source',
      text: 'Returns are accepted within 30 days.',
      metadata: { category: 'returns' },
    },
  ],
  { vectors: [vector] },
)
const hits = await ctx.vectorRetrieval.search({ vector }, { limit: 20 })
await ctx.vectorRetrieval.deleteVersion('faq-42', 1)
```

Configure `embedding` with `provider`, `model`, `baseUrl`, `apiKeyEnv`,
`tokenizerJson`, `tokenBudget`, and optional `revision` to use `embed(texts)`,
`search({text})`, or `index(chunks)` without supplied vectors. Credentials are
read from named environment variables only. Host calls accept an AbortSignal as
`options.signal`; subprocesses have bounded input/output, timeouts, and safe errors.

Python callers can use `Qdrant`, `Scope`, `DocumentChunk`, and `VectorIndex`.
The CLI (`python -I -m dsh_vector_retrieval`) reads one JSON action from stdin and
trusted configuration from `DSH_VECTOR_RETRIEVAL_CONFIG`, returning one JSON result.
Pages are capped at 100 hits within a 200-candidate window. Upserts have deterministic
scope-bound IDs; callers manage version activation and recheck source authority.
Deletes require a workspace. The scoped API also filters generation/fingerprint.
Raw `Qdrant` is a trusted transport, not an untrusted tool interface.

Triton retains its legacy payload mapping, point IDs, chunking, lineage, MySQL
hydration and case validation. Existing collections need no migration or reindex.
Tests use Vitest with local Python probes; synthetic vectors do not establish
production retrieval quality. Run `npm test` with `VECTOR_PLUGIN_TEST_PYTHON`
(or `DSH_PLUGIN_TEST_PYTHON`) set to an environment containing the installed
distribution. Adapter tests additionally need the Triton and multi-agent packages;
they use in-memory SQL, not a production database.
