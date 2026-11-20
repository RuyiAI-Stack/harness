import { beforeAll, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
let result: any
beforeAll(() => {
  for (const key of ['TRITON_MYSQL_URL', 'TRITON_AUTH_REDIS_URL', 'TRITON_TEST_AMQP_URL', 'TRITON_TEST_QDRANT_URL'])
    expect(process.env[key], `${key} must target disposable services`).toBeTruthy()
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) =>
        !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) &&
        !/^(DSH_|RISCV_|TRITON_RISCV_|TRITON_WORKBENCH_|TRITON_SHARED_)/.test(key),
    ),
  )
  const output = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python'),
    ['-I', '-B', join(root, 'tests/integration/knowledge-fixture.py')],
    {
      encoding: 'utf8',
      env: { ...env, TRITON_AMQP_URL: process.env.TRITON_TEST_AMQP_URL },
      timeout: 180000,
      maxBuffer: 8 * 1024 * 1024,
    },
  )
  expect(output.error, output.stderr).toBeUndefined()
  expect(output.status, output.stdout + output.stderr).toBe(0)
  result = JSON.parse(output.stdout)
}, 190000)

it('requires authenticated explicit admin identity, not a browser role header', () => {
  expect(result.anonymous).toBe(401)
  expect(result.regular_admin).toBe(403)
  expect(result.regular_write).toBe(403)
  expect(result.regular_flag).toBe(false)
  expect(result.admin_flag).toBe(true)
  expect(result.disabled_admin).toBe(401)
})
it('enforces CSRF, origin, bounded document bodies and immutable versions', () => {
  expect(result.missing_csrf).toBe(403)
  expect(result.bad_origin).toBe(403)
  expect(result.oversize).toBe(422)
  expect(result.update_version).toBe(2)
  expect(result.stale_update).toBe(409)
  expect(result.stale_replay).toBe(409)
  expect(result.versions_retained).toBe(true)
})
it('keeps drafts hidden and rejects publication without a completed index', () => {
  expect(result.save_status).toBe(201)
  expect(result.draft.state).toBe('draft')
  expect(result.early_publish).toBe(409)
  expect(result.draft_hidden).toBe(404)
  expect(result.draft_search).toBe('empty')
})
it('deduplicates save/index requests and consumes a real broker delivery', () => {
  expect(result.repeat_save_same_id).toBe(true)
  expect(result.request_conflict).toBe(409)
  expect(result.index_accepted).toBe(202)
  expect(result.indexed).toBe('ready')
  expect(result.duplicate_claim).toBe(true)
  expect(result.one_outbox).toBe(true)
})
it('returns published evidence from real Qdrant and delivers it to the Harness model boundary', () => {
  expect(result.published).toBe(1)
  expect(result.retrieval.status).toBe('ready')
  expect(result.retrieval.text).toContain('SILVER-271')
  expect(result.model_boundary).toEqual({ has_evidence: true, untrusted_marked: true, completed: true })
})
it('keeps the old published version when an update or embedding call fails', () => {
  expect(result.old_publication_preserved).toBe(true)
  expect(result.failed_draft).toBe('failed')
  expect(result.failed_publish).toBe(409)
  expect(result.failure_preserves_old).toBe(true)
})
it('retries a transient index failure and distinguishes outages from empty knowledge', () => {
  expect(result.index_retry).toBe(true)
  expect(result.unavailable_status).toBe(503)
  expect(result.unavailable_chat).toBe(true)
  expect(result.configuration_mismatch).toBe(true)
})
it('blocks unpublished vectors and cross-catalog data with SQL authority checks', () => {
  expect(result.unpublish).toBe(200)
  expect(result.unpublished_search).toBe('empty')
  expect(result.unpublished_document).toBe(404)
  expect(result.catalog_isolation).toBe(true)
})
it('preserves short and Chinese text and audits management without exposing secrets', () => {
  expect(result.short_chunk).toBe(true)
  expect(result.unicode_chunks).toBe(true)
  expect(result.audited).toBe(true)
  expect(result.overview_no_secrets).toBe(true)
  expect(result.overview.totals.users).toBeGreaterThanOrEqual(2)
})
it.runIf(Boolean(process.env.TRITON_TEST_EMBEDDING_URL))(
  'roundtrips real local embeddings through MySQL and Qdrant',
  () => {
    expect(result.real_embedding).toEqual({
      status: 'ready',
      count: 1,
      has_evidence: true,
      model: 'all-minilm:v2',
      dimensions: 384,
    })
  },
)
