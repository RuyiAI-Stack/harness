import { beforeAll, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
let result: any

beforeAll(() => {
  for (const key of ['TRITON_MYSQL_URL', 'TRITON_AUTH_REDIS_URL', 'TRITON_TEST_QDRANT_URL'])
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
    ['-I', '-B', join(root, 'tests/integration/rag-catalog-fixture.py')],
    {
      encoding: 'utf8',
      env: { ...env, TRITON_AMQP_URL: 'amqp://guest:guest@127.0.0.1:1' },
      timeout: 180000,
      maxBuffer: 8 * 1024 * 1024,
    },
  )
  expect(output.error, output.stderr).toBeUndefined()
  expect(output.status, output.stdout + output.stderr).toBe(0)
  result = JSON.parse(output.stdout)
}, 190000)

it('lists only shared knowledge and excludes both administrator and member private workspaces', () => {
  expect(result.catalog.authorized_sources).toBe(1)
  expect(result.catalog.private_scope_excluded).toBe(true)
  expect(result.catalog.admin_private_scope_excluded).toBe(true)
})

it('keeps published v1 readable while v2 is a draft and retrieves the original historical chunks', () => {
  expect(result.catalog.serving_v1_draft_v2).toBe(true)
  expect(result.catalog.historical_chunks_readable).toBe(true)
})

it('keeps the internal memory lifecycle and idempotent receipts independent of the shared admin catalog', () => {
  expect(result.memory.correction_preserves_original).toBe(true)
  expect(result.memory.one_note_after_retry).toBe(true)
  expect(result.memory.receipt_committed).toBe(true)
})

it('excludes archived cases from real Qdrant hydration before the cleanup worker runs and invalidates cache', () => {
  expect(result.memory.archive_excludes_before_worker).toBe(true)
  expect(result.memory.cache_invalidated).toBe(true)
})

it('cleans both exact vector versions, retains source text, and restores through a new indexable version', () => {
  expect(result.cleanup).toEqual({
    verified_old_point_count: 0,
    versions_processed: 2,
    originals_retained: true,
    stale_version_reuse_rejected: true,
    restored_new_v3_retrievable: true,
  })
  expect(result.passed).toBe(true)
  expect(result.embedding).toBe('deterministic synthetic vectors; no external model')
})

it('checks actual scoped Qdrant counts, distinguishes outages, and classifies editable imported sources', () => {
  expect(result.vectors).toEqual({
    actual_count_verified: true,
    draft_not_claimed_indexed: true,
    private_and_member_denied: true,
    outage_not_missing: true,
    origin_filter_matches_summary: true,
    deleted_vectors_read_back: true,
  })
})
