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
    ['-I', '-B', join(root, 'tests/integration/knowledge-management-fixture.py')],
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

it('lists the catalog with literal title/source search, Chinese, publication filters and totals', () => {
  expect(result.catalog).toEqual({
    total: 2,
    chinese: 1,
    source: 2,
    literal: 1,
    published: 1,
    unpublished: 1,
    next_page: [],
  })
})
it('enforces administrator identity, CSRF and optimistic version checks', () => {
  expect(result.regular_delete).toBe(403)
  expect(result.regular_list).toBe(403)
  expect(result.missing_csrf).toBe(403)
  expect(result.stale_delete).toBe(409)
})
it('removes publication and retrieval access but retains original contents and versions', () => {
  expect(result.before_delete).toContain('37 days')
  expect(result.delete).toEqual({ state: 'deleted', published_version: null, catalog_revision: 1 })
  expect(result.after_delete).toEqual([])
  expect(result.active_count).toBe(1)
  expect(result.trash_count).toBe(1)
  expect(result.preserved).toBe(true)
  expect(result.deleted_publish).toBe(404)
  expect(result.deleted_save).toBe(404)
  expect(result.deleted_public_read).toBe(404)
})
it('restores unpublished, accepts idempotent retries and rejects pre-restore requests', () => {
  expect(result.restore).toEqual({ state: 'draft', published_version: null, catalog_revision: 2 })
  expect(result.repeat_delete).toBe(200)
  expect(result.repeat_restore).toBe(200)
  expect(result.old_delete).toBe(409)
  expect(result.old_publish).toBe(404)
  expect(result.republish).toBe(200)
  expect(result.restored_search).toBe(true)
})
it('filters stale Qdrant evidence and leaves unrelated published documents available', () => {
  expect(result.stale_vectors_filtered).toBe(true)
  expect(result.other_document_survives).toBe(true)
})
it('cancels queued jobs and fences running parse/index commits across delete and restore', () => {
  expect(result.queued_cancelled).toBe('cancelled')
  expect(result.index_race.job).toBe('cancelled')
  expect(result.index_race.state).toBe('failed')
  expect(result.parse_race.job).toBe('cancelled')
  expect(result.parse_race.state).toBe('parse_failed')
  expect(result.parse_race.content).toBe('')
  expect(result.audit).toBe(true)
})
