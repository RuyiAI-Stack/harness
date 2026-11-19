import { beforeAll, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
let r: any
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
    ['-I', '-B', join(root, 'tests/integration/knowledge-files-fixture.py')],
    { encoding: 'utf8', env, timeout: 240000, maxBuffer: 8 * 1024 * 1024 },
  )
  expect(output.error, output.stderr).toBeUndefined()
  expect(output.status, output.stdout + output.stderr).toBe(0)
  r = JSON.parse(output.stdout)
}, 250000)

it('extracts PDF pages, DOCX document-order paragraphs, Chinese text and table rows', () => {
  expect(r.pdf.pages).toBe(2)
  expect(r.pdf.second.locator).toBe('PDF page 2')
  expect(r.pdf.second.text).toContain('PDF-428')
  expect(r.docx.sections.map((s: any) => s.text).join('\n')).toContain('中文资料')
  expect(r.docx.sections.at(-1).text).toBe('Service | Retention\nAtlas | 37 days')
  expect(r.docx.sections.at(-1).locator).toContain('table')
})
it('preserves long content without truncation and bounds preview section sizes', () => {
  expect(r.long.chars).toBeGreaterThan(60000)
  expect(r.long.largest).toBeLessThanOrEqual(20000)
  expect(r.long.distinct_ids).toBe(true)
})
it('reports scans, partial extraction, encryption, corruption and resource failures', () => {
  expect(r.scan).toBe('ocr_required')
  expect(r.mixed.state).toBe('review')
  expect(r.mixed.warnings.join(' ')).toContain('pages: 2')
  for (const name of ['encrypted', 'signature', 'corrupt', 'macro', 'zip_bomb', 'traversal', 'entity', 'linked_media'])
    expect(r[name], name).not.toBe('accepted')
  expect(r.timeout).toBe('parse_failed')
  expect(r.hash_mismatch.state).toBe('parse_failed')
  expect(r.hash_mismatch.error).toContain('hash mismatch')
})
it('keeps binary upload admin-only, CSRF-protected, same-origin and size bounded', () => {
  expect([r.anonymous, r.ordinary, r.csrf, r.origin, r.oversize]).toEqual([401, 403, 403, 403, 413])
  expect([r.bad_type, r.other_route_binary, r.extension, r.forged]).toEqual([415, 415, 422, 422])
  expect(r.ordinary_original).toBe(403)
})
it('deduplicates uploads, retains original bytes and cannot index an unreviewed parse', () => {
  expect(r.upload).toBe(202)
  expect(r.initial_state).toBe('parsing')
  expect(r.parsed_state).toBe('review')
  expect(r.duplicate).toBe(true)
  expect(r.conflict).toBe(409)
  expect(r.original).toEqual({ equal: true, attachment: true })
  expect(r.no_chunks_before_review).toBe(true)
  expect(r.file_source.reviewed).toBe(false)
  expect(r.early_index).toBe(409)
})
it('versions reviewed edits without forged sections or lost provenance', () => {
  expect(r.forged_sections).toBe(409)
  expect(r.review.state).toBe('draft')
  expect(r.review.latest_version).toBe(2)
  expect(r.review_replay).toBe(2)
  expect(r.stale_review).toBe(409)
  expect(r.indexed.source_file.reviewed).toBe(true)
  expect(r.indexed.sections[1].edited).toBe(true)
  expect(r.indexed.versions).toHaveLength(2)
})
it('indexes source-bounded chunks and sends cited evidence through the chat augmentation path', () => {
  expect(r.indexed.state).toBe('ready')
  expect(r.indexed.chunks.every((c: any) => c.locator.includes('PDF page'))).toBe(true)
  expect(r.published).toBe(2)
  expect(r.context).toContain('PDF-428')
  expect(r.context).toContain('PDF page 2')
  expect(r.context).toContain('reviewed edit')
  expect(r.context).toContain('untrusted reference data')
  expect(r.unpublished).toBe('empty')
})
it.runIf(Boolean(process.env.TRITON_TEST_EMBEDDING_URL))(
  'parses via RabbitMQ without embedding enabled, then indexes and retrieves with real local embeddings',
  () => {
    expect(r.async_docx).toBe('review')
    expect(r.async_index).toBe('ready')
    expect(r.real_retrieval.status).toBe('ready')
    expect(r.real_retrieval.context).toContain('DOCX-529')
    expect(r.real_retrieval.context).toContain('DOCX paragraph')
  },
)
