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
    ['-I', '-B', join(root, 'tests/integration/knowledge-images-fixture.py')],
    { env, encoding: 'utf8', timeout: 240000, maxBuffer: 8 * 1024 * 1024 },
  )
  expect(output.error, output.stderr).toBeUndefined()
  expect(output.status, output.stdout + output.stderr).toBe(0)
  r = JSON.parse(output.stdout)
}, 250000)

it('handles PNG/JPG/JPEG, orientation, paragraph boundaries and source coordinates', () => {
  expect(r.uppercase).toBe('jpeg')
  expect(r.orientation).toEqual([1200, 420])
  expect(r.preview_clean).toBe(true)
  for (const result of Object.values(r.formats) as any[]) {
    expect(result.state).toBe('review')
    expect(result.sections).toHaveLength(2)
    expect(result.sections[0].text).toContain('\nKeep 37 days')
    expect(result.sections[0].bbox).toEqual([40, 40, 300, 100])
    expect(result.sections[1].text).toContain('中文资料')
    expect(result.warnings.join(' ')).toContain('Low OCR confidence')
  }
})
it('rejects corrupted, disguised, oversized and animated images without inventing text', () => {
  for (const name of ['signature', 'truncated', 'disguised', 'pixels', 'animation']) expect(r[name]).toBe('rejected')
  expect(r.empty).toBe('ocr_empty')
  expect(r.timeout.state).toBe('parse_failed')
  expect(r.timeout.error).toContain('timed out')
  expect(r.timeout_preview).toBe(true)
  expect(r.missing_engine).toContain('install Tesseract')
  expect(r.missing_chinese).toContain('chi_sim and eng')
  expect(r.preview_failure).toEqual({
    state: 'parse_failed',
    error: 'Unable to save image preview; contact the maintainer and upload again',
    published: null,
    chunks: [],
  })
})
it('enforces admin identity, CSRF, upload limits, preview permissions and source versions', () => {
  expect([r.anonymous, r.member, r.csrf, r.oversize]).toEqual([401, 403, 403, 413])
  expect([r.member_preview, r.wrong_version, r.catalog_member, r.catalog_wrong_version]).toEqual([403, 404, 403, 404])
  expect(r.catalog_preview).toBe(200)
  expect(r.preview).toEqual({ status: 200, mime: 'image/png', cache: 'no-store', clean: true })
})
it('deduplicates uploads and gates indexing/publication on explicit review', () => {
  expect(r.upload).toBe(202)
  expect(r.duplicate).toBe(true)
  expect(r.replay_conflict).toBe(409)
  expect(r.before_review).toEqual({ state: 'review', chunks: [], published: null })
  expect([r.early_index, r.early_publish]).toEqual([409, 409])
  expect(r.original_exact).toBe(true)
})
it('preserves immutable originals, OCR locations and reviewed edits in new versions', () => {
  expect(r.saved).toBe(201)
  expect(r.forged_sections).toBe(409)
  expect(r.stale_edit).toBe(409)
  expect(r.version_retained).toBe(200)
  expect(r.edited.edited).toBe(true)
  expect(r.edited.locator).toContain('bbox')
  expect(r.edited.text).toContain('REVIEW-842')
})
it.runIf(process.env.TRITON_TEST_OCR === '1')(
  'executes real local OCR in the isolated parser on all image extensions',
  () => {
    expect(r.real_blank).toBe('ocr_empty')
    for (const fmt of ['png', 'jpg', 'jpeg']) {
      const result = r[`real_${fmt}`]
      expect(result.state, result.error).toBe('review')
      const text = result.sections.map((s: any) => s.text).join('\n')
      expect(text).toContain('OCR-731')
      if (process.env.TRITON_TEST_CJK_FONT) expect(text).toContain('恢复前检查备份')
    }
  },
)
it.runIf(process.env.TRITON_TEST_OCR === '1' && Boolean(process.env.TRITON_TEST_EMBEDDING_URL))(
  'runs RabbitMQ OCR, reviewed local embeddings, Qdrant retrieval, chat evidence and delete/restore gates',
  () => {
    expect(r.async_parse).toBe('review')
    expect(r.confirmed).toBe(202)
    expect(r.async_publish.sync.state, r.async_publish.sync.error).toBe('published')
    expect(r.async_publish.published).toBe(2)
    expect(r.async_publish.chunks.every((c: any) => c.locator.includes('bbox'))).toBe(true)
    expect(r.chat_context).toContain('OCR-731')
    expect(r.chat_context).toContain('bbox')
    expect(r.chat_context).toContain('untrusted reference data')
    expect(r.delete).toBe(200)
    expect(r.after_delete.items).toEqual([])
    expect(r.restored_unpublished).toBe(true)
  },
)
