import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { loadLocalEnvironment } from '../../scripts/local-environment.mjs'

const roots: string[] = []
function fixture(model?: string, local?: Record<string, unknown>) {
  const root = mkdtempSync(join(tmpdir(), 'triton-launcher-env-'))
  roots.push(root)
  mkdirSync(join(root, '.state'))
  if (model !== undefined) writeFileSync(join(root, '.state/model.env'), model, { mode: 0o600 })
  if (local !== undefined) writeFileSync(join(root, '.state/native-local.json'), JSON.stringify(local))
  return root
}
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

describe('shared local launcher environment', () => {
  it('preserves the existing environment when local files are absent', () => {
    const ambient = { DSH_MODEL: 'old-model', PATH: '/bin' }
    const result = loadLocalEnvironment(fixture(), ambient)
    expect(result).toEqual(ambient)
    expect(result).not.toBe(ambient)
  })
  it('loads existing target configuration without overriding shell settings', () => {
    const result = loadLocalEnvironment(fixture(undefined, {
      TRITON_RISCV_REPO_ROOT: '/operators', RISCV_HOST: 'sg2044', DSH_MODEL: 'saved-model',
    }), { DSH_MODEL: 'shell-model' })
    expect(result).toEqual({ TRITON_RISCV_REPO_ROOT: '/operators', RISCV_HOST: 'sg2044', DSH_MODEL: 'shell-model' })
  })
  it('uses the explicit model file instead of stale model and key exports', () => {
    const ambient = { DSH_MODEL: 'AZ-GPT-5.6-Sol', ISRC_API_KEY: 'old-test-key' }
    const result = loadLocalEnvironment(fixture(
      'DSH_MODEL=gpt-6-astra\nISRC_BASE_URL=https://model.example/v1\nISRC_API_KEY="new-test-key"\n',
      { DSH_MODEL: 'gpt-5.6-sol' },
    ), ambient)
    expect(result.DSH_MODEL).toBe('gpt-6-astra')
    expect(result.ISRC_API_KEY).toBe('new-test-key')
    expect(result.ISRC_BASE_URL).toBe('https://model.example/v1')
    expect(ambient.ISRC_API_KEY).toBe('old-test-key')
  })
  it('supports comments and quoted values without executing shell expressions', () => {
    const result = loadLocalEnvironment(fixture(
      '# local values only\nDSH_MODEL="gpt-6-astra"\nISRC_API_KEY=literal-$(whoami)\n',
    ), {})
    expect(result.ISRC_API_KEY).toBe('literal-$(whoami)')
  })
  it('rejects an empty key rather than silently reusing an old credential', () => {
    expect(() => loadLocalEnvironment(fixture('ISRC_API_KEY=\n'), { ISRC_API_KEY: 'old-test-key' }))
      .toThrow('Fill ISRC_API_KEY')
  })
  it('does not disclose malformed credentials in errors', () => {
    const secret = 'private test secret'
    let message = ''
    try { loadLocalEnvironment(fixture(`ISRC_API_KEY="${secret}"\n`), {}) }
    catch (error) { message = (error as Error).message }
    expect(message).toContain('raw key only')
    expect(message).not.toContain(secret)
  })
  it('rejects empty model names and unrelated permissions in the model file', () => {
    expect(() => loadLocalEnvironment(fixture('DSH_MODEL=\n'), {})).toThrow('Fill DSH_MODEL')
    expect(() => loadLocalEnvironment(fixture('TRITON_RISCV_ALLOW_REPAIR_APPLY=1\n'), {}))
      .toThrow('Unsupported setting')
  })
  it('still rejects credentials or unknown settings in native-local.json', () => {
    expect(() => loadLocalEnvironment(fixture(undefined, { ISRC_API_KEY: 'test-only' }), {}))
      .toThrow('Invalid local configuration key')
  })
})
