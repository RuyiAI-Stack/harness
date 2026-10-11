import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { loadLocalEnvironment } from '../../scripts/local-environment.mjs'
import { configFromEnvironment } from '../../lib/config.js'

const roots: string[] = []
function fixture(model?: string, local?: Record<string, unknown>) {
  const root = mkdtempSync(join(tmpdir(), 'triton-launcher-env-'))
  roots.push(root)
  mkdirSync(join(root, '.state'))
  if (model !== undefined) writeFileSync(join(root, '.state/model.env'), model, { mode: 0o600 })
  if (local !== undefined) writeFileSync(join(root, '.state/native-local.json'), JSON.stringify(local))
  return root
}
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

describe('shared local launcher environment', () => {
  it('shares structured retrieval settings between workbench and native launcher', () => {
    const root = fixture()
    writeFileSync(
      join(root, '.state/runtime.json'),
      JSON.stringify({
        schemaVersion: 1,
        repoRoot: '/operators',
        stateDir: '/operator-state',
        memory: { retrievalMode: 'qdrant' },
      }),
    )
    writeFileSync(join(root, '.state/services.env'), 'TRITON_MYSQL_URL=mysql+pymysql://test@localhost/test\n')
    const env = loadLocalEnvironment(root, {})
    expect(env.TRITON_RISCV_REPO_ROOT).toBe('/operators')
    expect(JSON.parse(env.TRITON_RISCV_CONFIG).memory.retrievalMode).toBe('qdrant')
    expect(configFromEnvironment(env).memory.retrievalMode).toBe('qdrant')
    expect(env.TRITON_MYSQL_URL).toContain('test@localhost')
  })
  it('rejects broken structured config without disclosing its contents', () => {
    const root = fixture()
    writeFileSync(join(root, '.state/runtime.json'), 'private-invalid-data')
    expect(() => loadLocalEnvironment(root, {})).toThrow('Invalid .state/runtime.json')
    expect(() => configFromEnvironment({ TRITON_RISCV_CONFIG: 'private-invalid-data' })).toThrow(
      'Invalid TRITON_RISCV_CONFIG',
    )
    writeFileSync(join(root, '.state/services.env'), 'PATH=/untrusted')
    expect(() => loadLocalEnvironment(root, {})).toThrow('Unsupported setting')
  })
  it('does not override explicit service credentials or runtime documents', () => {
    const root = fixture()
    writeFileSync(join(root, '.state/services.env'), 'TRITON_MYSQL_URL=local-only')
    writeFileSync(join(root, '.state/runtime.json'), 'unused-invalid')
    const env = { TRITON_MYSQL_URL: 'explicit', TRITON_RISCV_CONFIG: '{}' }
    expect(loadLocalEnvironment(root, env)).toEqual(env)
  })
  it('preserves the existing environment when local files are absent', () => {
    const ambient = { DSH_MODEL: 'old-model', PATH: '/bin' }
    const result = loadLocalEnvironment(fixture(), ambient)
    expect(result).toEqual(ambient)
    expect(result).not.toBe(ambient)
  })
  it('loads existing target configuration without overriding shell settings', () => {
    const result = loadLocalEnvironment(
      fixture(undefined, {
        TRITON_RISCV_REPO_ROOT: '/operators',
        RISCV_HOST: 'sg2044',
        DSH_MODEL: 'saved-model',
      }),
      { DSH_MODEL: 'shell-model' },
    )
    expect(result).toEqual({ TRITON_RISCV_REPO_ROOT: '/operators', RISCV_HOST: 'sg2044', DSH_MODEL: 'shell-model' })
  })
  it('uses the explicit model file instead of stale model and key exports', () => {
    const ambient = { DSH_MODEL: 'AZ-GPT-5.6-Sol', ISRC_API_KEY: 'old-test-key' }
    const result = loadLocalEnvironment(
      fixture('DSH_MODEL=gpt-6-astra\nISRC_BASE_URL=https://model.example/v1\nISRC_API_KEY="new-test-key"\n', {
        DSH_MODEL: 'gpt-5.6-sol',
      }),
      ambient,
    )
    expect(result.DSH_MODEL).toBe('gpt-6-astra')
    expect(result.ISRC_API_KEY).toBe('new-test-key')
    expect(result.ISRC_BASE_URL).toBe('https://model.example/v1')
    expect(ambient.ISRC_API_KEY).toBe('old-test-key')
  })
  it('supports comments and quoted values without executing shell expressions', () => {
    const result = loadLocalEnvironment(
      fixture('# local values only\nDSH_MODEL="gpt-6-astra"\nISRC_API_KEY=literal-$(whoami)\n'),
      {},
    )
    expect(result.ISRC_API_KEY).toBe('literal-$(whoami)')
  })
  it('rejects an empty key rather than silently reusing an old credential', () => {
    expect(() => loadLocalEnvironment(fixture('ISRC_API_KEY=\n'), { ISRC_API_KEY: 'old-test-key' })).toThrow(
      'Fill ISRC_API_KEY',
    )
  })
  it('does not disclose malformed credentials in errors', () => {
    const secret = 'private test secret'
    let message = ''
    try {
      loadLocalEnvironment(fixture(`ISRC_API_KEY="${secret}"\n`), {})
    } catch (error) {
      message = (error as Error).message
    }
    expect(message).toContain('raw key only')
    expect(message).not.toContain(secret)
  })
  it('rejects empty model names and unrelated permissions in the model file', () => {
    expect(() => loadLocalEnvironment(fixture('DSH_MODEL=\n'), {})).toThrow('Fill DSH_MODEL')
    expect(() => loadLocalEnvironment(fixture('TRITON_RISCV_ALLOW_REPAIR_APPLY=1\n'), {})).toThrow(
      'Unsupported setting',
    )
  })
  it('still rejects credentials or unknown settings in native-local.json', () => {
    expect(() => loadLocalEnvironment(fixture(undefined, { ISRC_API_KEY: 'test-only' }), {})).toThrow(
      'Invalid local configuration key',
    )
  })
})
