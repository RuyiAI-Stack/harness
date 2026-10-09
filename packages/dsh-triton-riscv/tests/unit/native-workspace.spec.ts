import { afterEach, expect, it } from 'vitest'
import { mkdtempSync, mkdirSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { resolveSessionConfig } from '../../lib/native-workspace.js'
import { bridgeEnvironment, mcpConfiguration, resolveConfig } from '../../lib/config.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})
function fixture() {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'triton-workspaces-')))
  roots.push(root)
  const a = join(root, 'a'),
    b = join(root, 'b with spaces')
  mkdirSync(a)
  mkdirSync(b)
  return { root, a, b }
}
const session = (cwd?: string) => ({ id: 'same-id', header: cwd === undefined ? {} : { cwd } })

it('enables without a preconfigured repo and waits for a host workspace', () => {
  expect(resolveConfig({ enabled: true }).enabled).toBe(true)
  expect(resolveSessionConfig({ enabled: true }, session())).toBeNull()
  expect(() => mcpConfiguration(resolveConfig({ enabled: true }))).toThrow('Select a Harness session workspace')
})
it('selected workspace overrides the legacy fallback and binds every execution path', () => {
  const { a, b } = fixture()
  const config = resolveSessionConfig({ enabled: true, repoRoot: a }, session(b))!
  expect(config.repoRoot).toBe(b)
  expect(mcpConfiguration(config).cwd).toBe(b)
  expect(JSON.parse(bridgeEnvironment(config, { TRITON_RISCV_REPO_ROOT: a }).TRITON_RISCV_CONFIG).repoRoot).toBe(b)
  expect(config.stateDir).toBe(join(b, 'agent-results'))
  expect(config.storage.urlEnv).toBe('TRITON_MYSQL_URL')
})
it('preserves explicit state paths only for the legacy target', () => {
  const { root, a, b } = fixture()
  const input = { repoRoot: a, stateDir: join(root, 'state') }
  const old = resolveSessionConfig(input, session())!
  expect(old.repoRoot).toBe(a)
  expect(old.stateDir).toBe(input.stateDir)
  const changed = resolveSessionConfig(input, session(b))!
  expect(changed.stateDir).toContain('/workspaces/')
  expect(changed.repoRoot).not.toBe(old.repoRoot)
})
it('namespaces custom storage across workspaces and reuses it across sessions in the same workspace', () => {
  const { root, a, b } = fixture()
  const input = { stateDir: join(root, 'state') }
  const one = resolveSessionConfig(input, session(a))!,
    two = resolveSessionConfig(input, session(b))!
  expect(one.stateDir).not.toBe(two.stateDir)
  expect(one.repoRoot).not.toBe(two.repoRoot)
  expect(one.storage).toEqual(two.storage)
  expect(resolveSessionConfig(input, { ...session(a), id: 'another' })).toEqual(one)
})
it('canonicalizes symlinks so aliases share the same repository history', () => {
  const { root, a } = fixture()
  const alias = join(root, 'alias')
  symlinkSync(a, alias)
  expect(resolveSessionConfig({}, session(alias))).toEqual(resolveSessionConfig({}, session(a)))
})
it('rejects an invalid explicit workspace instead of silently falling back', () => {
  const { root, a } = fixture()
  const file = join(root, 'file')
  writeFileSync(file, 'not a directory')
  for (const cwd of ['', 'relative', join(root, 'missing'), file, '/tmp/x\ny'])
    expect(() => resolveSessionConfig({ repoRoot: a }, session(cwd))).toThrow()
})
it('does not require an obsolete fallback directory when a valid workspace is selected', () => {
  const { root, a } = fixture()
  expect(resolveSessionConfig({ repoRoot: join(root, 'removed') }, session(a))!.repoRoot).toBe(a)
})
it('keeps permissions off and existing retrieval mode unchanged in every workspace', () => {
  const { a, b } = fixture()
  for (const cwd of [a, b]) {
    const config = resolveSessionConfig({}, session(cwd))!
    expect(config.permissions).toEqual({ validation: false, development: false, repair: false })
    expect(config.memory.retrievalMode).toBe('legacy')
    expect(config.memory.embedding.provider).toBe('none')
  }
})
