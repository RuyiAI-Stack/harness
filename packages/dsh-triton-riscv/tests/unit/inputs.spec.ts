import { describe, it, expect, vi } from 'vitest'
import {
  CONFIG_ENV,
  workerEnvironment,
  resolveConfig,
  bridgeEnvironment,
  mcpConfiguration,
  configFromEnvironment,
} from '../../lib/config.js'
import { apply } from '../../native.js'
import { apply as policy } from '../../index.js'

describe('inputs: configuration and activation boundaries', () => {
  it('is inert until explicitly configured, without requiring Python or a checkout', async () => {
    const ctx = { plugin: vi.fn(), on: vi.fn(), effect: vi.fn() }
    await apply(ctx, {})
    expect(ctx.plugin).not.toHaveBeenCalled()
    expect(ctx.on).not.toHaveBeenCalled()
    expect(ctx.effect).not.toHaveBeenCalled()
  })
  it('validates paths, booleans, field names and remote pairs', () => {
    for (const value of [
      { enabled: 'false' },
      { repoRoot: 'relative' },
      { repoRoot: '/tmp/op', permissions: { validation: '1' } },
      { apiKey: 'not-an-allowed-field' },
      { remote: { required: true } },
      { remote: { requireTaskQuotas: 'true' } },
      { remote: { requireTaskQuotas: true } },
      { remote: { host: 'h;sh', repository: '/repo' } },
      { remote: { host: 'host', repository: '/repo/../other' } },
      { memory: { embedding: { tokenBudget: -1 } } },
      { memory: { embedding: { apiKeyEnv: 'TRITON_RISCV_ALLOW_VALIDATION' } } },
    ])
      expect(() => resolveConfig(value)).toThrow()
  })
  it('defaults capabilities off and keeps the existing RAG defaults', () => {
    const result = resolveConfig({ enabled: true, repoRoot: '/tmp/operators' })
    expect(result.permissions).toEqual({ validation: false, repair: false, development: false })
    expect(result.remote.requireTaskQuotas).toBe(false)
    expect(result.memory.retrievalMode).toBe('legacy')
    expect(result.memory.contextFormat).toBe('classic')
    expect(result.memory.embedding.provider).toBe('none')
    expect(result).not.toHaveProperty('env')
  })
  it('uses the same explicit target and permissions for MCP and approval bridge', () => {
    const config = resolveConfig({
      enabled: true,
      repoRoot: '/tmp/operators with spaces',
      python: '/tmp/python',
      stateDir: '/tmp/state',
      permissions: { validation: true },
    })
    const ambient = {
      PATH: '/bin',
      TRITON_RISCV_REPO_ROOT: '/wrong',
      TRITON_RISCV_ALLOW_REPAIR_APPLY: '1',
      RISCV_HOST: 'wrong',
      AGENT_EMBEDDING_PROVIDER: 'wrong',
    }
    const bridge = bridgeEnvironment(config, ambient)
    const mcp = mcpConfiguration(config, ambient)
    expect(bridge[CONFIG_ENV]).toBe(mcp.env[CONFIG_ENV])
    const document = JSON.parse(bridge[CONFIG_ENV])
    expect(document.repoRoot).toBe(config.repoRoot)
    expect(document.permissions).toEqual(config.permissions)
    expect(document.schemaVersion).toBe(1)
    expect(bridge.TRITON_RISCV_ALLOW_REPAIR_APPLY).toBeUndefined()
    expect(Object.keys(mcp.env)).toEqual([CONFIG_ENV])
    expect(bridge.AGENT_EMBEDDING_PROVIDER).toBeUndefined()
    expect(bridge.PATH).toBe('/bin')
    expect(mcp.args).toEqual(['-I', '-m', 'codex_agent.harness.mcp_server'])
    expect(mcp.cwd).toBe(config.repoRoot)
    expect(ambient.RISCV_HOST).toBe('wrong')
  })
  it('does not put provider credentials in declarative config', () => {
    const config = resolveConfig({
      repoRoot: '/tmp/op',
      memory: { embedding: { apiKeyEnv: 'EMBEDDING_TEST_KEY' } },
    })
    expect(JSON.stringify(config)).not.toContain('secret-value')
    expect(mcpConfiguration(config, { EMBEDDING_TEST_KEY: 'secret-value' }).env.EMBEDDING_TEST_KEY).toBe('secret-value')
  })
  it('passes the strict resource requirement to both execution paths, never from ambient settings', () => {
    const config = resolveConfig({
      repoRoot: '/tmp/op',
      remote: { host: 'host', repository: '/repo', required: true, requireTaskQuotas: true },
    })
    expect(
      JSON.parse(bridgeEnvironment(config, { TRITON_RISCV_REQUIRE_TASK_QUOTAS: '0' })[CONFIG_ENV]).remote
        .requireTaskQuotas,
    ).toBe(true)
    expect(JSON.parse(mcpConfiguration(config).env[CONFIG_ENV]).remote.requireTaskQuotas).toBe(true)
    expect(
      JSON.parse(bridgeEnvironment(resolveConfig({}), { TRITON_RISCV_REQUIRE_TASK_QUOTAS: '1' })[CONFIG_ENV]).remote
        .requireTaskQuotas,
    ).toBe(false)
    expect(() => configFromEnvironment({ TRITON_RISCV_REQUIRE_TASK_QUOTAS: 'tru' })).toThrow()
    const legacy = configFromEnvironment({
      TRITON_RISCV_REPO_ROOT: '/tmp/op',
      RISCV_HOST: 'host',
      RISCV_REPO: '/repo',
      TRITON_RISCV_REQUIRE_REMOTE: '1',
      TRITON_RISCV_REQUIRE_TASK_QUOTAS: 'true',
    })
    expect(resolveConfig(legacy).remote.requireTaskQuotas).toBe(true)
  })
  it('keeps legacy launcher translation explicit instead of reading ambient settings in apply', async () => {
    const input = configFromEnvironment({
      TRITON_RISCV_REPO_ROOT: '/tmp/op',
      TRITON_RISCV_ALLOW_VALIDATION: '1',
    })
    expect(resolveConfig(input).permissions.validation).toBe(true)
    vi.stubEnv('TRITON_RISCV_ALLOW_VALIDATION', '1')
    vi.stubEnv('TRITON_RISCV_WORKBENCH_AUTOSTART', '1')
    try {
      expect(resolveConfig({}).enabled).toBe(false)
      const sections = []
      policy({
        effect: fn => fn(),
        systemPrompt: {
          section: s => {
            sections.push(s)
            return () => {}
          },
        },
      })
      expect(sections).toHaveLength(5)
    } finally {
      vi.unstubAllEnvs()
    }
  })
  it('separates credentials, ignores stale native payloads, and rejects reserved key names', () => {
    const config = resolveConfig({ repoRoot: '/tmp/op', memory: { embedding: { apiKeyEnv: 'MY_SECRET' } } })
    const env = bridgeEnvironment(config, {
      [CONFIG_ENV]: '{"permissions":{"repair":true}}',
      MY_SECRET: 'secret-value',
    })
    expect(env.MY_SECRET).toBe('secret-value')
    expect(env[CONFIG_ENV]).not.toContain('secret-value')
    expect(JSON.parse(env[CONFIG_ENV]).permissions.repair).toBe(false)
    for (const apiKeyEnv of [CONFIG_ENV, 'PATH', 'HOME', 'RISCV_HOST'])
      expect(() => resolveConfig({ memory: { embedding: { apiKeyEnv } } })).toThrow()
  })
  it('limits configuration transport size before spawning a worker', () => {
    const config = resolveConfig({ repoRoot: '/tmp/op', memory: { embedding: { model: 'x'.repeat(66000) } } })
    expect(() => workerEnvironment(config)).toThrow('64 KiB')
  })
})
