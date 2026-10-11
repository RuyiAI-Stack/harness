import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { DEFAULT_SCHEMA, Type, load } from 'js-yaml'

// Preserve !!js as text while parsing; evaluate only the trusted MCP env expression.
const schema = DEFAULT_SCHEMA.extend(new Type('tag:yaml.org,2002:js', { kind: 'scalar' }))
const composition = load(
  readFileSync(new URL('../../python/codex_agent/harness/cordis.yml', import.meta.url), 'utf8'),
  { schema },
) as Array<{ id: string; config: { env: string } }>
const expression = composition.find(item => item.id === 'mcp-triton-riscv')!.config.env

function forward(config: object, env: Record<string, string> = {}) {
  return runInNewContext(
    expression,
    {
      process: { env: { TRITON_RISCV_CONFIG: JSON.stringify(config), ...env }, cwd: () => '/fallback' },
    },
    { timeout: 1000 },
  )
}

describe('SDK MCP credential forwarding', () => {
  it('forwards declared tool credentials even when ambient host credentials are scrubbed', () => {
    const forwarded = forward(
      {
        storage: { urlEnv: 'CUSTOM_MYSQL' },
        indexing: { enabled: true, apiKeyEnv: 'VECTOR_SECRET', embedding: { apiKeyEnv: 'EMBED_SECRET' } },
      },
      {
        DSH_CWD: '/operators',
        CUSTOM_MYSQL: 'db-test-only',
        VECTOR_SECRET: 'vector-test-only',
        EMBED_SECRET: 'embedding-test-only',
        ISRC_API_KEY: 'model-must-not-pass',
      },
    )
    expect(forwarded.VECTOR_SECRET).toBe('vector-test-only')
    expect(forwarded.EMBED_SECRET).toBe('embedding-test-only')
    expect(forwarded.CUSTOM_MYSQL).toBe('db-test-only')
    expect(forwarded.TRITON_RISCV_REPO_ROOT).toBe('/operators')
    expect(forwarded.TRITON_RISCV_REQUIRE_APPROVED_VALIDATION).toBe('1')
    expect(forwarded).not.toHaveProperty('ISRC_API_KEY')
  })

  it('does not forward credentials for disabled indexing, cache or queue', () => {
    const forwarded = forward(
      {},
      {
        TRITON_QDRANT_API_KEY: 'test-only',
        TRITON_REDIS_URL: 'test-only',
        TRITON_AMQP_URL: 'test-only',
        ISRC_API_KEY: 'test-only',
        UNRELATED_SECRET: 'test-only',
      },
    )
    expect(forwarded.TRITON_RISCV_REPO_ROOT).toBe('/fallback')
    for (const key of [
      'TRITON_QDRANT_API_KEY',
      'TRITON_REDIS_URL',
      'TRITON_AMQP_URL',
      'ISRC_API_KEY',
      'UNRELATED_SECRET',
    ])
      expect(forwarded).not.toHaveProperty(key)
  })

  it('allows indexing to use its configured queue without enabling validation jobs', () => {
    const forwarded = forward(
      { indexing: { enabled: true }, queue: { enabled: false, urlEnv: 'INDEX_QUEUE' } },
      { INDEX_QUEUE: 'queue-test-only', TRITON_AMQP_URL: 'must-not-forward' },
    )
    expect(forwarded.INDEX_QUEUE).toBe('queue-test-only')
    expect(forwarded).not.toHaveProperty('TRITON_AMQP_URL')
  })

  it('rejects malformed configuration instead of forwarding arbitrary environment variables', () => {
    expect(() => forward({}, { TRITON_RISCV_CONFIG: '{', UNRELATED_SECRET: 'test-only' })).toThrow()
  })
})
