import { dirname, isAbsolute, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)))
function object(value, keys, label) {
  const result = value ?? {}
  if (typeof result !== 'object' || Array.isArray(result)) throw new Error(`${label} must be an object`)
  for (const key of Object.keys(result)) if (!keys.includes(key)) throw new Error(`Unknown ${label} field: ${key}`)
  return result
}
function boolean(value, fallback, label) {
  if (value === undefined) return fallback
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`)
  return value
}
function text(value, fallback, label) {
  if (value === undefined) return fallback
  if (typeof value !== 'string' || /[\0\r\n]/.test(value)) throw new Error(`${label} must be a single-line string`)
  return value.trim()
}
function absolute(value, fallback, label) {
  const result = text(value, fallback, label)
  if (result && !isAbsolute(result)) throw new Error(`${label} must be an absolute path`)
  return result
}

export const CONFIG_ENV = 'TRITON_RISCV_CONFIG'

// Native JS and Python share one versioned document, not individual settings in env.
// Plugin code never changes process.env or inherits ambient capability switches.
export function resolveConfig(input = {}) {
  const c = object(
    input,
    ['enabled', 'repoRoot', 'python', 'stateDir', 'permissions', 'remote', 'memory', 'storage', 'cache', 'queue'],
    'triton-riscv config',
  )
  const p = object(c.permissions, ['validation', 'development', 'repair'], 'permissions')
  const r = object(c.remote, ['host', 'repository', 'required', 'requireTaskQuotas'], 'remote')
  const m = object(c.memory, ['retrievalMode', 'contextFormat', 'embedding'], 'memory')
  const e = object(
    m.embedding,
    ['provider', 'model', 'baseUrl', 'tokenizerJson', 'tokenBudget', 'apiKeyEnv'],
    'embedding',
  )
  const db = object(c.storage, ['urlEnv', 'poolSize', 'maxOverflow', 'poolTimeout'], 'storage')
  const cacheDefaults = {
    enabled: false,
    urlEnv: 'TRITON_REDIS_URL',
    ttlSeconds: 60,
    negativeTtlSeconds: 20,
    lockMs: 10000,
    waitMs: 1000,
    maxBytes: 262144,
    maxConcurrent: 4,
    requestsPerSecond: 100,
    rebuildsPerSecond: 10,
    bloom: false,
    bloomCapacity: 10000,
    bloomErrorRate: 0.001,
  }
  const cache = { ...cacheDefaults, ...object(c.cache, Object.keys(cacheDefaults), 'cache') }
  for (const name of ['enabled', 'bloom']) cache[name] = boolean(cache[name], false, 'cache.' + name)
  for (const [key, [min, max]] of Object.entries({
    ttlSeconds: [1, 3600],
    negativeTtlSeconds: [1, 60],
    lockMs: [100, 60000],
    waitMs: [10, 5000],
    maxBytes: [1024, 1048576],
    maxConcurrent: [1, 32],
    requestsPerSecond: [1, 10000],
    rebuildsPerSecond: [1, 1000],
    bloomCapacity: [100, 10000000],
  })) {
    if (!Number.isInteger(cache[key]) || cache[key] < min || cache[key] > max) throw new Error('Invalid cache.' + key)
  }
  if (
    typeof cache.bloomErrorRate !== 'number' ||
    !Number.isFinite(cache.bloomErrorRate) ||
    cache.bloomErrorRate <= 0 ||
    cache.bloomErrorRate > 0.1
  )
    throw new Error('Invalid cache.bloomErrorRate')
  cache.urlEnv = text(cache.urlEnv, 'TRITON_REDIS_URL', 'cache.urlEnv')
  if (
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(cache.urlEnv) ||
    /^(TRITON_RISCV_|RISCV_)/.test(cache.urlEnv) ||
    ['PATH', 'HOME', 'PYTHONPATH', 'PYTHONHOME', db.urlEnv ?? 'TRITON_MYSQL_URL'].includes(cache.urlEnv)
  )
    throw new Error('Invalid cache.urlEnv')
  const urlEnv = text(db.urlEnv, 'TRITON_MYSQL_URL', 'storage.urlEnv')
  if (
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(urlEnv) ||
    /^(TRITON_RISCV_|RISCV_)/.test(urlEnv) ||
    ['PATH', 'HOME', 'PYTHONPATH', 'PYTHONHOME'].includes(urlEnv)
  )
    throw new Error('Invalid storage.urlEnv')
  const storage = {
    urlEnv,
    poolSize: db.poolSize ?? 5,
    maxOverflow: db.maxOverflow ?? 5,
    poolTimeout: db.poolTimeout ?? 10,
  }
  const queueDefaults = {
    enabled: false,
    urlEnv: 'TRITON_AMQP_URL',
    leaseSeconds: 60,
    maxAttempts: 3,
    jobTimeoutSeconds: 1200,
    maxPending: 100,
  }
  const queue = { ...queueDefaults, ...object(c.queue, Object.keys(queueDefaults), 'queue') }
  queue.enabled = boolean(queue.enabled, false, 'queue.enabled')
  queue.urlEnv = text(queue.urlEnv, 'TRITON_AMQP_URL', 'queue.urlEnv')
  if (
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(queue.urlEnv) ||
    /^(TRITON_RISCV_|RISCV_)/.test(queue.urlEnv) ||
    [
      'PATH',
      'HOME',
      'PYTHONPATH',
      'PYTHONHOME',
      storage.urlEnv,
      cache.urlEnv,
      e.apiKeyEnv ?? 'AGENT_EMBEDDING_API_KEY',
    ].includes(queue.urlEnv)
  )
    throw new Error('Invalid queue.urlEnv')
  for (const [key, [min, max]] of Object.entries({
    leaseSeconds: [15, 600],
    maxAttempts: [1, 10],
    jobTimeoutSeconds: [30, 7200],
    maxPending: [1, 10000],
  })) {
    if (!Number.isInteger(queue[key]) || queue[key] < min || queue[key] > max) throw new Error('Invalid queue.' + key)
  }
  for (const [key, value] of Object.entries(storage)) {
    if (key === 'urlEnv') continue
    if (
      !Number.isInteger(value) ||
      value < (key === 'maxOverflow' ? 0 : 1) ||
      value > (key === 'poolTimeout' ? 60 : 50)
    )
      throw new Error('Invalid storage.' + key)
  }
  const enabled = boolean(c.enabled, false, 'enabled')
  const repoRoot = absolute(c.repoRoot, '', 'repoRoot')
  const python = absolute(c.python, join(packageRoot, '.venv/bin/python'), 'python')
  if (!python) throw new Error('python must name an absolute executable')
  const stateDir = absolute(c.stateDir, repoRoot ? join(repoRoot, 'agent-results') : '', 'stateDir')
  const host = text(r.host, '', 'remote.host')
  const repository = absolute(r.repository, '', 'remote.repository')
  const required = boolean(r.required, false, 'remote.required')
  const requireTaskQuotas = boolean(r.requireTaskQuotas, false, 'remote.requireTaskQuotas')
  if (requireTaskQuotas && !required) throw new Error('remote.requireTaskQuotas requires remote.required=true')
  if (Boolean(host) !== Boolean(repository) || (required && !host))
    throw new Error('remote.host and remote.repository must be configured together')
  if (host && !/^[A-Za-z0-9_.-]+$/.test(host)) throw new Error('Invalid remote.host')
  if (repository && (!/^\/[A-Za-z0-9_./-]+$/.test(repository) || repository.split('/').includes('..')))
    throw new Error('Invalid remote.repository')
  const keyName = text(e.apiKeyEnv, 'AGENT_EMBEDDING_API_KEY', 'embedding.apiKeyEnv')
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(keyName)) throw new Error('Invalid embedding.apiKeyEnv')
  const tokenBudget = e.tokenBudget ?? null
  if (tokenBudget !== null && (!Number.isInteger(tokenBudget) || tokenBudget <= 0))
    throw new Error('embedding.tokenBudget must be a positive integer')
  if (/^(TRITON_RISCV_|RISCV_)/.test(keyName) || ['PATH', 'HOME', 'PYTHONPATH', 'PYTHONHOME'].includes(keyName))
    throw new Error('embedding.apiKeyEnv conflicts with a reserved configuration key')
  return {
    enabled,
    repoRoot,
    python,
    stateDir,
    storage,
    cache,
    queue,
    permissions: {
      validation: boolean(p.validation, false, 'permissions.validation'),
      development: boolean(p.development, false, 'permissions.development'),
      repair: boolean(p.repair, false, 'permissions.repair'),
    },
    remote: { host, repository, required, requireTaskQuotas },
    memory: {
      retrievalMode: text(m.retrievalMode, 'legacy', 'memory.retrievalMode'),
      contextFormat: text(m.contextFormat, 'classic', 'memory.contextFormat'),
      embedding: {
        provider: text(e.provider, 'none', 'embedding.provider'),
        model: text(e.model, '', 'embedding.model'),
        baseUrl: text(e.baseUrl, '', 'embedding.baseUrl'),
        tokenizerJson: absolute(e.tokenizerJson, '', 'embedding.tokenizerJson'),
        tokenBudget,
        apiKeyEnv: keyName,
      },
    },
  }
}
export function workerEnvironment(config, ambient = process.env) {
  const { enabled, python, ...settings } = config
  const document = JSON.stringify({ schemaVersion: 1, ...settings })
  if (Buffer.byteLength(document) > 65536) throw new Error('Triton-RISCV configuration exceeds 64 KiB')
  const env = { [CONFIG_ENV]: document }
  const key = config.memory.embedding.apiKeyEnv
  if (ambient[key]) env[key] = ambient[key]
  const databaseKey = config.storage.urlEnv
  if (ambient[databaseKey]) env[databaseKey] = ambient[databaseKey]
  const cacheKey = config.cache.urlEnv
  if (config.cache.enabled && ambient[cacheKey]) env[cacheKey] = ambient[cacheKey]
  const queueKey = config.queue.urlEnv
  if (config.queue.enabled && ambient[queueKey]) env[queueKey] = ambient[queueKey]
  return env
}
export function bridgeEnvironment(config, ambient = process.env) {
  const inherited = Object.fromEntries(
    Object.entries(ambient).filter(([key]) => !/^(TRITON_RISCV_|RISCV_|AGENT_EMBEDDING_)/.test(key)),
  )
  return { ...inherited, ...workerEnvironment(config, ambient) }
}
export function mcpConfiguration(config, ambient = process.env) {
  if (!config.repoRoot) throw new Error('Select a Harness session workspace before connecting Triton-RISCV tools')
  const env = workerEnvironment(config, ambient)
  return {
    serverName: 'triton_riscv',
    transport: 'stdio',
    command: config.python,
    args: ['-I', '-m', 'codex_agent.harness.mcp_server'],
    cwd: config.repoRoot,
    env,
    toolCallTimeoutMs: 960000,
    failOnStartupError: true,
  }
}
// Compatibility is confined to the optional standalone launcher, not apply().
export function configFromEnvironment(env) {
  const quota = (env.TRITON_RISCV_REQUIRE_TASK_QUOTAS ?? '0').trim().toLowerCase()
  if (!['', '0', 'false', '1', 'true'].includes(quota))
    throw new Error('TRITON_RISCV_REQUIRE_TASK_QUOTAS must be 0 or 1')
  return {
    enabled: true,
    repoRoot: env.TRITON_RISCV_REPO_ROOT || env.TRITON_RISCV_CHECKOUT || env.DSH_CWD,
    python: env.TRITON_RISCV_MCP_PYTHON,
    stateDir: env.TRITON_RISCV_STATE_DIR,
    permissions: {
      validation: env.TRITON_RISCV_ALLOW_VALIDATION === '1',
      development: env.TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY === '1',
      repair: env.TRITON_RISCV_ALLOW_REPAIR_APPLY === '1',
    },
    remote: {
      host: env.RISCV_HOST,
      repository: env.RISCV_REPO,
      required: env.TRITON_RISCV_REQUIRE_REMOTE === '1',
      requireTaskQuotas: quota === '1' || quota === 'true',
    },
    memory: {
      retrievalMode: env.TRITON_RISCV_MEMORY_RETRIEVAL_MODE,
      contextFormat: env.TRITON_RISCV_MEMORY_CONTEXT_FORMAT,
      embedding: {
        provider: env.TRITON_RISCV_EMBEDDING_PROVIDER,
        model: env.TRITON_RISCV_EMBEDDING_MODEL,
        baseUrl: env.TRITON_RISCV_EMBEDDING_BASE_URL,
        tokenizerJson: env.TRITON_RISCV_EMBEDDING_TOKENIZER_JSON,
        tokenBudget: env.TRITON_RISCV_EMBEDDING_TOKEN_BUDGET
          ? Number(env.TRITON_RISCV_EMBEDDING_TOKEN_BUDGET)
          : undefined,
        apiKeyEnv: env.TRITON_RISCV_EMBEDDING_API_KEY_ENV,
      },
    },
  }
}
