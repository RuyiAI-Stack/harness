import { isAbsolute } from 'node:path'

export function fields(value, allowed) {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    Object.keys(value).some(key => !allowed.includes(key))
  )
    throw new Error('Invalid vector retrieval fields')
  return value
}

function url(value) {
  try {
    const parsed = new URL(value)
    if (
      !['http:', 'https:'].includes(parsed.protocol) ||
      !parsed.hostname ||
      parsed.username ||
      parsed.password ||
      parsed.search ||
      parsed.hash
    )
      throw new Error()
    return value
  } catch {
    throw new Error('Invalid vector retrieval URL')
  }
}

export function resolveConfig(input = {}) {
  fields(input, [
    'python',
    'workspaceId',
    'generation',
    'fingerprint',
    'collection',
    'qdrantUrl',
    'dimensions',
    'apiKeyEnv',
    'timeoutSeconds',
    'timeoutMs',
    'embedding',
  ])
  const c = {
    qdrantUrl: 'http://127.0.0.1:6333',
    apiKeyEnv: 'DSH_VECTOR_API_KEY',
    timeoutSeconds: 30,
    timeoutMs: 120000,
    ...input,
  }
  if (typeof c.python !== 'string' || !isAbsolute(c.python) || c.python.includes('\0')) {
    throw new Error('Vector retrieval requires an absolute Python executable')
  }
  for (const key of ['workspaceId', 'generation', 'fingerprint', 'collection']) {
    if (typeof c[key] !== 'string' || !c[key].trim())
      throw new Error('Vector retrieval requires a trusted scope and collection')
  }
  if (!/^[A-Za-z0-9_-]+$/.test(c.collection)) throw new Error('Invalid vector collection')
  if (!Number.isInteger(c.dimensions) || c.dimensions < 1 || c.dimensions > 65536)
    throw new Error('Invalid vector dimensions')
  if (
    !Number.isFinite(c.timeoutSeconds) ||
    c.timeoutSeconds <= 0 ||
    c.timeoutSeconds > 300 ||
    !Number.isInteger(c.timeoutMs) ||
    c.timeoutMs < 1 ||
    c.timeoutMs > 600000
  )
    throw new Error('Invalid vector timeout')
  url(c.qdrantUrl)
  const e = fields(c.embedding ?? {}, [
    'provider',
    'model',
    'baseUrl',
    'apiKeyEnv',
    'tokenizerJson',
    'tokenBudget',
    'revision',
  ])
  c.embedding = { provider: 'none', apiKeyEnv: 'DSH_EMBEDDING_API_KEY', ...e }
  if (!['none', 'openai-compatible', 'ollama', 'sentence-transformers'].includes(c.embedding.provider)) {
    throw new Error('Invalid embedding provider')
  }
  if (e.baseUrl) url(e.baseUrl)
  if (e.tokenBudget !== undefined && (!Number.isInteger(e.tokenBudget) || e.tokenBudget < 1))
    throw new Error('Invalid embedding token budget')
  for (const key of [c.apiKeyEnv, c.embedding.apiKeyEnv]) {
    if (typeof key !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key))
      throw new Error('Invalid credential environment name')
  }
  Object.freeze(c.embedding)
  return Object.freeze(c)
}

export function bridgeEnvironment(config, env = process.env) {
  const { python, timeoutMs, ...settings } = config
  const result = {}
  for (const key of [
    'PATH',
    'HOME',
    'TMPDIR',
    'TEMP',
    'SYSTEMROOT',
    'SSL_CERT_FILE',
    'SSL_CERT_DIR',
    config.apiKeyEnv,
    config.embedding.apiKeyEnv,
  ]) {
    if (env[key] !== undefined) result[key] = env[key]
  }
  result.DSH_VECTOR_RETRIEVAL_CONFIG = JSON.stringify(settings)
  return result
}
