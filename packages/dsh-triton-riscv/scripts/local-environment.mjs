import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseEnv } from 'node:util'
import { resolveConfig, workerEnvironment } from '../lib/config.js'

const localKeys = new Set([
  'DSH_NATIVE_SOURCE', 'DSH_HOME', 'DSH_MODEL', 'DSH_PROVIDER', 'ISRC_BASE_URL',
  'TRITON_RISCV_REPO_ROOT', 'TRITON_RISCV_STATE_DIR',
  'TRITON_RISCV_MCP_PYTHON', 'TRITON_RISCV_NATIVE_PORT', 'TRITON_RISCV_KEYCHAIN_SERVICE',
  'TRITON_RISCV_BROWSER_DIRECTORY_PICKER', 'TRITON_RISCV_MEMORY_RETRIEVAL_MODE',
  'TRITON_RISCV_MEMORY_CONTEXT_FORMAT', 'TRITON_RISCV_EMBEDDING_PROVIDER',
  'RISCV_HOST', 'RISCV_REPO', 'TRITON_RISCV_ALLOW_VALIDATION',
  'TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY', 'TRITON_RISCV_ALLOW_REPAIR_APPLY',
  'TRITON_RISCV_REQUIRE_REMOTE', 'TRITON_RISCV_REQUIRE_APPROVED_VALIDATION',
])
const modelKeys = new Set(['DSH_MODEL', 'ISRC_BASE_URL', 'ISRC_API_KEY'])

export function loadLocalEnvironment(root, ambient = process.env) {
  const env = { ...ambient }
  const settingsPath = join(root, '.state/native-local.json')
  if (existsSync(settingsPath)) {
    const local = JSON.parse(readFileSync(settingsPath, 'utf8'))
    for (const [key, value] of Object.entries(local)) {
      if (key === 'TRITON_RISCV_MEMORY_DB')
        throw new Error('Legacy memory database setting: import SQLite into MySQL, then remove TRITON_RISCV_MEMORY_DB from .state/native-local.json')
      if (!localKeys.has(key) || typeof value !== 'string')
        throw new Error('Invalid local configuration key: ' + key)
      if (env[key] === undefined) env[key] = value
    }
  }

  const servicesPath = join(root, '.state/services.env')
  if (existsSync(servicesPath)) {
    const services = parseEnv(readFileSync(servicesPath, 'utf8'))
    const allowed = new Set(['TRITON_MYSQL_URL', 'TRITON_REDIS_URL', 'TRITON_AMQP_URL', 'TRITON_QDRANT_API_KEY', 'AGENT_EMBEDDING_API_KEY'])
    if (Object.keys(services).some(key => !allowed.has(key)))
      throw new Error('Unsupported setting in .state/services.env')
    for (const [key, value] of Object.entries(services)) if (env[key] === undefined) env[key] = value
  }
  const runtimePath = join(root, '.state/runtime.json')
  if (existsSync(runtimePath) && env.TRITON_RISCV_CONFIG === undefined) {
    let config
    try {
      const raw = readFileSync(runtimePath, 'utf8')
      if (Buffer.byteLength(raw) > 65536) throw new Error('size')
      const { schemaVersion, ...settings } = JSON.parse(raw)
      if (schemaVersion !== 1) throw new Error('version')
      config = resolveConfig(settings)
    } catch {
      throw new Error('Invalid .state/runtime.json; check schemaVersion and plugin configuration fields')
    }
    Object.assign(env, workerEnvironment(config, env))
    env.TRITON_RISCV_REPO_ROOT = config.repoRoot
    env.TRITON_RISCV_STATE_DIR = config.stateDir
  }

  const modelPath = join(root, '.state/model.env')
  if (!existsSync(modelPath)) return env
  const model = parseEnv(readFileSync(modelPath, 'utf8'))
  if (Object.keys(model).some(key => !modelKeys.has(key)))
    throw new Error('Unsupported setting in .state/model.env; only model, API URL and API key are allowed')
  for (const key of ['DSH_MODEL', 'ISRC_BASE_URL']) {
    if (Object.hasOwn(model, key)) {
      model[key] = model[key].trim()
      if (!model[key]) throw new Error('Fill ' + key + ' in .state/model.env')
    }
  }
  if (Object.hasOwn(model, 'ISRC_API_KEY') && !/^[!-~]+$/.test(model.ISRC_API_KEY))
    throw new Error('Fill ISRC_API_KEY in .state/model.env with the raw key only; its value is not logged')
  // An explicit local model file wins over stale shell exports or saved defaults.
  // Do not write its values back to configuration, logs, or the parent environment.
  return { ...env, ...model }
}
