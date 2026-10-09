import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseEnv } from 'node:util'

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
