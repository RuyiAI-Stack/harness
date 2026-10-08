import { spawn, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { configFromEnvironment, resolveConfig } from '../lib/config.js'
import { loadLocalEnvironment } from './local-environment.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const env = loadLocalEnvironment(root)
const candidates = [
  env.DSH_NATIVE_SOURCE,
  resolve(root, '../../thirdparty/deepseek-harness'),
  join(root, '.runtime/native-host'),
].filter(Boolean)
const host = candidates.find(p => existsSync(join(p, 'apps/cli/lib/bin.js')))
if (!host)
  throw new Error(
    'Build the pinned Harness source first; set DSH_NATIVE_SOURCE to that persistent directory. See README.',
  )
env.DSH_HOME ||= join(root, '.state/native-host')
env.TRITON_RISCV_MCP_PYTHON ||= join(root, '.venv/bin/python')
if (!env.TRITON_RISCV_REPO_ROOT || !existsSync(env.TRITON_RISCV_REPO_ROOT))
  throw new Error('Set TRITON_RISCV_REPO_ROOT to the operator checkout')
env.TRITON_RISCV_STATE_DIR ||= join(root, '.state/agent')
env.TRITON_RISCV_REQUIRE_APPROVED_VALIDATION = '1'
env.TRITON_RISCV_WORKBENCH_AUTOSTART = '0'
const cli = join(host, 'apps/cli/lib/bin.js')
const mode = process.argv[2] || 'web'
if (mode === 'install') {
  const result = spawnSync(process.execPath, [cli, 'plugin', '--profile', 'web', 'add', root], {
    cwd: host,
    env,
    stdio: 'inherit',
  })
  process.exit(result.status ?? 1)
}
if (mode !== 'web') throw new Error('Supported modes: install, web')
if (!env.ISRC_API_KEY && env.TRITON_RISCV_KEYCHAIN_SERVICE && process.platform === 'darwin') {
  const loaded = spawnSync(
    '/usr/bin/security',
    ['find-generic-password', '-a', env.USER || '', '-s', env.TRITON_RISCV_KEYCHAIN_SERVICE, '-w'],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] },
  )
  if (loaded.status === 0) env.ISRC_API_KEY = loaded.stdout.trim()
}
if (env.ISRC_API_KEY && !/^[!-~]+$/.test(env.ISRC_API_KEY))
  throw new Error('ISRC_API_KEY must contain only the raw key; its value has not been logged')
const port = Number(env.TRITON_RISCV_NATIVE_PORT || 8780)
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid native port')
// JSON is also YAML; values are data, never executable !!js interpolations.
const overlay = [
  { id: 'triton-riscv-native-host', config: configFromEnvironment(env) },
  {
    id: 'agent-default-model',
    config: {
      provider: env.DSH_PROVIDER || 'isrc-proxy',
      model: env.DSH_MODEL || 'gpt-5.6-sol',
    },
  },
  {
    id: 'llm-pi-ai',
    config: {
      providers: {
        [env.DSH_PROVIDER || 'isrc-proxy']: {
          displayName: 'ISRC Responses',
          apiKeyEnv: 'ISRC_API_KEY',
          api: 'openai-responses',
          baseURL: env.ISRC_BASE_URL || 'https://llmapi.isrc.ac.cn/v1',
          models: [
            {
              id: env.DSH_MODEL || 'gpt-5.6-sol',
              name: env.DSH_MODEL || 'gpt-5.6-sol',
            },
          ],
        },
      },
    },
  },
]
resolveConfig(overlay[0].config)
if (env.TRITON_RISCV_BROWSER_DIRECTORY_PICKER === '1') {
  overlay.push(
    { id: 'directory-picker', disabled: true },
    {
      insert: [
        {
          id: 'directory-picker-browse',
          name: '@deepseek-ai/dsh-host-directory-picker-browse',
        },
        {
          id: 'ui-directory-picker-browse',
          name: '@deepseek-ai/dsh-client-ui-directory-picker-browse',
        },
      ],
    },
  )
}
mkdirSync(env.DSH_HOME, { recursive: true, mode: 0o700 })
const overlayPath = join(env.DSH_HOME, 'triton-provider.patch.yml')
writeFileSync(overlayPath, JSON.stringify(overlay, null, 2), { mode: 0o600 })
console.log(
  `Native Harness: http://127.0.0.1:${port}; model credential ${env.ISRC_API_KEY ? 'available' : 'not in environment (configure in Models)'}; repository ${env.TRITON_RISCV_REPO_ROOT}`,
)
// Launcher options must precede web-app options, which the CLI forwards verbatim.
const child = spawn(
  process.execPath,
  [cli, 'web', '--patch', overlayPath, '--host', '127.0.0.1', '--port', String(port), '--no-open'],
  { cwd: host, env, stdio: 'inherit' },
)
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal))
child.on('error', error => {
  console.error(error.message)
  process.exitCode = 1
})
child.on('exit', code => {
  process.exitCode = code ?? 1
})
