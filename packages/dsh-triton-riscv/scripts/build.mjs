import { cpSync, mkdirSync, readFileSync, existsSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { TRITON_RISCV_SYSTEM_PROMPT } from '../lib/prompts.js'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const bundle = join(root, 'python/codex_agent/harness/bundle')
writeFileSync(join(root, 'policy.md'), TRITON_RISCV_SYSTEM_PROMPT + '\n')
mkdirSync(bundle, { recursive: true })
for (const name of ['package.json', 'index.js', 'policy.md', 'cordis.patch.yml']) {
  cpSync(join(root, name), join(bundle, name))
}
cpSync(join(root, 'native.js'), join(bundle, 'native.js'))
mkdirSync(join(bundle, 'lib'), { recursive: true })
cpSync(join(root, 'prompts'), join(bundle, 'prompts'), { recursive: true })
for (const name of [
  'client.js',
  'native-adapter.js',
  'tool-policy.js',
  'native-bridge.js',
  'native-state.js',
  'native-workspace.js',
  'config.js',
  'prompts.js',
]) {
  cpSync(join(root, 'lib', name), join(bundle, 'lib', name))
}

if (!process.argv.includes('--resources-only')) {
  const frontend = join(root, 'frontend')
  if (!existsSync(join(frontend, 'node_modules/.bin/vite'))) {
    const install = spawnSync('npm', ['ci', '--no-audit', '--no-fund'], {
      cwd: frontend,
      stdio: 'inherit',
    })
    if (install.status !== 0) process.exit(install.status ?? 1)
  }
  const build = spawnSync('npm', ['run', 'build'], {
    cwd: frontend,
    stdio: 'inherit',
  })
  if (build.status !== 0) process.exit(build.status ?? 1)
  const destination = join(root, 'python/codex_agent/frontend/dist')
  rmSync(destination, { recursive: true, force: true })
  cpSync(join(frontend, 'dist'), destination, { recursive: true })
  const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))
  console.log(`Built ${manifest.name}: frontend and Python package resources`)
}
