import { cpSync, existsSync, rmSync } from 'node:fs'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const host = process.env.TRITON_PLUGIN_TEST_HOST
if (!host || !existsSync(join(host, 'node_modules/vitest/vitest.mjs'))) {
  throw new Error(
    'Set TRITON_PLUGIN_TEST_HOST to an installed disposable Harness source checkout (not your working checkout)',
  )
}
const destination = join(host, 'packages/core/tools/tests/triton-riscv-plugin.native.spec.ts')
if (existsSync(destination)) throw new Error('Refusing to overwrite an existing host test file')
try {
  cpSync(join(root, 'tests/native-host.spec.ts'), destination)
  const result = spawnSync(
    process.execPath,
    ['node_modules/vitest/vitest.mjs', 'run', destination, '--reporter=verbose'],
    {
      cwd: host,
      stdio: 'inherit',
      env: {
        ...process.env,
        TRITON_PLUGIN_TEST_PACKAGE: root,
        TRITON_PLUGIN_TEST_PYTHON: process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python'),
      },
    },
  )
  process.exitCode = result.status ?? 1
} finally {
  rmSync(destination, { force: true })
}
