import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

it('integrates the configured plugin with the actual pinned Harness and Python MCP', () => {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
  const host = process.env.TRITON_PLUGIN_TEST_HOST || resolve(root, '../../thirdparty/deepseek-harness')
  const result = spawnSync(process.execPath, ['scripts/test-native-host.mjs'], {
    cwd: root,
    encoding: 'utf8',
    timeout: 110000,
    env: { ...process.env, TRITON_PLUGIN_TEST_HOST: host },
  })
  expect(result.error, 'Native host prerequisite/build missing; run repository install-all first').toBeUndefined()
  expect(result.status, result.stdout + result.stderr).toBe(0)
  expect(result.stdout).toMatch(/5 passed/)
}, 120000)
