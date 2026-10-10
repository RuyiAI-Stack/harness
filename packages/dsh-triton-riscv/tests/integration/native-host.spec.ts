import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { stripVTControlCharacters } from 'node:util'

const passedTests = /\bTests\s+[1-9]\d* passed\b/

it.each([
  ['plain', 'Tests 12 passed (12)', true],
  ['colored', '\u001b[2m Tests \u001b[22m \u001b[1m\u001b[32m12 passed\u001b[39m (12)', true],
  ['empty', 'Tests 0 passed (0)', false],
  ['files only', 'Test Files 1 passed (1)', false],
])('checks the native test summary: %s', (_label, output, expected) => {
  expect(passedTests.test(stripVTControlCharacters(output))).toBe(expected)
})

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
  expect(stripVTControlCharacters(result.stdout)).toMatch(passedTests)
}, 120000)
