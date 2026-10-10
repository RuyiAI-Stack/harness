import { beforeAll, describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const source = join(root, 'python')
const tests = join(source, 'codex_agent/tests')
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python')
const modules = readdirSync(tests)
  .filter(name => /^test_.*\.py$/.test(name))
  .sort()

beforeAll(() => {
  expect(existsSync(python), 'Install the plugin backend with npm run setup:backend first').toBe(true)
  for (const name of ['TRITON_MYSQL_URL', 'TRITON_TEST_REDIS_URL', 'TRITON_TEST_AMQP_URL']) {
    expect(process.env[name], `Set ${name} to a disposable test service; see the plugin README`).toBeTruthy()
  }
  expect(modules.length, 'No Python test modules discovered').toBeGreaterThan(0)
})

// Pytest runs the existing TestCase and function-style tests; Vitest owns CI status.
describe.sequential('Python backend', () => {
  for (const module of modules) {
    it(
      module,
      () => {
        const result = spawnSync(
          python,
          [
            '-I',
            '-c',
            'import sys; sys.path.insert(0, sys.argv.pop(1)); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))',
            source,
            join(tests, module),
            '-q',
            '-ra',
            '--import-mode=importlib',
          ],
          {
            cwd: source,
            encoding: 'utf8',
            timeout: 110_000,
            maxBuffer: 16 * 1024 * 1024,
            env: { ...process.env, PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1', PYTEST_ADDOPTS: '' },
          },
        )
        const output = result.stdout + result.stderr
        console.log(`${module}\n${output}`)
        expect(result.error, 'Python process failed or timed out').toBeUndefined()
        // Pytest also returns nonzero on collection errors or when no tests run.
        expect(result.status, output).toBe(0)
      },
      120_000,
    )
  }
})
