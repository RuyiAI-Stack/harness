import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect } from 'vitest'

export const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
export const python =
  process.env.DSH_MULTI_AGENT_TEST_PYTHON ||
  process.env.DSH_PLUGIN_TEST_PYTHON ||
  resolve(root, '../dsh-triton-riscv/.venv/bin/python')

export function probe(driver: string, input: object = {}) {
  const result = spawnSync(
    python,
    [
      '-I',
      '-B',
      '-c',
      'import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))',
      resolve(root, 'python'),
      driver,
    ],
    {
      input: JSON.stringify(input),
      encoding: 'utf8',
      timeout: 15000,
      env: Object.fromEntries(
        Object.entries(process.env).filter(
          ([key]) => !/^(DSH_|TRITON_|RISCV_|PYTHONPATH)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
        ),
      ),
    },
  )
  expect(result.error).toBeUndefined()
  expect(result.status, result.stderr + result.stdout).toBe(0)
  return JSON.parse(result.stdout)
}
