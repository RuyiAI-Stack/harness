import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { expect } from 'vitest'

export const python =
  process.env.VECTOR_PLUGIN_TEST_PYTHON ||
  process.env.DSH_PLUGIN_TEST_PYTHON ||
  process.env.TRITON_PLUGIN_TEST_PYTHON ||
  fileURLToPath(new URL('../../dsh-triton-riscv/.venv/bin/python', import.meta.url))
const source = fileURLToPath(new URL('../python', import.meta.url))
const triton = fileURLToPath(new URL('../../dsh-triton-riscv/python', import.meta.url))
const multiAgent = fileURLToPath(new URL('../../dsh-multi-agent/python', import.meta.url))

export function probe(code: string, input: object = {}, adapter = false) {
  const child = spawnSync(
    python,
    [
      '-I',
      '-B',
      '-c',
      'import sys; sys.path[:0] = sys.argv[1:4]; exec(sys.argv[4])',
      source,
      adapter ? triton : source,
      adapter ? multiAgent : source,
      code,
    ],
    {
      input: JSON.stringify(input),
      encoding: 'utf8',
      timeout: 30000,
      env: { PATH: process.env.PATH, HOME: process.env.HOME },
    },
  )
  expect(child.error, child.stderr).toBeUndefined()
  expect(child.status, child.stdout + child.stderr).toBe(0)
  return JSON.parse(child.stdout)
}
