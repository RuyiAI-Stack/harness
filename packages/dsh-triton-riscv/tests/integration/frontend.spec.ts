import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { stripVTControlCharacters } from 'node:util'

it('runs the workbench Vitest suite, including workflow approvals and validation states', () => {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
  const result = spawnSync('npm', ['run', 'test:frontend'], {
    cwd: root,
    encoding: 'utf8',
    timeout: 110_000,
    maxBuffer: 8 * 1024 * 1024,
  })
  expect(result.error, 'Install frontend dependencies with npm run build first').toBeUndefined()
  expect(result.status, result.stdout + result.stderr).toBe(0)
  expect(stripVTControlCharacters(result.stdout)).toMatch(/\bTests\s+[1-9]\d* passed\b/)
}, 120_000)
