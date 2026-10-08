import assert from 'node:assert/strict'
import { test } from 'vitest'
import { mkdtempSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { createStateStore } from '../../lib/native-state.js'

test('domain state survives reload, remains session/repository scoped and rejects corruption', () => {
  const root = mkdtempSync(join(tmpdir(), 'triton-native-state-'))
  try {
    const stateDir = join(root, 'state')
    const env = {
      TRITON_RISCV_REPO_ROOT: root,
      TRITON_RISCV_STATE_DIR: stateDir,
    }
    const store = createStateStore(env)
    const state = store.load('session-one')
    state.task = { operator: 'square', status: 'failed' }
    state.artifacts = { 'run-1': 'run' }
    store.save(state)
    assert.deepEqual(createStateStore(env).load('session-one'), state)
    assert.deepEqual(store.load('session-two').artifacts, {})
    const second = join(root, 'other')
    mkdirSync(second)
    assert.deepEqual(createStateStore({ ...env, TRITON_RISCV_REPO_ROOT: second }).load('session-one').artifacts, {})
    const path = join(stateDir, 'native-harness', readdirSync(join(stateDir, 'native-harness'))[0])
    writeFileSync(path, '{invalid')
    assert.throws(() => store.load('session-one'), SyntaxError)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
