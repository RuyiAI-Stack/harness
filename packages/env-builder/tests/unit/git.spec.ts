import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { cloneRepo, remoteOriginUrl } from '../../src/service/git.ts'

describe('cloneRepo', () => {
  let root: string

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'env-builder-git-'))
  })

  afterEach(() => {
    rmSync(root, { recursive: true, force: true })
  })

  it('removes partial dir on clone failure', async () => {
    const dir = join(root, 'partial')
    await expect(cloneRepo('file:///no/such/repo.git', dir)).rejects.toThrow(/git clone failed/)
    expect(existsSync(dir)).toBe(false)
  })
})

describe('remoteOriginUrl', () => {
  it('throws when not a repo', () => {
    expect(() => remoteOriginUrl('/tmp/env-builder-no-such-git-dir')).toThrow(/git remote/)
  })
})
