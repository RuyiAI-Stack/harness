import { describe, expect, it } from 'vitest'
import { parseRepoRef } from '../../src/parse.ts'

describe('parseRepoRef', () => {
  it('parses owner/repo', () => {
    expect(parseRepoRef('DangoSys/buckyball')).toEqual({
      owner: 'DangoSys',
      repo: 'buckyball',
      url: 'https://github.com/DangoSys/buckyball.git',
      dir: 'DangoSys/buckyball',
    })
  })

  it('parses https github url', () => {
    expect(parseRepoRef('https://github.com/DangoSys/harness.git').dir).toBe('DangoSys/harness')
  })

  it('throws on garbage', () => {
    expect(() => parseRepoRef('not a repo')).toThrow(/env-builder/)
  })

  it('uses owner/repo as dir to avoid basename collision', () => {
    expect(parseRepoRef('alice/app').dir).toBe('alice/app')
    expect(parseRepoRef('bob/app').dir).toBe('bob/app')
  })

  it('throws on . or .. segments', () => {
    expect(() => parseRepoRef('owner/.')).toThrow(/invalid repo ref/)
    expect(() => parseRepoRef('owner/..')).toThrow(/invalid repo ref/)
    expect(() => parseRepoRef('./repo')).toThrow(/invalid repo ref/)
    expect(() => parseRepoRef('../repo')).toThrow(/invalid repo ref/)
    expect(() => parseRepoRef('https://github.com/owner/..')).toThrow(/invalid repo ref/)
  })
})
