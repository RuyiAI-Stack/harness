import { describe, expect, it } from 'vitest'
import { nextProjectId } from '../../src/ids.ts'

describe('nextProjectId', () => {
  it('starts at project1', () => {
    expect(nextProjectId([])).toBe('project1')
  })

  it('fills gaps', () => {
    expect(nextProjectId(['project1', 'project3'])).toBe('project2')
  })

  it('skips custom ids', () => {
    expect(nextProjectId(['env1', 'alpha'])).toBe('project1')
    expect(nextProjectId(['alpha', 'project1'])).toBe('project2')
  })
})
