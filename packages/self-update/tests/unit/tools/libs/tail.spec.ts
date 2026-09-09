import { describe, expect, it } from 'vitest'
import { tailText } from '../../../../src/tools/libs/tail.ts'

describe('tailText', () => {
  it('returns the whole text when under the limit', () => {
    expect(tailText('abc', 10)).toBe('abc')
  })

  it('returns exactly the input at the limit', () => {
    expect(tailText('abc', 3)).toBe('abc')
  })

  it('keeps only the last characters beyond the limit', () => {
    expect(tailText('abcdef', 3)).toBe('def')
  })

  it('returns an empty string for empty input', () => {
    expect(tailText('', 3)).toBe('')
  })
})
