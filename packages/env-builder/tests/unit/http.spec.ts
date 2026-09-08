import { describe, expect, it } from 'vitest'
import { HttpError, statusOf } from '../../src/http.ts'

describe('statusOf', () => {
  it('uses HttpError status when present', () => {
    expect(statusOf(new HttpError('env-builder: missing repo', 400))).toBe(400)
    expect(statusOf(new HttpError('env-builder: bad path /x', 404))).toBe(404)
  })

  it('defaults to 500 for plain errors', () => {
    expect(statusOf(new Error('env-builder: unknown environment project9'))).toBe(500)
    expect(statusOf(new Error('env-builder: invalid manifest at /x'))).toBe(500)
  })
})
