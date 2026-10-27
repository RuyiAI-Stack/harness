import { afterEach, expect, it, vi } from 'vitest'
import { accountStreamUrl, authFetch, setAccount } from './auth'

const alice = { id: 'alice-id', username: 'alice', csrf_token: 'alice-csrf' }
const bob = { id: 'bob-id', username: 'bob', csrf_token: 'bob-csrf' }
const response = (status = 200) => new Response(JSON.stringify({ title: 'alice-private' }), { status })

afterEach(() => {
  setAccount(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('binds reads and mutations to the displayed account without trusting caller headers', async () => {
  setAccount(alice)
  const fetch = vi.fn().mockImplementation(async () => response())
  vi.stubGlobal('fetch', fetch)
  await authFetch('/api/sessions', { headers: { 'X-Triton-Account': bob.id } })
  await authFetch('/api/sessions', { method: 'POST' })
  await authFetch('/api/auth/me')
  expect(fetch.mock.calls[0][1].headers.get('X-Triton-Account')).toBe(alice.id)
  expect(fetch.mock.calls[0][1].headers.has('X-CSRF-Token')).toBe(false)
  expect(fetch.mock.calls[1][1].headers.get('X-CSRF-Token')).toBe(alice.csrf_token)
  expect(fetch.mock.calls[2][1].headers.has('X-Triton-Account')).toBe(false)
})

it.each([200, 401])('discards an old account response (%s) without signing out the new account', async status => {
  setAccount(alice)
  let resolve!: (value: Response) => void
  vi.stubGlobal(
    'fetch',
    vi.fn(
      () =>
        new Promise<Response>(r => {
          resolve = r
        }),
    ),
  )
  const event = vi.spyOn(window, 'dispatchEvent')
  const pending = authFetch('/api/sessions')
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  setAccount(bob)
  resolve(response(status))
  await rejected
  expect(event).not.toHaveBeenCalled()
})

it('discards an old JSON body when account switching happens after headers arrived', async () => {
  setAccount(alice)
  const old = response()
  let resolve!: (body: unknown) => void
  vi.spyOn(old, 'json').mockImplementation(
    () =>
      new Promise(r => {
        resolve = r
      }),
  )
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(old))
  const body = (await authFetch('/api/sessions')).json()
  const rejected = expect(body).rejects.toMatchObject({ name: 'AbortError' })
  setAccount(bob)
  resolve({ title: 'alice-private' })
  await rejected
})

it('keeps responses valid for routine identity refreshes but invalidates rotated sessions', async () => {
  setAccount(alice)
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(async () => response()),
  )
  const current = await authFetch('/api/sessions')
  setAccount({ ...alice })
  await expect(current.json()).resolves.toEqual({ title: 'alice-private' })
  const old = await authFetch('/api/sessions')
  setAccount({ ...alice, csrf_token: 'new-session' })
  await expect(old.json()).rejects.toMatchObject({ name: 'AbortError' })
})

it('binds EventSource to an account without exposing CSRF or cookie secrets in the URL', () => {
  const path = '/api/runs/run-a/events/stream?after=5'
  expect(accountStreamUrl(path)).toBe(path)
  setAccount(alice)
  expect(accountStreamUrl(path)).toBe(path + '&account_id=alice-id')
  expect(accountStreamUrl(path)).not.toContain(alice.csrf_token)
})
