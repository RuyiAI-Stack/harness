import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { AccountGate } from './AccountGate'
import { authFetch, setAccount } from './auth'

const user = { id: 'a', username: 'alice', csrf_token: 'csrf-a' }
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

afterEach(() => {
  cleanup()
  setAccount(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('blocks the workbench until login, then unmounts private data on logout', async () => {
  let loggedIn = false
  const fetch = vi.fn().mockImplementation(async (path: string) => {
    if (path.endsWith('/login')) {
      loggedIn = true
      return reply({ user })
    }
    if (path.endsWith('/logout')) {
      loggedIn = false
      return reply({ signed_out: true })
    }
    return loggedIn ? reply({ enabled: true, user }) : reply({}, 401)
  })
  vi.stubGlobal('fetch', fetch)
  render(
    <AccountGate>
      <div>private workspace</div>
    </AccountGate>,
  )
  expect(screen.queryByText('private workspace')).not.toBeInTheDocument()
  await screen.findByRole('heading', { name: '登录工作台' })
  fireEvent.change(screen.getByLabelText('用户名'), { target: { value: 'alice' } })
  fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'long-test-password' } })
  fireEvent.click(screen.getByRole('button', { name: '登录' }))
  await screen.findByText('private workspace')
  fireEvent.click(screen.getByRole('button', { name: '退出登录' }))
  await screen.findByRole('heading', { name: '登录工作台' })
  expect(screen.queryByText('private workspace')).not.toBeInTheDocument()
  expect(fetch.mock.calls.find(call => call[0].endsWith('/logout'))?.[1].headers.get('X-CSRF-Token')).toBe('csrf-a')
})

it('requires an invitation and returns to login after registration', async () => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValueOnce(reply({}, 401))
      .mockResolvedValueOnce(reply({ registered: true }, 201)),
  )
  render(
    <AccountGate>
      <div>private</div>
    </AccountGate>,
  )
  fireEvent.click(await screen.findByRole('tab', { name: '注册' }))
  fireEvent.change(screen.getByLabelText('用户名'), { target: { value: 'alice' } })
  fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'long-test-password' } })
  fireEvent.change(screen.getByLabelText('邀请码'), { target: { value: 'invitation' } })
  fireEvent.click(screen.getByRole('button', { name: '注册账号' }))
  await screen.findByText('注册成功，请登录')
  expect(screen.queryByText('private')).not.toBeInTheDocument()
})

it('fails closed when the identity service is unavailable', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply({}, 503)))
  render(
    <AccountGate>
      <div>private</div>
    </AccountGate>,
  )
  await screen.findByRole('alert')
  expect(screen.queryByText('private')).not.toBeInTheDocument()
})

it('preserves the explicit local single-user mode', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply({ enabled: false, user: null })))
  render(
    <AccountGate>
      <div>local workspace</div>
    </AccountGate>,
  )
  await screen.findByText('local workspace')
  expect(screen.queryByRole('button', { name: '退出登录' })).not.toBeInTheDocument()
})

it('ignores a late identity response after signout', async () => {
  let resolve!: (response: Response) => void
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(
      () =>
        new Promise<Response>(r => {
          resolve = r
        }),
    ),
  )
  render(
    <AccountGate>
      <div>private</div>
    </AccountGate>,
  )
  await waitFor(() => expect(resolve).toBeTypeOf('function'))
  act(() => window.dispatchEvent(new Event('triton:signout')))
  await act(async () => resolve(reply({ enabled: true, user })))
  expect(screen.queryByText('private')).not.toBeInTheDocument()
  expect(screen.getByRole('heading', { name: '登录工作台' })).toBeInTheDocument()
})

it('adds CSRF only to mutations and clears the UI on an expired API request', async () => {
  setAccount(user)
  const fetch = vi.fn().mockResolvedValue(reply({}, 401))
  const listener = vi.fn()
  window.addEventListener('triton:signout', listener)
  vi.stubGlobal('fetch', fetch)
  await authFetch('/api/sessions')
  await authFetch('/api/sessions/a', { method: 'DELETE' })
  expect(fetch.mock.calls[0][1].headers.has('X-CSRF-Token')).toBe(false)
  expect(fetch.mock.calls[1][1].headers.get('X-CSRF-Token')).toBe('csrf-a')
  expect(listener).toHaveBeenCalledTimes(2)
  window.removeEventListener('triton:signout', listener)
})

it('does not let a background identity check interrupt an in-flight login', async () => {
  let finishLogin!: (response: Response) => void
  const fetch = vi.fn().mockImplementation((path: string) =>
    path.endsWith('/login')
      ? new Promise<Response>(resolve => {
          finishLogin = resolve
        })
      : Promise.resolve(reply({}, 401)),
  )
  vi.stubGlobal('fetch', fetch)
  render(
    <AccountGate>
      <div>private workspace</div>
    </AccountGate>,
  )
  await screen.findByRole('heading', { name: '登录工作台' })
  fireEvent.change(screen.getByLabelText('用户名'), { target: { value: 'alice' } })
  fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'long-test-password' } })
  fireEvent.click(screen.getByRole('button', { name: '登录' }))
  act(() => window.dispatchEvent(new Event('pageshow')))
  expect(fetch).toHaveBeenCalledTimes(2)
  await act(async () => finishLogin(reply({ user })))
  expect(screen.getByText('private workspace')).toBeInTheDocument()
})

it('remounts private component state when another tab changes accounts', async () => {
  const bob = { id: 'b', username: 'bob', csrf_token: 'csrf-b' }
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(reply({ enabled: true, user }))
    .mockResolvedValueOnce(reply({ enabled: true, user: bob }))
  vi.stubGlobal('fetch', fetch)
  render(
    <AccountGate>
      <input aria-label="private draft" defaultValue="" />
    </AccountGate>,
  )
  fireEvent.change(await screen.findByLabelText('private draft'), { target: { value: 'alice-only draft' } })
  act(() => window.dispatchEvent(new Event('pageshow')))
  await screen.findByText('bob')
  expect(screen.getByLabelText('private draft')).toHaveValue('')
})
