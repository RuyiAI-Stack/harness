import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AccountGate } from './AccountGate'
import { AccountUser, setAccount } from './auth'
import { UsageDashboard } from './UsageDashboard'
import { AccountUsage, readAccountUsage } from './usage-api'

const alice = { id: 'alice-id', username: 'alice', csrf_token: 'alice-csrf' }
const bob = { id: 'bob-id', username: 'bob', csrf_token: 'bob-csrf' }
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
const usage = (user = alice): AccountUsage => ({
  account: { id: user.id, username: user.username, status: 'active' },
  totals: { sessions: 3, runs: 12, failed_runs: 2 },
  last_used_at: '2026-11-14T02:01:00Z',
  recent_runs: [{ id: `${user.username}-private-run`, status: 'failed', created_at: '2026-11-13T01:00:00Z', updated_at: '2026-11-14T02:01:00Z' }],
  as_of: '2026-11-14T02:10:00Z',
})
let account: AccountUser | null
let fetchMock: ReturnType<typeof vi.fn>

function response(path: string): Response {
  if (path === '/api/auth/me') return reply({ enabled: true, user: account })
  if (path === '/api/account/usage') return reply(usage(account || alice))
  if (path.startsWith('/api/admin/overview')) return reply({ users: [], totals: { users: 2, sessions: 5, runs: 25, failed_runs: 3 } })
  if (path === '/api/admin/rag/sources') return reply({ default_source_id: 'current', sources: [{ id: 'current', label: '当前资料库', status: 'configured', kind: 'current' }] })
  if (path.startsWith('/api/admin/rag/catalog/summary')) return reply({ total: 0, eligible: 0, pending: 0, attention: 0, excluded: 0, complete: true, source_health: [], scopes: [], catalog_revision: '1', as_of: '' })
  if (path.startsWith('/api/admin/rag/catalog')) return reply({ items: [], total: 0, next_cursor: null, complete: true, source_health: [], scopes: [], catalog_revision: '1', as_of: '' })
  if (path.startsWith('/api/admin/knowledge')) return reply({ documents: [], total: 0, enabled: true })
  throw new Error(`Unexpected request ${path}`)
}

beforeEach(() => {
  account = alice
  setAccount(alice)
  fetchMock = vi.fn().mockImplementation(async (path: string) => response(path))
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  cleanup()
  setAccount(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('personal usage entry and isolation', () => {
  it('lets a normal user read only their own usage without requesting admin or RAG data', async () => {
    render(<AccountGate><input aria-label="chat draft" defaultValue="keep me" /></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '我的使用情况' }))
    expect(await screen.findByText('alice-private-run')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '管理后台' })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'RAG 资料' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '确认修改并更新 RAG' })).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.every(([path]) => path === '/api/auth/me' || path === '/api/account/usage')).toBe(true)
    expect(fetchMock.mock.calls.find(([path]) => path === '/api/account/usage')?.[1].headers.get('X-Triton-Account')).toBe(alice.id)
    fireEvent.click(screen.getByRole('button', { name: '返回对话' }))
    expect(screen.getByLabelText('chat draft')).toBeVisible()
    expect(screen.getByLabelText('chat draft')).toHaveValue('keep me')
  })

  it.each(['account', 'csrf'] as const)('clears personal statistics on %s identity change and does not resurrect old data', async kind => {
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '我的使用情况' }))
    await screen.findByText('alice-private-run')
    account = kind === 'account' ? bob : { ...alice, csrf_token: 'new-csrf' }
    await act(async () => window.dispatchEvent(new Event('pageshow')))
    expect(screen.queryByRole('main', { name: '我的使用情况' })).not.toBeInTheDocument()
    expect(screen.queryByText('alice-private-run')).not.toBeInTheDocument()
    account = alice
    await act(async () => window.dispatchEvent(new Event('pageshow')))
    expect(screen.getByRole('button', { name: '我的使用情况' })).toHaveAttribute('aria-expanded', 'false')
  })

  it('discards a pending usage response when a different account signs in', async () => {
    let finish!: (value: Response) => void
    fetchMock.mockImplementation((path: string) => path === '/api/account/usage'
      ? new Promise<Response>(resolve => { finish = resolve }) : Promise.resolve(response(path)))
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '我的使用情况' }))
    await waitFor(() => expect(finish).toBeTypeOf('function'))
    account = bob
    await act(async () => window.dispatchEvent(new Event('pageshow')))
    await act(async () => finish(reply(usage(alice))))
    expect(screen.queryByText('alice-private-run')).not.toBeInTheDocument()
    expect(screen.getByText('bob')).toBeInTheDocument()
  })

  it('does not mount the personal panel in unauthenticated local mode', async () => {
    fetchMock.mockResolvedValue(reply({ enabled: false, user: null }))
    render(<AccountGate><div>local workspace</div></AccountGate>)
    await screen.findByText('local workspace')
    expect(screen.queryByRole('button', { name: '我的使用情况' })).not.toBeInTheDocument()
  })

  it('keeps the usage shortcut disabled while admin edits may have unsaved data', async () => {
    account = { ...alice, is_admin: true }
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '管理后台' }))
    await screen.findByRole('main', { name: '管理后台' })
    expect(screen.getByRole('button', { name: '我的使用情况' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '我的使用情况' }))
    expect(fetchMock.mock.calls.some(([path]) => path === '/api/account/usage')).toBe(false)
    fireEvent.click(screen.getByRole('tab', { name: '账号概览' }))
    expect(await screen.findByText('全部账号')).toBeInTheDocument()
    expect(screen.getByText('管理员可查看本服务全部账号的使用统计；普通用户只能查看自己的记录。')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '返回对话' }))
    expect(screen.getByRole('button', { name: '我的使用情况' })).toBeEnabled()
  })
})

describe('personal usage presentation and failure handling', () => {
  it('shows counts, scoped recent runs, readable statuses and no fabricated billing', async () => {
    render(<UsageDashboard accountId={alice.id} onClose={vi.fn()} />)
    await screen.findByText('alice-private-run')
    const main = screen.getByRole('main', { name: '我的使用情况' })
    expect(within(main).getByText('会话记录').parentElement).toHaveTextContent('3')
    expect(within(main).getByText('运行记录').parentElement).toHaveTextContent('12')
    expect(within(main).getByText('失败运行').parentElement).toHaveTextContent('2')
    expect(within(main).getByText('失败', { exact: true })).toBeInTheDocument()
    expect(within(main).getByText('正常')).toBeInTheDocument()
    expect(within(main).getByText('这里展示会话与运行记录，暂不统计模型 Token 或费用。')).toBeInTheDocument()
  })

  it('shows a useful zero-history state', async () => {
    fetchMock.mockResolvedValue(reply({ ...usage(), totals: { sessions: 0, runs: 0, failed_runs: 0 }, recent_runs: [], last_used_at: null }))
    render(<UsageDashboard accountId={alice.id} onClose={vi.fn()} />)
    expect(await screen.findByText('还没有运行记录')).toBeInTheDocument()
    expect(screen.getByText('未使用')).toBeInTheDocument()
  })

  it('clears prior results while refreshing and after a failure, then retries explicitly', async () => {
    render(<UsageDashboard accountId={alice.id} onClose={vi.fn()} />)
    await screen.findByText('alice-private-run')
    fetchMock.mockResolvedValue(reply({}, 503))
    fireEvent.click(screen.getByRole('button', { name: '刷新我的使用情况' }))
    expect(screen.queryByText('alice-private-run')).not.toBeInTheDocument()
    expect(await screen.findByRole('alert')).toHaveTextContent('使用情况暂时无法读取')
    fetchMock.mockResolvedValue(reply(usage()))
    fireEvent.click(screen.getByRole('button', { name: '重新读取' }))
    await screen.findByText('alice-private-run')
  })

  it('does not render a response for a different account', async () => {
    fetchMock.mockResolvedValue(reply(usage(bob)))
    render(<UsageDashboard accountId={alice.id} onClose={vi.fn()} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('账号已变化')
    expect(screen.queryByText('bob-private-run')).not.toBeInTheDocument()
  })

  it('bounds unresponsive requests at 15 seconds without retrying', async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}))
    const controller = new AbortController()
    const request = readAccountUsage(alice.id, controller.signal)
    const result = expect(request).rejects.toThrow('使用情况读取超时')
    await vi.advanceTimersByTimeAsync(15000)
    await result
    expect(fetchMock).toHaveBeenCalledOnce()
    expect(fetchMock.mock.calls[0][1].signal.aborted).toBe(true)
  })
})
