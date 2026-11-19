import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { KnowledgeLibrary } from './KnowledgeLibrary'
import { KnowledgeDetail, KnowledgeSync, adminApi } from './admin-api'
import { setAccount } from './auth'

const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
const initial: KnowledgeDetail = {
  id: 'doc-a', title: 'Shared guide', source: 'Confirmed source', content: 'Old draft', latest_version: 2,
  published_version: 1, state: 'draft', chunk_count: 0, error: null, updated_at: '2026-11-14T00:00:00Z', catalog_revision: 7,
  chunks: [], versions: [{ version: 1, state: 'ready', chunk_count: 1, error: null }, { version: 2, state: 'draft', chunk_count: 0, error: null }],
}
let document: KnowledgeDetail
let fetchMock: ReturnType<typeof vi.fn>
let receiptState: KnowledgeSync['state']
let guard: (() => boolean) | null
const forbidden = vi.fn()
const pathOf = (path: string) => new URL(path, 'http://localhost').pathname
const posts = () => fetchMock.mock.calls.filter(([path, options]) => pathOf(path) === '/api/admin/knowledge/confirm' && options?.method === 'POST')
const receipts = () => fetchMock.mock.calls.filter(([path]) => path.includes('/knowledge/operations/'))
function defaultFetch(path: string, options?: RequestInit): Response {
  const route = pathOf(path)
  if (route === '/api/admin/knowledge') return reply({ enabled: true, total: 1, documents: [document] })
  if (route === '/api/admin/knowledge/confirm') {
    const data = JSON.parse(String(options?.body))
    document = { ...document, title: data.title, content: data.content ?? '', source: data.source,
      ...(data.sections ? { sections: data.sections.map((section: {id: string; text: string}) => ({ ...section, locator: section.id, edited: true })) } : {}),
      latest_version: data.document_id ? data.expected_version + 1 : 1, state: 'pending',
      sync: { state: 'pending', request_id: data.request_id, job_id: 'job-a', version: data.document_id ? data.expected_version + 1 : 1, error: null } }
    return reply(document, 202)
  }
  if (route.startsWith('/api/admin/knowledge/operations/')) {
    document = { ...document, state: receiptState === 'published' ? 'ready' : receiptState === 'failed' ? 'failed' : 'pending',
      published_version: receiptState === 'published' ? document.latest_version : document.published_version,
      sync: { ...document.sync!, state: receiptState, error: receiptState === 'failed' ? 'Embedding unavailable' : null } }
    return reply(document)
  }
  if (route === '/api/admin/knowledge/doc-a') return reply(document)
  throw new Error('Unexpected test request')
}
beforeEach(() => {
  document = structuredClone(initial); receiptState = 'published'; guard = null; forbidden.mockClear()
  setAccount({ id: 'admin-a', username: 'admin', csrf_token: 'csrf', is_admin: true })
  fetchMock = vi.fn(async (path: string, options?: RequestInit) => defaultFetch(path, options))
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => { cleanup(); setAccount(null); vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.useRealTimers() })
async function open(existing = true) {
  render(<KnowledgeLibrary onForbidden={forbidden} onGuardChange={value => { guard = value }} />)
  await screen.findByLabelText('标题')
  if (existing) {
    fireEvent.click(screen.getByRole('button', { name: /Shared guide/ }))
    await waitFor(() => expect(screen.getByLabelText('标题')).toHaveValue('Shared guide'))
  }
}
async function tick(ms = 3000) { await act(async () => { await vi.advanceTimersByTimeAsync(ms) }) }

describe('confirmed shared RAG updates', () => {
  it('submits one atomic confirmation and polls until publication while keeping the old version', async () => {
    await open()
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'Verified new content' } })
    fireEvent.click(screen.getByRole('button', { name: '确认修改并更新 RAG' }))
    await screen.findByText(/正在自动更新共享 RAG/)
    expect(posts()).toHaveLength(1)
    expect(JSON.parse(String(posts()[0][1].body))).toMatchObject({ document_id: 'doc-a', content: 'Verified new content', expected_version: 2, catalog_revision: 7, request_id: expect.any(String) })
    expect(screen.getByText('已发布版本 v1')).toBeInTheDocument()
    expect(screen.getByLabelText('正文')).toBeDisabled()
    vi.useFakeTimers()
    // Resume from the actor-bound stored intent, as on a normal editor refresh.
    cleanup(); render(<KnowledgeLibrary onForbidden={forbidden} requestedDocumentId="doc-a" navigationKey={1} />)
    await tick(0)
    await tick()
    expect(receipts()).toHaveLength(1)
    expect(screen.getByText('v3 已发布到共享 RAG，所有用户后续检索可使用。')).toBeInTheDocument()
    expect(posts()).toHaveLength(1)
    expect(fetchMock.mock.calls.some(([path]) => /\/(index|publish)$/.test(pathOf(path)))).toBe(false)
  })

  it('stops after a failed receipt and explicitly preserves the prior published version', async () => {
    document.sync = { state: 'pending', request_id: 'request-existing', job_id: 'job-a', version: 2, error: null }
    document.state = 'pending'; receiptState = 'failed'
    vi.useFakeTimers()
    render(<KnowledgeLibrary onForbidden={forbidden} requestedDocumentId="doc-a" navigationKey={1} />)
    await tick(0); await tick()
    expect(screen.getByText(/本次更新失败，原有已发布版本仍可使用/)).toHaveTextContent('Embedding unavailable')
    expect(screen.getByText('已发布版本 v1')).toBeInTheDocument()
    await tick(15000)
    expect(receipts()).toHaveLength(1)
    expect(posts()).toHaveLength(0)
    expect(screen.getByRole('button', { name: '确认修改并更新 RAG' })).toBeEnabled()
  })

  it('continues polling after a newly confirmed document becomes the selected editor', async () => {
    await open(false)
    vi.useFakeTimers()
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'New shared guide' } })
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'New verified source' } })
    fireEvent.click(screen.getByRole('button', { name: '确认加入共享 RAG' }))
    await tick(0)
    expect(screen.getByText(/正在自动更新共享 RAG/)).toBeInTheDocument()
    await tick()
    expect(screen.getByText('v1 已发布到共享 RAG，所有用户后续检索可使用。')).toBeInTheDocument()
    expect(posts()).toHaveLength(1)
    expect(receipts()).toHaveLength(1)
  })

  it('pauses automatic polling after a bounded number of pending receipts', async () => {
    document.sync = { state: 'pending', request_id: 'request-existing', job_id: 'job-a', version: 2, error: null }
    document.state = 'pending'; receiptState = 'pending'
    vi.useFakeTimers()
    render(<KnowledgeLibrary onForbidden={forbidden} requestedDocumentId="doc-a" navigationKey={1} />)
    await tick(0)
    // Process each timer turn separately so React can render state changes.
    for (let i = 0; i < 31; i++) await tick()
    expect(screen.getByText('自动核对已暂停，后台仍可能处理中。请稍后核对更新结果。')).toBeInTheDocument()
    expect(receipts()).toHaveLength(30)
    await tick(15000)
    expect(receipts()).toHaveLength(30)
    expect(posts()).toHaveLength(0)
  })

  it('keeps unknown results blocked and retries only the exact original request on explicit action', async () => {
    let unknown = true
    fetchMock.mockImplementation(async (path, options) => {
      if (unknown && pathOf(path) === '/api/admin/knowledge/confirm') throw new TypeError('network interrupted')
      if (unknown && path.includes('/knowledge/operations/')) return reply({ detail: 'Unknown request' }, 404)
      return defaultFetch(path, options)
    })
    await open()
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'Keep this exact draft' } })
    fireEvent.click(screen.getByRole('button', { name: '确认修改并更新 RAG' }))
    await screen.findByText(/尚未查到确认请求/)
    expect(posts()).toHaveLength(1)
    expect(receipts()).toHaveLength(1)
    expect(guard?.()).toBe(false)
    expect(screen.getByLabelText('正文')).toHaveValue('Keep this exact draft')
    expect(screen.getByLabelText('正文')).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '新建文档' }))
    expect(screen.getByLabelText('正文')).toHaveValue('Keep this exact draft')
    unknown = false
    fireEvent.click(screen.getByRole('button', { name: '使用原请求重试' }))
    await screen.findByText(/正在自动更新共享 RAG/)
    expect(posts()).toHaveLength(2)
    expect(posts()[1][1].body).toEqual(posts()[0][1].body)
  })

  it('uses reviewed sections in the confirmation without losing uploaded source structure', async () => {
    document.sections = [{ id: 'page-1', locator: 'Page 1', text: 'Parsed page', edited: false }]
    document.source_file = { filename: 'guide.pdf', format: 'pdf', sha256: 'a'.repeat(64), parser_version: 'v1', warnings: [], reviewed: false }
    document.state = 'review'
    await open()
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: 'Reviewed page' } })
    fireEvent.click(screen.getByRole('button', { name: '确认修改并更新 RAG' }))
    await screen.findByText(/正在自动更新共享 RAG/)
    expect(JSON.parse(String(posts()[0][1].body))).toMatchObject({ sections: [{ id: 'page-1', text: 'Reviewed page' }] })
    expect(JSON.parse(String(posts()[0][1].body))).not.toHaveProperty('content')
  })

  it.each(['superseded', 'withdrawn'] as const)('stops receipt polling for a %s intent without claiming successful publication', async state => {
    document.sync = { state: 'pending', request_id: 'request-existing', job_id: 'job-a', version: 2, error: null }
    document.state = 'pending'; receiptState = state
    vi.useFakeTimers()
    render(<KnowledgeLibrary onForbidden={forbidden} requestedDocumentId="doc-a" navigationKey={1} />)
    await tick(0); await tick(); await tick(15000)
    expect(receipts()).toHaveLength(1)
    expect(screen.getAllByText(state === 'superseded' ? '本次确认已被更新版本替代，未自动发布旧版本。' : '本次更新已撤销，未继续发布。').length).toBeGreaterThan(0)
    expect(screen.queryByText(/已发布到共享 RAG/)).not.toBeInTheDocument()
  })

  it('bounds confirmation submission at 30 seconds without silently treating it as a rejection', async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation((_path, options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))))
    const request = adminApi.confirm({ title: 'Guide', source: '', content: 'Verified' }, 'fixed-request')
    const check = expect(request).rejects.toThrow('结果尚未确认')
    await vi.advanceTimersByTimeAsync(30000)
    await check
  })
})
