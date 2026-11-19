import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useLayoutEffect } from 'react'
import { AdminDashboard } from './AdminDashboard'
import { AccountGate } from './AccountGate'
import { AccountUser, setAccount } from './auth'
import { adminApi, KnowledgeDetail, sourceFileByteLimit, validateDraft } from './admin-api'

const admin = { id: 'admin-a', username: 'alice', csrf_token: 'csrf-a', is_admin: true }
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
const overview = {
  users: [{ id: 'user-a', username: 'worker', status: 'active', last_used_at: null, session_count: 2, run_count: 4, failed_run_count: 1 }],
  totals: { users: 1, sessions: 2, runs: 4, failed_runs: 1 },
}
const initialDocument: KnowledgeDetail = {
  id: 'doc-a', title: 'Operator notes', source: '', content: 'Plain <script>text</script>',
  latest_version: 2, published_version: 1, state: 'draft', chunk_count: 0, error: null, updated_at: '2026-11-10T01:00:00Z',
  chunks: [], versions: [{ version: 2, state: 'draft', chunk_count: 0, error: null }, { version: 1, state: 'ready', chunk_count: 1, error: null }],
}
const parsedSections = [
  { id: 'page-1', locator: 'Page 1', text: 'First page <script>plain text</script>', edited: false },
  { id: 'paragraph-2', locator: 'Paragraph 2', text: 'Second paragraph', edited: false },
  { id: 'table-3', locator: 'Table 3', text: 'Third table', edited: false },
]
const sourceFile = {
  filename: 'manual.pdf', format: 'pdf', sha256: 'a'.repeat(64), parser_version: 'parser-v1',
  reviewed: false, warnings: ['Page 2 contains an image.', '<script>warning text</script>'],
}
function importedDocument(state = 'review'): KnowledgeDetail {
  return { ...structuredClone(initialDocument), state, published_version: null,
    source_file: structuredClone(sourceFile), content: '',
    sections: state === 'review' ? structuredClone(parsedSections) : [] }
}
let document: KnowledgeDetail
let documents: KnowledgeDetail[]
let account: AccountUser | null
let enabled: boolean
let fetchMock: ReturnType<typeof vi.fn>

function defaultFetch(path: string, options?: RequestInit): Response {
  const url = new URL(path, 'http://localhost')
  const offset = Number(url.searchParams.get('offset') || 0)
  path = url.pathname
  if (path === '/api/auth/me') return account ? reply({ enabled: true, user: account }) : reply({ enabled: false, user: null })
  if (path === '/api/auth/logout') return reply({ signed_out: true })
  if (path === '/api/admin/overview') return reply(overview)
  if (path === '/api/admin/rag/sources') return reply({ default_source_id: 'current', sources: [{ id: 'current', label: '当前资料库', read_only: false, status: 'configured', kind: 'current' }] })
  if (path === '/api/admin/rag/catalog') return reply({ items: [], total: 0, next_cursor: null, complete: true, source_health: [], scopes: [], catalog_revision: 'test-r1', as_of: '' })
  if (path === '/api/admin/rag/catalog/summary') return reply({ total: 0, eligible: 0, pending: 0, attention: 0, excluded: 0, complete: true, source_health: [], scopes: [], catalog_revision: 'test-r1', as_of: '' })
  if (path === '/api/admin/knowledge' && options?.method !== 'POST') {
    const q = (url.searchParams.get('q') || '').toLowerCase()
    const filtered = documents.filter(doc => (doc.state === 'deleted') === (url.searchParams.get('view') === 'trash'))
      .filter(doc => `${doc.title} ${doc.source}`.toLowerCase().includes(q))
      .filter(doc => url.searchParams.get('status') === 'published' ? doc.published_version !== null
        : url.searchParams.get('status') === 'unpublished' ? doc.published_version === null : true)
    return reply({ enabled, documents: filtered.slice(offset, offset + 50), offset, total: filtered.length })
  }
  if (path === '/api/admin/knowledge/files') {
    document = { ...importedDocument('parsing'), latest_version: 1,
      title: url.searchParams.get('title') || url.searchParams.get('filename')!.replace(/\.(pdf|docx)$/i, ''),
      source: url.searchParams.get('source') || '' }
    documents = [document]
    const { content: _content, source_file: _source, sections: _sections, chunks: _chunks, versions: _versions, ...summary } = document
    return reply(summary)
  }
  if (path === '/api/admin/knowledge' && options?.method === 'POST') {
    const body = JSON.parse(String(options.body))
    document = { ...document, title: body.title, source: body.source, content: body.content,
      latest_version: body.document_id ? body.expected_version + 1 : 1, published_version: body.document_id ? document.published_version : null,
      state: 'draft', chunks: [], chunk_count: 0 }
    if (body.sections) {
      document = { ...document, content: body.sections.map((section: { text: string }) => section.text).join('\n'),
        source_file: { ...document.source_file!, reviewed: true },
        sections: document.sections!.map((section, index) => ({ ...section, text: body.sections[index].text,
          edited: section.edited || section.text !== body.sections[index].text })) }
    }
    documents = [document]
    return reply(document)
  }
  if (path.endsWith('/index')) { document = { ...document, state: 'pending' }; documents = [document]; return reply(document) }
  if (path.endsWith('/publish')) { document = { ...document, published_version: document.latest_version }; documents = [document]; return reply(document) }
  if (path.endsWith('/unpublish')) { document = { ...document, published_version: null }; documents = [document]; return reply(document) }
  if (path.endsWith('/delete') || path.endsWith('/restore')) {
    document = { ...document, state: path.endsWith('/delete') ? 'deleted' : 'draft',
      published_version: null, catalog_revision: (document.catalog_revision ?? 0) + 1 }
    documents = [document]
    return reply(document)
  }
  if (path === '/api/admin/knowledge/doc-a') return reply(document)
  if (path === '/api/admin/knowledge/doc-a/original') return new Response('private original', { headers: { 'Content-Type': 'application/pdf' } })
  if (path === '/api/knowledge/search') return reply({ status: 'ready', items: [{ document_id: document.id, version: document.published_version, title: document.title, source: document.source, text: 'matched text', chunk_id: 'chunk-a', score: 0.8 }], context: 'retrieved context' })
  throw new Error(`Unexpected request: ${path}`)
}

beforeEach(() => {
  document = structuredClone(initialDocument)
  documents = [document]
  account = admin
  enabled = true
  setAccount(admin)
  fetchMock = vi.fn(async (path: string, options?: RequestInit) => defaultFetch(path, options))
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  cleanup()
  setAccount(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

async function library() {
  render(<AdminDashboard onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('tab', { name: '文档管理' }))
  await screen.findByLabelText('标题')
}
async function openDocument() {
  await library()
  fireEvent.click(screen.getByRole('button', { name: /Operator notes/ }))
  await waitFor(() => expect(screen.getByLabelText('标题')).toHaveValue(document.title))
}
const posts = (path: string) => fetchMock.mock.calls.filter(([url, options]) => url === path && options?.method === 'POST')

describe('knowledge catalog management', () => {
  it('filters existing documents by title/source and publication status', async () => {
    documents = [document, { ...document, id: 'other', title: 'Other guide', source: 'manual', published_version: null }]
    await library()
    fireEvent.change(screen.getByRole('combobox', { name: '发布状态' }), { target: { value: 'unpublished' } })
    await waitFor(() => expect(screen.queryByRole('button', { name: /Operator notes/ })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Other guide/ })).toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: '查找资料' }), { target: { value: 'missing' } })
    fireEvent.click(screen.getByRole('button', { name: '查找资料' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /Other guide/ })).not.toBeInTheDocument())
    expect(fetchMock.mock.calls.some(([url]) => url.includes('q=missing') && url.includes('status=unpublished'))).toBe(true)
  })

  it('requires confirmation, deletes with version/revision and clears displayed retrieval results', async () => {
    await openDocument()
    fireEvent.change(screen.getByLabelText('查询'), { target: { value: 'notes' } })
    fireEvent.click(screen.getByRole('button', { name: '搜索知识库' }))
    await screen.findByText('matched text')
    fireEvent.click(screen.getByRole('button', { name: '删除资料' }))
    expect(screen.getByRole('alertdialog', { name: '删除知识资料' })).toHaveTextContent('Operator notes')
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(posts('/api/admin/knowledge/doc-a/delete')).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: '删除资料' }))
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /Operator notes/ })).not.toBeInTheDocument())
    expect(JSON.parse(String(posts('/api/admin/knowledge/doc-a/delete')[0][1].body))).toEqual({ version: 2, catalog_revision: 0 })
    await waitFor(() => expect(screen.queryByText('matched text')).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole('tab', { name: '回收站' }))
    fireEvent.click(await screen.findByRole('button', { name: /Operator notes/ }))
    await waitFor(() => expect(screen.getByLabelText('标题')).toBeDisabled())
    expect(screen.queryByRole('button', { name: '发布最新版本' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '恢复资料' }))
    await waitFor(() => expect(posts('/api/admin/knowledge/doc-a/restore')).toHaveLength(1))
    expect(JSON.parse(String(posts('/api/admin/knowledge/doc-a/restore')[0][1].body))).toEqual({ version: 2, catalog_revision: 1 })
    expect(posts('/api/admin/knowledge/doc-a/publish')).toHaveLength(0)
  })

  it('preserves the editor when another administrator has changed the delete target', async () => {
    await openDocument()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path.endsWith('/delete')
      ? reply({ detail: 'Document changed; reload before deleting or restoring' }, 409) : defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '删除资料' }))
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    await screen.findByText('Document changed; reload before deleting or restoring')
    expect(screen.getByLabelText('标题')).toHaveValue('Operator notes')
    expect(screen.getByRole('button', { name: /Operator notes/ })).toBeInTheDocument()
  })

  it('discards a pre-delete search response without clearing the query', async () => {
    await openDocument()
    let resolveSearch!: (value: Response) => void
    fetchMock.mockImplementation((path: string, options?: RequestInit) => path === '/api/knowledge/search'
      ? new Promise<Response>(resolve => { resolveSearch = resolve }) : Promise.resolve(defaultFetch(path, options)))
    fireEvent.change(screen.getByLabelText('查询'), { target: { value: 'preserve query' } })
    fireEvent.click(screen.getByRole('button', { name: '搜索知识库' }))
    await waitFor(() => expect(resolveSearch).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: '删除资料' }))
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    await screen.findByRole('heading', { name: '新建共享文档' })
    await act(async () => resolveSearch(reply({ status: 'ready', items: [{ document_id: 'doc-a', version: 1,
      title: 'Old evidence', source: '', text: 'deleted secret', chunk_id: 'chunk-a', score: 1 }], context: 'obsolete' })))
    expect(screen.queryByText('deleted secret')).not.toBeInTheDocument()
    expect(screen.getByLabelText('查询')).toHaveValue('preserve query')
  })

  it('refreshes publication filters after a mutation without closing the selected document', async () => {
    await library()
    fireEvent.change(screen.getByRole('combobox', { name: '发布状态' }), { target: { value: 'published' } })
    fireEvent.click(await screen.findByRole('button', { name: /Operator notes/ }))
    await waitFor(() => expect(screen.getByLabelText('标题')).toHaveValue('Operator notes'))
    fireEvent.click(screen.getByRole('button', { name: '取消发布' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /Operator notes/ })).not.toBeInTheDocument())
    expect(screen.getByLabelText('标题')).toHaveValue('Operator notes')
  })

  it('confirms discarding unsaved work before changing catalog filters', async () => {
    await openDocument()
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'unsaved evidence' } })
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    fireEvent.click(screen.getByRole('tab', { name: '回收站' }))
    expect(confirm).toHaveBeenCalled()
    expect(screen.getByRole('tab', { name: '资料库' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByLabelText('正文')).toHaveValue('unsaved evidence')
  })
})

describe('admin access and account lifetime', () => {
  it.each([undefined, false, 'true'])('does not show an admin entry for is_admin=%s', async flag => {
    account = { ...admin, is_admin: flag as boolean | undefined }
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    await screen.findByText('chat workspace')
    expect(screen.queryByRole('button', { name: '管理后台' })).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.every(([path]) => !path.startsWith('/api/admin'))).toBe(true)
  })

  it('does not expose admin in local unauthenticated mode', async () => {
    account = null
    render(<AccountGate><div>local chat</div></AccountGate>)
    await screen.findByText('local chat')
    expect(screen.queryByRole('button', { name: '管理后台' })).not.toBeInTheDocument()
  })

  it('opens admin and returns without discarding the chat draft', async () => {
    render(<AccountGate><input aria-label="chat draft" defaultValue="" /></AccountGate>)
    fireEvent.change(await screen.findByLabelText('chat draft'), { target: { value: 'keep this' } })
    fireEvent.click(screen.getByRole('button', { name: '管理后台' }))
    await screen.findByText('worker')
    expect(screen.getByLabelText('chat draft')).not.toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: '返回对话' }))
    expect(screen.getByLabelText('chat draft')).toBeVisible()
    expect(screen.getByLabelText('chat draft')).toHaveValue('keep this')
  })

  it('keeps an immediate admin click open when identity effects flush afterward', { repeats: 9 }, async () => {
    function OpenWhenReady() {
      useLayoutEffect(() => { screen.getByRole('button', { name: '管理后台' }).click() }, [])
      return <div>chat workspace</div>
    }
    render(<AccountGate><OpenWhenReady /></AccountGate>)
    await screen.findByText('worker')
    expect(screen.getByRole('button', { name: '管理后台' })).toHaveAttribute('aria-expanded', 'true')
  })

  it('keeps admin open across a refresh of the same identity', async () => {
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '管理后台' }))
    await screen.findByText('worker')
    account = { ...admin }
    await act(async () => { window.dispatchEvent(new Event('pageshow')) })
    expect(screen.getByRole('main', { name: '管理后台' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '管理后台' })).toHaveAttribute('aria-expanded', 'true')
  })

  it('does not let the active top-level admin shortcut dismiss an unsaved document', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '管理后台' }))
    fireEvent.click(screen.getByRole('tab', { name: '文档管理' }))
    fireEvent.change(await screen.findByLabelText('标题'), { target: { value: 'unsaved shared draft' } })
    const shortcut = screen.getByRole('button', { name: '管理后台' })
    expect(shortcut).toBeDisabled()
    fireEvent.click(shortcut)
    expect(screen.getByRole('main', { name: '管理后台' })).toBeInTheDocument()
    expect(screen.getByLabelText('标题')).toHaveValue('unsaved shared draft')
    expect(confirm).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: '返回对话' }))
    expect(confirm).toHaveBeenCalledOnce()
    expect(screen.getByLabelText('标题')).toHaveValue('unsaved shared draft')
  })

  it.each(['account', 'csrf', 'role'] as const)('does not resurrect admin when the original %s identity returns', async change => {
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '管理后台' }))
    fireEvent.click(screen.getByRole('tab', { name: '文档管理' }))
    fireEvent.change(await screen.findByLabelText('标题'), { target: { value: 'private admin draft' } })
    account = change === 'account' ? { ...admin, id: 'other-admin', username: 'bob' }
      : change === 'csrf' ? { ...admin, csrf_token: 'rotated-csrf' } : { ...admin, is_admin: false }
    await act(async () => { window.dispatchEvent(new Event('pageshow')) })
    expect(screen.queryByRole('main', { name: '管理后台' })).not.toBeInTheDocument()
    account = { ...admin }
    await act(async () => { window.dispatchEvent(new Event('pageshow')) })
    expect(screen.getByText('chat workspace')).toBeVisible()
    expect(screen.getByRole('button', { name: '管理后台' })).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(screen.getByRole('button', { name: '管理后台' }))
    fireEvent.click(screen.getByRole('tab', { name: '文档管理' }))
    expect(await screen.findByLabelText('标题')).toHaveValue('')
  })

  it.each(['switch', 'logout', 'revoke'] as const)('clears admin content and drafts on %s', async action => {
    render(<AccountGate><div>chat workspace</div></AccountGate>)
    fireEvent.click(await screen.findByRole('button', { name: '管理后台' }))
    fireEvent.click(screen.getByRole('tab', { name: '文档管理' }))
    fireEvent.change(await screen.findByLabelText('标题'), { target: { value: 'private draft' } })
    if (action === 'logout') {
      fireEvent.click(screen.getByRole('button', { name: '退出登录' }))
      await screen.findByRole('heading', { name: '登录工作台' })
    } else {
      account = action === 'switch' ? { ...admin, id: 'admin-b', username: 'bob', csrf_token: 'csrf-b' } : { ...admin, is_admin: false }
      act(() => window.dispatchEvent(new Event('pageshow')))
      await waitFor(() => expect(screen.queryByRole('main', { name: '管理后台' })).not.toBeInTheDocument())
      expect(screen.getByText('chat workspace')).toBeVisible()
    }
    expect(screen.queryByDisplayValue('private draft')).not.toBeInTheDocument()
    if (action === 'switch') {
      fireEvent.click(screen.getByRole('button', { name: '管理后台' }))
      fireEvent.click(screen.getByRole('tab', { name: '文档管理' }))
      expect(await screen.findByLabelText('标题')).toHaveValue('')
    }
  })

  it('clears loaded admin data on a 403 and retains the return action', async () => {
    const close = vi.fn()
    render(<AdminDashboard onClose={close} />)
    fireEvent.click(screen.getByRole('tab', { name: '账号概览' }))
    await screen.findByText('worker')
    fetchMock.mockImplementation(async () => new Response('Forbidden', { status: 403 }))
    fireEvent.click(screen.getByRole('button', { name: '刷新账号概览' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('无权访问')
    expect(screen.queryByText('worker')).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: '文档管理' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '返回对话' }))
    expect(close).toHaveBeenCalledOnce()
  })

  it('discards an old admin response after the account changes', async () => {
    let finish!: (value: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve }))
    const request = adminApi.overview()
    setAccount({ ...admin, id: 'other' })
    finish(reply(overview))
    await expect(request).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('recovers from a temporary overview failure via refresh', async () => {
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path.startsWith('/api/admin/overview') ? reply({}, 503) : defaultFetch(path, options))
    render(<AdminDashboard onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole('tab', { name: '账号概览' }))
    await screen.findByText('请求失败 (503)')
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '刷新账号概览' }))
    await screen.findByText('worker')
    expect(screen.getByRole('table')).toHaveTextContent('未使用')
  })
})

describe('knowledge library', () => {
  it('requires a new saved version after physical cleanup made the current index stale', async () => {
    document = { ...document, state: 'stale', published_version: null }
    documents = [document]
    await openDocument()
    expect(screen.getByText('此版本的索引已失效。请先保存为新版本，再建立索引；原文不会被删除。')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('最新版本 v3')
    expect(JSON.parse(String(posts('/api/admin/knowledge')[0][1].body))).toMatchObject({ expected_version: 2, content: 'Plain <script>text</script>' })
    expect(screen.getByRole('button', { name: '建立索引' })).toBeEnabled()
  })
  it('creates text with an empty source, previews, indexes, publishes and searches explicitly', async () => {
    documents = []
    await library()
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'New notes' } })
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: '<script>safe text</script>' } })
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('最新版本 v1')
    const create = posts('/api/admin/knowledge')[0]
    expect(JSON.parse(create[1].body)).toEqual({ title: 'New notes', source: '', content: '<script>safe text</script>', request_id: expect.any(String) })
    expect(create[1].headers.get('X-CSRF-Token')).toBe('csrf-a')
    expect(create[1].headers.get('X-Triton-Account')).toBe('admin-a')
    expect(create[1].headers.get('Content-Type')).toBe('application/json')
    const preview = screen.getByRole('tabpanel', { name: '文本预览' })
    expect(preview).toHaveTextContent('<script>safe text</script>')
    expect(preview.querySelector('script')).toBeNull()
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
    expect(posts('/api/admin/knowledge/doc-a/index')).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('索引已排队')
    expect(JSON.parse(posts('/api/admin/knowledge/doc-a/index')[0][1].body)).toEqual({ version: 1, request_id: expect.any(String), catalog_revision: 0 })
    document = { ...document, state: 'ready', chunk_count: 1, chunks: [{ id: 'chunk-a', text: 'indexed text', position: 0 }] }
    fireEvent.click(screen.getByRole('button', { name: '重新读取文档' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '发布最新版本' })).toBeEnabled())
    fireEvent.click(screen.getByRole('tab', { name: '分块预览' }))
    expect(screen.getByText('indexed text')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '发布最新版本' }))
    await screen.findByText('已发布版本 v1')
    expect(JSON.parse(posts('/api/admin/knowledge/doc-a/publish')[0][1].body)).toEqual({ version: 1, catalog_revision: 0 })
    fireEvent.change(screen.getByLabelText('查询'), { target: { value: '  operator  ' } })
    fireEvent.click(screen.getByRole('button', { name: '搜索知识库' }))
    await screen.findByText('matched text')
    expect(JSON.parse(posts('/api/knowledge/search')[0][1].body)).toEqual({ query: 'operator', limit: 5 })
    expect(screen.getByText('retrieved context')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '取消发布' }))
    await screen.findByText('已发布版本 无')
    expect(JSON.parse(posts('/api/admin/knowledge/doc-a/unpublish')[0][1].body)).toEqual({ catalog_revision: 0 })
  })

  it('keeps the old publication visible while a newer draft fails and saves with expected_version', async () => {
    document = { ...document, state: 'failed', error: 'Index backend unavailable' }
    documents = [document]
    await openDocument()
    expect(screen.getByText('最新版本 v2')).toBeInTheDocument()
    expect(screen.getByText('已发布版本 v1')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Index backend unavailable')
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
    fireEvent.click(screen.getByRole('tab', { name: '版本记录' }))
    expect(screen.getByRole('tabpanel', { name: '版本记录' })).toHaveTextContent('已发布')
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'revised text' } })
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('最新版本 v3')
    expect(JSON.parse(posts('/api/admin/knowledge')[0][1].body)).toMatchObject({ document_id: 'doc-a', expected_version: 2, content: 'revised text' })
    expect(screen.getByText('已发布版本 v1')).toBeInTheDocument()
  })

  it.each([null, 2])('blocks reindexing a ready version with published_version=%s until a new version is saved', async publishedVersion => {
    document = { ...document, state: 'ready', published_version: publishedVersion }
    documents = [document]
    await openDocument()
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    expect(posts('/api/admin/knowledge/doc-a/index')).toHaveLength(0)
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'new version content' } })
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '建立索引' })).toBeEnabled())
    expect(screen.getByText('最新版本 v3')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('索引已排队')
    expect(JSON.parse(posts('/api/admin/knowledge/doc-a/index')[0][1].body)).toMatchObject({ version: 3 })
  })

  it('reloads chunks and versions after saving an existing document without replacing the draft fields', async () => {
    await openDocument()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => {
      const response = defaultFetch(path, options)
      if (path === '/api/admin/knowledge' && options?.method === 'POST') {
        document = { ...document, title: 'server title', source: 'server source', content: 'server content',
          chunk_count: 1, chunks: [{ id: 'new-chunk', text: 'fresh saved-version chunk', position: 0 }],
          versions: [{ version: 3, state: 'draft', chunk_count: 1, error: null }] }
      }
      return response
    })
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'local title' } })
    fireEvent.change(screen.getByLabelText('来源'), { target: { value: 'local source' } })
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'local content' } })
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    fireEvent.click(screen.getByRole('tab', { name: '分块预览' }))
    await screen.findByText('fresh saved-version chunk')
    expect(screen.getByLabelText('标题')).toHaveValue('local title')
    expect(screen.getByLabelText('来源')).toHaveValue('local source')
    expect(screen.getByLabelText('正文')).toHaveValue('local content')
    expect(fetchMock.mock.calls.filter(([path]) => path === '/api/admin/knowledge/doc-a')).toHaveLength(2)
    fireEvent.click(screen.getByRole('tab', { name: '版本记录' }))
    expect(screen.getByRole('tabpanel', { name: '版本记录' })).toHaveTextContent('v3')
    expect(screen.getByRole('tabpanel', { name: '版本记录' })).not.toHaveTextContent('v2')
  })

  it('reports a post-save detail failure without resubmitting the saved version', async () => {
    await openDocument()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path === '/api/admin/knowledge/doc-a'
      ? reply({ detail: 'Detail refresh failed' }, 503) : defaultFetch(path, options))
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'saved content' } })
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('Detail refresh failed')
    expect(screen.getByText('已保存 v3')).toBeInTheDocument()
    expect(screen.getByLabelText('正文')).toHaveValue('saved content')
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    document = { ...document, chunk_count: 1, chunks: [{ id: 'recovered-chunk', text: 'recovered preview', position: 0 }] }
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '重新读取文档' }))
    fireEvent.click(screen.getByRole('tab', { name: '分块预览' }))
    await screen.findByText('recovered preview')
    expect(posts('/api/admin/knowledge')).toHaveLength(1)
  })

  it('retains save request_id for an unchanged failed request and replaces it on edits', async () => {
    await library()
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'Retry me' } })
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'retry content' } })
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/admin/knowledge' && options?.method === 'POST') throw new Error('connection lost')
      return defaultFetch(path, options)
    })
    for (let attempt = 0; attempt < 2; attempt++) {
      fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
      await screen.findByText('connection lost')
    }
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'changed payload' } })
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('connection lost')
    const ids = posts('/api/admin/knowledge').map(([, options]) => JSON.parse(options.body).request_id)
    expect(ids).toHaveLength(3)
    expect(ids[0]).toBe(ids[1])
    expect(ids[2]).not.toBe(ids[1])
  })

  it('retains index request_id on transport failure and keeps the pending state non-publishable', async () => {
    await openDocument()
    let attempt = 0
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path.endsWith('/index') && attempt++ === 0) throw new Error('index response lost')
      return defaultFetch(path, options)
    })
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('index response lost')
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('索引已排队')
    const requests = posts('/api/admin/knowledge/doc-a/index').map(([, options]) => JSON.parse(options.body))
    expect(requests[0]).toEqual(requests[1])
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
  })

  it('polls pending indexing without replacing unsaved edits and stops after ready', async () => {
    await openDocument()
    vi.useFakeTimers()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '建立索引' })))
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'local unsaved draft' } })
    document = { ...document, state: 'ready', chunk_count: 1, chunks: [{ id: 'indexed', text: 'new chunk', position: 0 }] }
    await act(async () => vi.advanceTimersByTimeAsync(3000))
    expect(screen.getByLabelText('正文')).toHaveValue('local unsaved draft')
    fireEvent.click(screen.getByRole('tab', { name: '分块预览' }))
    expect(screen.getByText('new chunk')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
    const calls = fetchMock.mock.calls.length
    await act(async () => vi.advanceTimersByTimeAsync(6000))
    expect(fetchMock).toHaveBeenCalledTimes(calls)
  })

  it('does not let a delayed detail response replace a new document editor', async () => {
    await library()
    let finish!: (response: Response) => void
    fetchMock.mockImplementation((path: string, options?: RequestInit) => path === '/api/admin/knowledge/doc-a'
      ? new Promise<Response>(resolve => { finish = resolve }) : Promise.resolve(defaultFetch(path, options)))
    fireEvent.click(screen.getByRole('button', { name: /Operator notes/ }))
    fireEvent.click(screen.getByRole('button', { name: '新建文档' }))
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'new local draft' } })
    await act(async () => finish(reply(document)))
    expect(screen.getByLabelText('标题')).toHaveValue('new local draft')
    expect(screen.getByLabelText('正文')).toHaveValue('')
  })

  it('suspends polling during a slow manual refresh so the editor can unlock', async () => {
    await openDocument()
    vi.useFakeTimers()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '建立索引' })))
    let finish!: (response: Response) => void
    fetchMock.mockImplementation((path: string, options?: RequestInit) => path === '/api/admin/knowledge/doc-a'
      ? new Promise<Response>(resolve => { finish = resolve }) : Promise.resolve(defaultFetch(path, options)))
    fireEvent.click(screen.getByRole('button', { name: '重新读取文档' }))
    const calls = fetchMock.mock.calls.length
    await act(async () => vi.advanceTimersByTimeAsync(9000))
    expect(fetchMock).toHaveBeenCalledTimes(calls)
    await act(async () => finish(reply({ ...document, state: 'ready' })))
    expect(screen.getByLabelText('正文')).toBeEnabled()
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeEnabled()
  })

  it('discards a late file read after selecting another document', async () => {
    await library()
    let finish!: (content: string) => void
    const file = new File(['old upload'], 'notes.txt')
    Object.defineProperty(file, 'text', { value: () => new Promise<string>(resolve => { finish = resolve }) })
    fireEvent.change(screen.getByLabelText('上传文本文件', { selector: 'input' }), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: /Operator notes/ }))
    await waitFor(() => expect(screen.getByLabelText('正文')).toHaveValue(document.content))
    await act(async () => finish('old upload'))
    expect(screen.getByLabelText('正文')).toHaveValue(document.content)
  })

  it.each(['txt', 'md'])('reads a .%s upload as text without sending it automatically', async extension => {
    await library()
    const file = new File(['file content'], `notes.${extension}`, { type: 'text/plain' })
    Object.defineProperty(file, 'text', { value: async () => 'file content' })
    fireEvent.change(screen.getByLabelText('上传文本文件', { selector: 'input' }), { target: { files: [file] } })
    await waitFor(() => expect(screen.getByLabelText('正文')).toHaveValue('file content'))
    expect(screen.getByLabelText('标题')).toHaveValue('notes')
    expect(posts('/api/admin/knowledge')).toHaveLength(0)
  })

  it('rejects unsupported and oversized uploads without replacing the draft', async () => {
    await library()
    const input = screen.getByLabelText('上传文本文件', { selector: 'input' })
    fireEvent.change(input, { target: { files: [new File(['x'], 'notes.pdf')] } })
    await screen.findByText('仅支持 .txt 和 .md 文件。')
    fireEvent.change(input, { target: { files: [new File(['x'.repeat(262145)], 'large.txt')] } })
    await screen.findByText('文件不能超过 256 KB。')
    expect(screen.getByLabelText('正文')).toHaveValue('')
  })

  it('disables over-limit saves and requires a nonblank title', async () => {
    await library()
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'x'.repeat(201) } })
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('200')
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: 'notes' } })
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'x'.repeat(60001) } })
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    expect(posts('/api/admin/knowledge')).toHaveLength(0)
  })

  it('blocks mutations and search when knowledge is disabled', async () => {
    enabled = false
    await openDocument()
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    expect(screen.getByLabelText('查询')).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '取消发布' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '新建文档' })).toBeDisabled()
  })

  it.each(['empty', 'unavailable'])('shows search %s without stale hits', async status => {
    await library()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path === '/api/knowledge/search' ? reply({ status, items: [], context: '' }) : defaultFetch(path, options))
    fireEvent.change(screen.getByLabelText('查询'), { target: { value: 'query' } })
    fireEvent.click(screen.getByRole('button', { name: '搜索知识库' }))
    await screen.findByText(status === 'empty' ? '未找到匹配的已发布内容。' : '知识检索暂不可用。')
  })

  it('preserves the structured unavailable search response on HTTP 503', async () => {
    await library()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path === '/api/knowledge/search' ? reply({ status: 'unavailable', items: [], context: '' }, 503) : defaultFetch(path, options))
    fireEvent.change(screen.getByLabelText('查询'), { target: { value: 'query' } })
    fireEvent.click(screen.getByRole('button', { name: '搜索知识库' }))
    await screen.findByText('知识检索暂不可用。')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('handles mutation 403 by removing the editor and all private data', async () => {
    await openDocument()
    fetchMock.mockImplementation(async () => reply({ detail: 'forbidden' }, 403))
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('无权访问管理后台，管理员权限可能已变更。')
    expect(screen.queryByLabelText('正文')).not.toBeInTheDocument()
  })

  it('retains a conflict draft without indexing or publishing it', async () => {
    await openDocument()
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'my unsaved revision' } })
    fetchMock.mockImplementation(async () => reply({ detail: 'Version conflict' }, 409))
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('Version conflict')
    expect(screen.getByLabelText('正文')).toHaveValue('my unsaved revision')
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
  })

  it('confirms before discarding edits when selecting another document', async () => {
    await openDocument()
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'unsaved' } })
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    fireEvent.click(screen.getByRole('button', { name: '新建文档' }))
    expect(confirm).toHaveBeenCalledOnce()
    expect(screen.getByLabelText('正文')).toHaveValue('unsaved')
  })
})

const filePosts = () => fetchMock.mock.calls.filter(([path, options]) => path.startsWith('/api/admin/knowledge/files?') && options?.method === 'POST')
const chooseFile = (file = new File(['original bytes'], 'manual.pdf')) => {
  fireEvent.change(screen.getByLabelText('选择文档或图片', { selector: 'input' }), { target: { files: [file] } })
  return file
}

describe('imported knowledge documents', () => {
  it.each(['pdf', 'DOCX', 'png', 'JPG', 'jpeg'])('uploads .%s raw bytes with auth, CSRF, encoded metadata and the caller signal', async extension => {
    const file = new File(['raw\0bytes'], `来源 & 文档.${extension}`)
    const signal = new AbortController().signal
    await adminApi.upload(file, { title: 'Title & review', source: 'https://source.test/?a=1&b=2' }, 'stable-id', signal)
    const [path, options] = filePosts()[0]
    const url = new URL(path, 'http://localhost')
    expect(Object.fromEntries(url.searchParams)).toEqual({ filename: file.name, title: 'Title & review', source: 'https://source.test/?a=1&b=2', request_id: 'stable-id' })
    expect(options.body).toBe(file)
    expect(options.body).not.toBeInstanceOf(FormData)
    expect(options.headers.get('Content-Type')).toBe('application/octet-stream')
    expect(options.headers.get('X-CSRF-Token')).toBe('csrf-a')
    expect(options.headers.get('X-Triton-Account')).toBe('admin-a')
    expect(options.credentials).toBe('same-origin')
    expect(options.signal).toBe(signal)
  })

  it('uploads, polls into review, saves an unchanged review, then explicitly indexes and publishes', async () => {
    documents = []
    await library()
    const file = chooseFile()
    expect(screen.getByLabelText('标题')).toHaveValue('manual')
    expect(filePosts()).toHaveLength(0)
    fireEvent.change(screen.getByLabelText('标题'), { target: { value: '' } })
    expect(screen.getByRole('button', { name: '上传并解析' })).toBeEnabled()
    vi.useFakeTimers()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '上传并解析' })))
    expect(screen.getByText('最新版本 v1')).toBeInTheDocument()
    expect(filePosts()[0][1].body).toBe(file)
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存审核稿' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
    expect(screen.queryByLabelText('正文')).not.toBeInTheDocument()
    document = { ...document, state: 'review', sections: structuredClone(parsedSections) }
    await act(async () => vi.advanceTimersByTimeAsync(3000))
    expect(screen.getByLabelText('审核正文')).toHaveValue(parsedSections[0].text)
    expect(screen.getByRole('button', { name: '保存审核稿' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    expect(posts('/api/admin/knowledge/doc-a/index')).toHaveLength(0)
    expect(posts('/api/admin/knowledge')).toHaveLength(0)
    const calls = fetchMock.mock.calls.length
    await act(async () => vi.advanceTimersByTimeAsync(6000))
    expect(fetchMock).toHaveBeenCalledTimes(calls)
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '保存审核稿' })))
    expect(screen.getByText('最新版本 v2')).toBeInTheDocument()
    expect(screen.getByText('已保存审核稿')).toBeInTheDocument()
    expect(JSON.parse(posts('/api/admin/knowledge')[0][1].body)).toEqual({
      title: 'manual', source: '', document_id: 'doc-a', expected_version: 1, catalog_revision: 0,
      request_id: expect.any(String), sections: parsedSections.map(({ id, text }) => ({ id, text })),
    })
    expect(posts('/api/admin/knowledge/doc-a/index')).toHaveLength(0)
    expect(screen.getByRole('button', { name: '建立索引' })).toBeEnabled()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '建立索引' })))
    document = { ...document, state: 'ready', chunk_count: 1,
      chunks: [{ id: 'chunk-1', position: 0, text: 'indexed page', locator: 'Page 1' }] }
    await act(async () => vi.advanceTimersByTimeAsync(3000))
    fireEvent.click(screen.getByRole('tab', { name: '分块预览' }))
    expect(screen.getByRole('tabpanel', { name: '分块预览' })).toHaveTextContent('Page 1')
    expect(posts('/api/admin/knowledge/doc-a/publish')).toHaveLength(0)
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '发布最新版本' })))
    expect(screen.getByText('已发布版本 v2')).toBeInTheDocument()
  })

  it('keeps exact review labels independent of populated text and section options', async () => {
    document = importedDocument()
    documents = [document]
    await openDocument()
    const text = screen.getByLabelText('审核正文', { exact: true }) as HTMLTextAreaElement
    const select = screen.getByLabelText('审核块', { exact: true }) as HTMLSelectElement
    expect(text).toHaveValue(parsedSections[0].text)
    expect(select.options).toHaveLength(parsedSections.length)
    for (const [control, name] of [[text, '审核正文'], [select, '审核块']] as const) {
      expect(control.labels).toHaveLength(1)
      const label = control.labels![0]
      expect(label.htmlFor).toBe(control.id)
      expect(label.textContent).toBe(name)
      expect(label.contains(control)).toBe(false)
      expect(control).toHaveAccessibleName(name)
    }
    fireEvent.change(select, { target: { value: '1' } })
    expect(screen.getByRole('textbox', { name: '审核正文' })).toHaveValue(parsedSections[1].text)
    fireEvent.change(text, { target: { value: 'Edited review text' } })
    expect(screen.getByLabelText('审核正文', { exact: true })).toHaveValue('Edited review text')
    expect(screen.getByRole('combobox', { name: '审核块' })).toHaveValue('1')
  })

  it('opens an already parsed long document and keeps every ordered block across pages and saved versions', async () => {
    document = importedDocument()
    document.sections = Array.from({ length: 5 }, (_, index) => ({
      id: `section-${index}`, locator: `Page ${index + 1}`, text: String(index).repeat(16000), edited: false,
    }))
    documents = [document]
    await openDocument()
    const editor = within(screen.getByLabelText('分块审核'))
    expect(screen.getAllByRole('textbox').filter(element => element.tagName === 'TEXTAREA')).toHaveLength(1)
    expect(editor.getByText('原文')).toBeInTheDocument()
    expect(screen.getByLabelText('解析警告')).toHaveTextContent('Page 2 contains an image.')
    expect(screen.getByLabelText('解析警告').querySelector('script')).toBeNull()
    expect(screen.getByRole('button', { name: '上一审核块' })).toBeDisabled()
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: 'Edited first page' } })
    expect(editor.getByText('已编辑')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '下一审核块' }))
    expect(screen.getByLabelText('审核正文')).toHaveValue('1'.repeat(16000))
    expect(editor.getByText('原文')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: '' } })
    fireEvent.change(screen.getByLabelText('审核块'), { target: { value: '4' } })
    expect(screen.getByRole('button', { name: '下一审核块' })).toBeDisabled()
    expect(editor.getByText('Page 5')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: 'Last page revision' } })
    fireEvent.change(screen.getByLabelText('审核块'), { target: { value: '0' } })
    expect(screen.getByLabelText('审核正文')).toHaveValue('Edited first page')
    fireEvent.click(screen.getByRole('button', { name: '保存审核稿' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '建立索引' })).toBeEnabled())
    const payload = JSON.parse(posts('/api/admin/knowledge')[0][1].body)
    expect(payload).not.toHaveProperty('content')
    expect(payload.sections).toEqual([
      { id: 'section-0', text: 'Edited first page' }, { id: 'section-1', text: '' },
      { id: 'section-2', text: '2'.repeat(16000) }, { id: 'section-3', text: '3'.repeat(16000) },
      { id: 'section-4', text: 'Last page revision' },
    ])
    expect(payload).toMatchObject({ document_id: 'doc-a', expected_version: 2 })
    expect(screen.queryByLabelText('正文')).not.toBeInTheDocument()
    expect(editor.getByText('已编辑')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存审核稿' })).toBeDisabled()
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: 'Later revision' } })
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '保存审核稿' }))
    await screen.findByText('最新版本 v4')
    expect(JSON.parse(posts('/api/admin/knowledge')[1][1].body).expected_version).toBe(3)
  })

  it.each(['parse_failed', 'ocr_required'])('polls to %s, renders the parser error, and blocks save/index/publish', async state => {
    document = importedDocument('parsing')
    documents = [document]
    await library()
    vi.useFakeTimers()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: /Operator notes/ })))
    document = { ...document, state, error: 'No usable text was extracted' }
    await act(async () => vi.advanceTimersByTimeAsync(3000))
    expect(screen.getByRole('alert')).toHaveTextContent('No usable text was extracted')
    expect(screen.getByRole('button', { name: '保存审核稿' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
    const calls = fetchMock.mock.calls.length
    await act(async () => vi.advanceTimersByTimeAsync(9000))
    expect(fetchMock).toHaveBeenCalledTimes(calls)
  })

  it.each(['draft', 'failed', 'review'])('does not index an unreviewed import even in state %s', async state => {
    document = { ...importedDocument(), state }
    documents = [document]
    await openDocument()
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    expect(posts('/api/admin/knowledge/doc-a/index')).toHaveLength(0)
  })

  it('retries a failed index of a reviewed import with the existing idempotency behavior', async () => {
    document = { ...importedDocument(), state: 'failed', error: 'Index failed', source_file: { ...sourceFile, reviewed: true } }
    documents = [document]
    await openDocument()
    let attempt = 0
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path.endsWith('/index') && attempt++ === 0) throw new Error('Response lost')
      return defaultFetch(path, options)
    })
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('Response lost')
    fireEvent.click(screen.getByRole('button', { name: '建立索引' }))
    await screen.findByText('索引已排队')
    expect(posts('/api/admin/knowledge/doc-a/index')[0][1].body).toBe(posts('/api/admin/knowledge/doc-a/index')[1][1].body)
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
  })

  it('reuses the upload request ID after failure and changes it only for a changed request', async () => {
    await library()
    const file = chooseFile()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path.startsWith('/api/admin/knowledge/files?')) throw new Error('Upload response lost')
      return defaultFetch(path, options)
    })
    for (let attempt = 0; attempt < 2; attempt++) {
      fireEvent.click(screen.getByRole('button', { name: '上传并解析' }))
      await screen.findByText('Upload response lost')
    }
    fireEvent.change(screen.getByLabelText('来源'), { target: { value: 'Changed source' } })
    fireEvent.click(screen.getByRole('button', { name: '上传并解析' }))
    await screen.findByText('Upload response lost')
    chooseFile(new File(['different bytes'], 'manual.pdf'))
    fireEvent.click(screen.getByRole('button', { name: '上传并解析' }))
    await screen.findByText('Upload response lost')
    const requests = filePosts()
    const ids = requests.map(([path]) => new URL(path, 'http://localhost').searchParams.get('request_id'))
    expect(ids).toHaveLength(4)
    expect(ids[0]).toBe(ids[1])
    expect(ids[2]).not.toBe(ids[1])
    expect(ids[3]).not.toBe(ids[2])
    expect(requests[0][1].body).toBe(file)
    expect(requests[1][1].body).toBe(file)
    expect(screen.getByText('manual.pdf')).toBeInTheDocument()
  })

  it('confirms leaving a pending upload, aborts it, and discards a late success', async () => {
    await library()
    chooseFile()
    let finish!: (response: Response) => void
    fetchMock.mockImplementation((path: string, options?: RequestInit) => path.startsWith('/api/admin/knowledge/files?')
      ? new Promise<Response>(resolve => { finish = resolve }) : Promise.resolve(defaultFetch(path, options)))
    fireEvent.click(screen.getByRole('button', { name: '上传并解析' }))
    fireEvent.click(screen.getByRole('button', { name: '上传中...' }))
    expect(filePosts()).toHaveLength(1)
    const signal = filePosts()[0][1].signal as AbortSignal
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    fireEvent.click(screen.getByRole('button', { name: /Operator notes/ }))
    expect(confirm).toHaveBeenCalledOnce()
    expect(signal.aborted).toBe(false)
    const beforeUnload = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(beforeUnload)
    expect(beforeUnload.defaultPrevented).toBe(true)
    confirm.mockReturnValue(true)
    fireEvent.click(screen.getByRole('button', { name: /Operator notes/ }))
    await waitFor(() => expect(screen.getByLabelText('正文')).toHaveValue(initialDocument.content))
    expect(signal.aborted).toBe(true)
    await act(async () => finish(reply({ ...importedDocument('parsing'), id: 'late-upload' })))
    expect(screen.getByLabelText('正文')).toHaveValue(initialDocument.content)
    expect(screen.queryByLabelText('审核正文')).not.toBeInTheDocument()
  })

  it('preserves section edits after a version conflict and confirms before discarding them', async () => {
    document = importedDocument()
    documents = [document]
    await openDocument()
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: 'Unsaved review' } })
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path === '/api/admin/knowledge' && options?.method === 'POST'
      ? reply({ detail: 'Review version conflict' }, 409) : defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '保存审核稿' }))
    await screen.findByText('Review version conflict')
    expect(screen.getByLabelText('审核正文')).toHaveValue('Unsaved review')
    expect(screen.getByRole('button', { name: '建立索引' })).toBeDisabled()
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    fireEvent.click(screen.getByRole('button', { name: '重新读取文档' }))
    fireEvent.click(screen.getByRole('button', { name: '新建文档' }))
    expect(confirm).toHaveBeenCalledTimes(2)
    expect(screen.getByLabelText('审核正文')).toHaveValue('Unsaved review')
  })

  it('does not replace an edited block while polling an indexed import', async () => {
    document = { ...importedDocument(), state: 'draft', source_file: { ...sourceFile, reviewed: true } }
    documents = [document]
    await openDocument()
    vi.useFakeTimers()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '建立索引' })))
    fireEvent.change(screen.getByLabelText('审核正文'), { target: { value: 'Local revision' } })
    document = { ...document, state: 'ready' }
    await act(async () => vi.advanceTimersByTimeAsync(3000))
    expect(screen.getByLabelText('审核正文')).toHaveValue('Local revision')
    expect(screen.getByRole('button', { name: '发布最新版本' })).toBeDisabled()
  })

  it.each([['unsupported.doc', 1, 'PDF、DOCX、PNG'], ['large.pdf', sourceFileByteLimit + 1, '12 MiB'], ['empty.docx', 0, '不能为空']] as const)(
    'rejects %s before sending bytes', async (name, size, error) => {
      await library()
      const file = new File(['x'], name)
      Object.defineProperty(file, 'size', { value: size })
      chooseFile(file)
      expect(screen.getByRole('alert')).toHaveTextContent(error)
      expect(filePosts()).toHaveLength(0)
      await expect(adminApi.upload(file, { title: 'title', source: '' }, 'unused')).rejects.toThrow(error)
      expect(filePosts()).toHaveLength(0)
    },
  )

  it('accepts the exact 12 MiB boundary', async () => {
    const file = new File(['x'], 'boundary.pdf')
    Object.defineProperty(file, 'size', { value: sourceFileByteLimit })
    await adminApi.upload(file, { title: '', source: '' }, 'boundary-id')
    expect(filePosts()).toHaveLength(1)
  })

  it.each([413, 415, 503])('shows upload HTTP %s errors and retains the file for retry', async status => {
    await library()
    chooseFile()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path.startsWith('/api/admin/knowledge/files?')
      ? reply({ detail: `Upload rejected ${status}` }, status) : defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '上传并解析' }))
    await screen.findByText(`Upload rejected ${status}`)
    expect(screen.getByText('manual.pdf')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '上传并解析' })).toBeEnabled()
    expect(posts('/api/admin/knowledge')).toHaveLength(0)
  })

  it('removes private data on an upload 403', async () => {
    await library()
    chooseFile()
    fetchMock.mockImplementation(async () => reply({}, 403))
    fireEvent.click(screen.getByRole('button', { name: '上传并解析' }))
    await screen.findByText('无权访问管理后台，管理员权限可能已变更。')
    expect(screen.queryByText('manual.pdf')).not.toBeInTheDocument()
  })

  it('renders the source locator on search results', async () => {
    await library()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path === '/api/knowledge/search'
      ? reply({ status: 'ready', items: [{ document_id: 'doc-a', version: 1, title: 'Manual', source: 'manual.pdf', locator: 'Table 4', text: 'cell text', chunk_id: 'c4', score: 0.7 }], context: '' })
      : defaultFetch(path, options))
    fireEvent.change(screen.getByLabelText('查询'), { target: { value: 'cell' } })
    fireEvent.click(screen.getByRole('button', { name: '搜索知识库' }))
    expect(await screen.findByText(/manual.pdf · Table 4/)).toBeInTheDocument()
  })

  it('downloads originals only through authenticated fetch and revokes the blob URL', async () => {
    document = importedDocument()
    documents = [document]
    await openDocument()
    const create = vi.fn(() => 'blob:private-original')
    const revoke = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: create })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.download).toBe('manual.pdf')
      expect(this.href).toBe('blob:private-original')
    })
    fireEvent.click(screen.getByRole('button', { name: '下载原文件' }))
    await waitFor(() => expect(click).toHaveBeenCalledOnce())
    const request = fetchMock.mock.calls.find(([path]) => path === '/api/admin/knowledge/doc-a/original')!
    expect(request[1].headers.get('X-Triton-Account')).toBe('admin-a')
    expect(request[1].credentials).toBe('same-origin')
    expect(create).toHaveBeenCalledOnce()
    expect(revoke).toHaveBeenCalledWith('blob:private-original')
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
  })
})

describe('review validation', () => {
  const draft = (texts: string[]) => ({ title: 'Review', source: '', sections: texts.map((text, index) => ({ id: String(index), text })) })
  it('accepts long documents and blank individual blocks but rejects all-blank text', () => {
    expect(validateDraft(draft(['', ...Array(5).fill('x'.repeat(20000))]))).toBe('')
    expect(validateDraft(draft(['', ' \n']))).toContain('审核正文')
    expect(validateDraft(draft(['\u{1f600}'.repeat(20000)]))).toBe('')
    expect(validateDraft(draft(['x'.repeat(20001)]))).toContain('第 1 块不能超过 20000')
    expect(validateDraft(draft(['text\0']))).toContain('空字符')
  })
  it('enforces aggregate and section-count limits', () => {
    expect(validateDraft(draft(Array(50).fill('x'.repeat(20000))))).toBe('')
    expect(validateDraft(draft([...Array(50).fill('x'.repeat(20000)), 'x']))).toContain('1000000')
    expect(validateDraft(draft(Array(2000).fill('x')))).toBe('')
    expect(validateDraft(draft(Array(2001).fill('x')))).toContain('2000 块')
  })
})

describe('admin pagination', () => {
  it('pages accounts in batches of 100 while retaining global totals', async () => {
    const users = Array.from({ length: 101 }, (_, index) => ({ ...overview.users[0], id: `user-${index}`, username: `account-${index}` }))
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => {
      const url = new URL(path, 'http://localhost')
      const offset = Number(url.searchParams.get('offset'))
      return url.pathname === '/api/admin/overview'
        ? reply({ users: users.slice(offset, offset + 100), totals: { ...overview.totals, users: 101 }, offset })
        : defaultFetch(path, options)
    })
    render(<AdminDashboard onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole('tab', { name: '账号概览' }))
    await screen.findByText('account-0')
    expect(screen.getByRole('button', { name: '上一页账号' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '下一页账号' }))
    await screen.findByText('account-100')
    expect(screen.queryByText('account-0')).not.toBeInTheDocument()
    expect(screen.getByText('第 2 页 · 共 101 个账号')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '下一页账号' })).toBeDisabled()
    expect(fetchMock.mock.calls.some(([path]) => path === '/api/admin/overview?offset=100')).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: '上一页账号' }))
    await screen.findByText('account-0')
  })

  it('pages documents with a 50-row limit without discarding the selected draft', async () => {
    documents = [document, ...Array.from({ length: 50 }, (_, index) => ({ ...document, id: `doc-${index}`, title: `Document ${index}` }))]
    await openDocument()
    fireEvent.change(screen.getByLabelText('正文'), { target: { value: 'preserved while paging' } })
    fireEvent.click(screen.getByRole('button', { name: '下一页文档' }))
    await screen.findByRole('button', { name: /Document 49/ })
    expect(screen.getByLabelText('正文')).toHaveValue('preserved while paging')
    expect(screen.queryByRole('button', { name: /Operator notes/ })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '下一页文档' })).toBeDisabled()
    expect(fetchMock.mock.calls.some(([path]) => path.startsWith('/api/admin/knowledge?offset=50&limit=50'))).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: '上一页文档' }))
    await screen.findByRole('button', { name: /Operator notes/ })
    expect(screen.getByLabelText('正文')).toHaveValue('preserved while paging')
  })

  it('retains the current document page when the next-page request fails', async () => {
    documents = Array.from({ length: 51 }, (_, index) => ({ ...document, id: `doc-${index}`, title: `Document ${index}` }))
    await library()
    fetchMock.mockImplementation(async (path: string, options?: RequestInit) => path.includes('offset=50') ? reply({}, 503) : defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '下一页文档' }))
    await screen.findByText('请求失败 (503)')
    expect(screen.getByRole('button', { name: /Document 0 / })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '上一页文档' })).toBeDisabled()
  })
})

it('validates character limits as Unicode characters and accepts empty provenance', () => {
  expect(validateDraft({ title: 'title', source: '', content: '\u{1f600}'.repeat(60000) })).toBe('')
  expect(validateDraft({ title: 'title', source: 'x'.repeat(1001), content: '' })).toContain('1000')
  expect(validateDraft({ title: ' ', source: '', content: '' })).toContain('标题')
  expect(validateDraft({ title: 'title', source: '', content: ' ' })).toContain('正文')
  expect(validateDraft({ title: 'title', source: '', content: 'text\0' })).toContain('空字符')
})
