import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useCallback, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RagCatalog } from './RagCatalog'
import { RagDetail, RagItem, initialRagFilters, ragApi } from './rag-api'
import { setAccount } from './auth'
// Exported from CatalogStore against test_rag_catalog_readonly's ephemeral SQLite fixture.
// No running workspace, model service or real uploaded source is involved.
import sqlFixture from './__fixtures__/rag-catalog-sqlite.json'

const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
const pathname = (path: string) => new URL(path, 'http://localhost').pathname
const sourceOf = (path: string) => new URL(path, 'http://localhost').searchParams.get('source_id')
const dataSources = { default_source_id: 'current', sources: [
  { id: 'current', label: '当前资料库', read_only: false, status: 'configured', kind: 'current' },
  { id: 'legacy', label: '历史研发资料', read_only: true, status: 'configured', kind: 'memory_case' },
] }
const scopes = [{ id: 'shared', label: '共享资料库', kind: 'shared' }, { id: 'team', label: '研发空间', kind: 'workspace' }]
const metadata = { complete: true, source_health: [{ scope_id: 'shared', source_kind: 'knowledge_document', status: 'ready' }, { scope_id: 'team', source_kind: 'memory_case', status: 'ready' }], scopes, as_of: '2026-11-11T12:00:00Z', catalog_revision: 'catalog-r1' }
const memory: RagDetail = {
  item_id: 'memory-one', source_kind: 'memory_case', source_id: '1', scope: scopes[1], title: '历史故障经验', excerpt: '输入步长检查', source: '任务一',
  format: 'memory', latest_version: 'case-version', serving_version: 'case-version', content_state: 'available', lifecycle_state: 'active',
  index_state: 'ready', retrieval_state: 'eligible', retrieval_reason: '当前经验可被检索', cleanup_state: 'not_requested', updated_at: metadata.as_of,
  verified_at: metadata.as_of, revision: 'record-r1', capabilities: ['view', 'versions', 'chunks', 'archive', 'correct'], chunk_count: 2,
  selected_version: 'case-version', content: '原始运行证据', sections: [{ id: 'problem', label: '问题与证据', text: '源任务保留的真实记录 <script>not executable</script>' }],
  source_file: null, provenance: { source_kind: 'memory_case' }, corrections: [],
}
const document: RagDetail = {
  ...memory, item_id: 'document-one', source_kind: 'knowledge_document', source_id: 'doc-one', scope: scopes[0], title: '共享规范',
  source: '管理员上传', format: 'text', latest_version: '2', serving_version: '1', selected_version: '1', index_state: 'queued',
  content: '已生效原文', sections: [{ id: 'body', label: '正文', text: '已生效原文' }], capabilities: ['view', 'versions', 'chunks', 'edit'],
}
let items: RagItem[]
let fetchMock: ReturnType<typeof vi.fn>
let onForbidden: ReturnType<typeof vi.fn>
let onManage: ReturnType<typeof vi.fn>
const summaries = () => ({ ...metadata, total: items.length, eligible: items.length, pending: 1, attention: 0, excluded: 0 })

function defaultFetch(path: string, options?: RequestInit): Response {
  const url = new URL(path, 'http://localhost')
  if (url.pathname === '/api/admin/rag/sources') return reply(dataSources)
  if (url.pathname === '/api/admin/rag/catalog') return reply({ ...metadata, items, total: items.length, next_cursor: null })
  if (url.pathname === '/api/admin/rag/catalog/summary') return reply(summaries())
  const id = url.pathname.split('/')[5]
  const record = id === 'document-one' ? structuredClone(document) : structuredClone(memory)
  if (url.pathname.endsWith('/versions')) return reply({ items: record.source_kind === 'memory_case'
    ? [{ version: 'case-version', state: 'ready', chunk_count: 2, is_latest: true, is_serving: true, created_at: metadata.as_of }]
    : [{ version: '2', state: 'queued', chunk_count: 0, is_latest: true, is_serving: false, created_at: metadata.as_of }, { version: '1', state: 'ready', chunk_count: 2, is_latest: false, is_serving: true, created_at: metadata.as_of }] })
  if (url.pathname.endsWith('/chunks')) return reply({ items: [{ id: 'chunk-a', text: '该版本原文片段', position: 0, locator: '正文段落' }], next_cursor: null, version: url.searchParams.get('version') })
  if (url.pathname.endsWith('/actions')) {
    const body = JSON.parse(String(options?.body))
    return reply(body.action === 'archive' ? { ...record, lifecycle_state: 'archived', serving_version: null, retrieval_state: 'excluded' } : { ...record, corrections: [{ id: 'correction-one', content: body.content, created_at: metadata.as_of }] })
  }
  if (url.pathname.startsWith('/api/admin/rag/operations/')) return reply({ status: 'unknown' })
  if (url.searchParams.get('version') === '2') return reply({ ...record, selected_version: '2', content: '新草稿原文', sections: [{ id: 'body', label: '正文', text: '新草稿原文' }] })
  return reply(record)
}
beforeEach(() => {
  items = [structuredClone(memory), structuredClone(document)]
  setAccount({ id: 'admin-one', username: 'admin', csrf_token: 'csrf', is_admin: true })
  fetchMock = vi.fn(async (path: string, options?: RequestInit) => defaultFetch(path, options))
  vi.stubGlobal('fetch', fetchMock)
  onForbidden = vi.fn(); onManage = vi.fn()
})
afterEach(() => { cleanup(); setAccount(null); vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.useRealTimers() })
function open() { return render(<RagCatalog onForbidden={onForbidden} onManageDocument={onManage} />) }
const reader = () => within(screen.getByRole('article', { name: '原始资料详情' }))
const actions = () => fetchMock.mock.calls.filter(([path, options]) => pathname(path).endsWith('/actions') && options?.method === 'POST')

describe('unified original-source catalog', () => {
  it('filters both the document list and summary by origin on the server', async () => {
    items = [structuredClone(document)]
    fetchMock.mockImplementation(async (path, options) => pathname(path) === '/api/admin/rag/sources'
      ? reply({ ...dataSources, origin_filters: [{ id: 'history:daily-history', label: '日常研发' }] })
      : defaultFetch(path, options))
    open()
    await screen.findByText('已生效原文')
    fireEvent.change(screen.getByRole('combobox', { name: '资料分类' }), { target: { value: 'history:daily-history' } })
    for (const endpoint of ['/api/admin/rag/catalog', '/api/admin/rag/catalog/summary']) {
      await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => {
        const url = new URL(path, 'http://localhost')
        return url.pathname === endpoint && url.searchParams.get('origin') === 'history:daily-history'
      })).toBe(true))
    }
    expect(fetchMock.mock.calls.some(([path]) => pathname(path).endsWith('/vectors'))).toBe(false)
  })
  it('shows history and documents together, renders raw text safely, and defaults to original reading', async () => {
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.getByRole('button', { name: /共享规范/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /历史故障经验/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: '原文' })).toHaveAttribute('aria-selected', 'true')
    expect(documentBodyScripts()).toBe(0)
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
    expect(actions()).toHaveLength(0)
  })
  it('opens the actual serving version and lets users inspect the newer draft without changing it', async () => {
    open()
    fireEvent.click(await screen.findByRole('button', { name: /共享规范/ }))
    await screen.findByText('已生效原文')
    const versions = await screen.findByRole('combobox', { name: '查看原文版本' })
    await waitFor(() => expect(within(versions).getByRole('option', { name: '2 · 最新保存' })).toBeInTheDocument())
    expect(versions).toHaveValue('1')
    fireEvent.change(versions, { target: { value: '2' } })
    await screen.findByText('新草稿原文')
    expect(reader().getByText('RAG 当前可使用版本：').parentElement).toHaveTextContent('1')
    expect(actions()).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: '管理 / 编辑此文档' }))
    expect(onManage).toHaveBeenCalledWith('doc-one')
  })
  it('passes literal text search, source and authorized scope filters to the API', async () => {
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    await screen.findByRole('combobox', { name: '资料范围' })
    fireEvent.change(screen.getByRole('searchbox', { name: '搜索标题或正文' }), { target: { value: '步长_%' } })
    fireEvent.click(screen.getByRole('button', { name: '查找' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path.includes('q=%E6%AD%A5%E9%95%BF_%25'))).toBe(true))
    fireEvent.click(screen.getByRole('button', { name: '历史经验' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path.includes('source_kind=memory_case'))).toBe(true))
    fireEvent.change(screen.getByRole('combobox', { name: '资料范围' }), { target: { value: 'team' } })
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path.includes('scope_id=team'))).toBe(true))
  })
  it('presents current knowledge as shared without a personal scope or memory category', async () => {
    items = [structuredClone(document)]
    fetchMock.mockImplementation(async (path, options) => {
      if (pathname(path) === '/api/admin/rag/catalog') return reply({ ...metadata, scopes: [scopes[0]], items, total: 1, next_cursor: null })
      return defaultFetch(path, options)
    })
    open()
    await screen.findByText('已生效原文')
    expect(screen.queryByRole('combobox', { name: '资料范围' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '历史经验' })).not.toBeInTheDocument()
    expect(screen.getByText('全体用户共享的知识资料')).toBeInTheDocument()
    expect(screen.getByText(/发布后供全体用户检索/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '新增 / 上传文档' }))
    expect(onManage).toHaveBeenCalledWith(null)
    fireEvent.click(screen.getByRole('button', { name: '管理 / 编辑此文档' }))
    expect(onManage).toHaveBeenCalledWith('doc-one')
  })
  it('makes partial source failures visible instead of claiming an empty full catalog', async () => {
    fetchMock.mockImplementation(async path => pathname(path).endsWith('/sources') ? reply(dataSources) : path.includes('/summary') ? reply({ ...summaries(), complete: false })
      : reply({ ...metadata, complete: false, source_health: [{ scope_id: 'team', source_kind: 'memory_case', status: 'unavailable', reason: '来源暂不可读' }], items: [], total: 0, next_cursor: null }))
    open()
    await screen.findByText('当前为部分结果，尚未覆盖全部授权资料。')
    expect(screen.getByText(/其他来源尚未读取成功/)).toBeInTheDocument()
    expect(screen.queryByText('没有符合条件的资料。')).not.toBeInTheDocument()
  })
  it('clears administrator content when a later request receives 403', async () => {
    function Host() { const [denied, setDenied] = useState(false); const deny = useCallback(() => setDenied(true), []); return denied ? <p>管理员已无权访问</p> : <RagCatalog onForbidden={deny} onManageDocument={onManage} /> }
    render(<Host />)
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fetchMock.mockImplementation(async () => reply({}, 403))
    fireEvent.click(screen.getByRole('button', { name: '刷新资料总览' }))
    await screen.findByText('管理员已无权访问')
    expect(screen.queryByText('源任务保留的真实记录 <script>not executable</script>')).not.toBeInTheDocument()
  })
  it('resets stale pagination instead of silently showing results from changed catalog revisions', async () => {
    fetchMock.mockImplementation(async (path, options) => {
      const url = new URL(path, 'http://localhost')
      if (url.pathname === '/api/admin/rag/catalog') return url.searchParams.has('cursor') ? reply({}, 409) : reply({ ...metadata, items, total: 50, next_cursor: 'signed-next' })
      return defaultFetch(path, options)
    })
    open()
    await waitFor(() => expect(screen.getByRole('button', { name: '下一页资料' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '下一页资料' }))
    await screen.findByText('资料或访问范围已经变化，请重新读取后再操作。')
    expect(screen.queryByText('源任务保留的真实记录 <script>not executable</script>')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '重新读取' }))
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.getByText('第 1 页')).toBeInTheDocument()
  })
  it('fetches text chunks for the selected source version only', async () => {
    open()
    fireEvent.click(await screen.findByRole('button', { name: /共享规范/ }))
    await screen.findByText('已生效原文')
    fireEvent.click(screen.getByRole('tab', { name: '检索片段' }))
    await screen.findByText('该版本原文片段')
    expect(fetchMock.mock.calls.some(([path]) => path.includes('/document-one/chunks?version=1'))).toBe(true)
    expect(screen.queryByText('embedding')).not.toBeInTheDocument()
  })
  it('keeps same-ID historical chunks from different index generations distinct', async () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
    fetchMock.mockImplementation(async (path, options) => {
      if (path.includes('/chunks?')) return reply({ items: [
        { id: 'same-id', text: '第一代历史片段', position: 0, generation: 'g1', revision: 1 },
        { id: 'same-id', text: '第二代历史片段', position: 0, generation: 'g2', revision: 2 },
      ], next_cursor: null, version: 'case-version' })
      return defaultFetch(path, options)
    })
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fireEvent.click(screen.getByRole('tab', { name: '检索片段' }))
    await screen.findByText('第一代历史片段')
    expect(screen.getByText('第二代历史片段')).toBeInTheDocument()
    expect(screen.getByText('片段 1 · 索引代次 g1 · 索引修订 1')).toBeInTheDocument()
    expect(screen.getByText('片段 1 · 索引代次 g2 · 索引修订 2')).toBeInTheDocument()
    expect(errors.mock.calls.some(values => values.join(' ').includes('same key'))).toBe(false)
  })
  it('renders the actual SQL adapter DTO, including historical memory and old document text', async () => {
    fetchMock.mockImplementation(async path => {
      const url = new URL(path, 'http://localhost')
      if (url.pathname === '/api/admin/rag/sources') return reply(dataSources)
      if (url.pathname.endsWith('/summary')) return reply(sqlFixture.summary)
      if (url.pathname === '/api/admin/rag/catalog') return reply(sqlFixture.listing)
      const id = decodeURIComponent(url.pathname.split('/')[5])
      const record = sqlFixture.records[id as keyof typeof sqlFixture.records]
      if (url.pathname.endsWith('/versions')) return reply(record.versions)
      const historical = 'history' in record ? record.history : null
      return reply(url.searchParams.has('version') && historical ? historical : record.detail)
    })
    open()
    await screen.findByText('旧版正文 保留原句')
    expect(screen.queryByRole('button', { name: '下载该版本原文件' })).not.toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: /同名资料 · failure-diagnosis/ })[0])
    await screen.findByText('输入步长异常')
    const historical = Object.values(sqlFixture.records).find(row => 'history' in row)!
    fireEvent.change(screen.getByRole('combobox', { name: '查看原文版本' }), { target: { value: ('history' in historical ? historical.history.selected_version : '') } })
    await screen.findByText('旧记忆')
    expect(screen.queryByText('999999')).not.toBeInTheDocument()
    expect(screen.queryByText('NEVER-RETURN')).not.toBeInTheDocument()
  })
  it('can load older version pages without dropping the selected serving version', async () => {
    fetchMock.mockImplementation(async (path, options) => {
      if (path.includes('/versions?cursor=older-page')) return reply({ items: [{ version: 'old-version', is_latest: false, is_serving: false, state: 'recorded', chunk_count: 1, created_at: metadata.as_of }], next_cursor: null })
      if (pathname(path).endsWith('/versions')) return reply({ items: [], next_cursor: 'older-page' })
      return defaultFetch(path, options)
    })
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.getByRole('combobox', { name: '查看原文版本' })).toHaveValue('case-version')
    fireEvent.click(screen.getByRole('button', { name: '继续加载历史版本' }))
    await screen.findByRole('option', { name: 'old-version' })
  })
  it('hides unsupported write actions on a read-only memory record', async () => {
    fetchMock.mockImplementation(async (path, options) => pathname(path).endsWith('/memory-one') ? reply({ ...memory, capabilities: ['view', 'versions', 'chunks'] }) : defaultFetch(path, options))
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.queryByRole('button', { name: '停用这条经验' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '添加纠错说明' })).not.toBeInTheDocument()
  })
  it('shows orphaned index metadata separately without inventing recovered documents', async () => {
    fetchMock.mockImplementation(async (path, options) => path.includes('/reconciliation?') ? reply({
      index_anomalies: [{ source_kind: 'memory_case', source_id: 'missing-source', scope: scopes[1], reason: 'source_missing', index_entries: 1, index_chunks: 3 }],
      index_anomalies_total: 1, index_anomalies_next_cursor: null, index_scan_complete: false,
      index_source_health: [{ scope_id: 'shared', source_kind: 'knowledge_document', status: 'unavailable', reason: '索引来源暂不可读' }],
      vector_scan_performed: false, as_of: metadata.as_of,
    }) : defaultFetch(path, options))
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fireEvent.change(screen.getByRole('combobox', { name: '使用状态' }), { target: { value: 'attention' } })
    fireEvent.click(await screen.findByRole('button', { name: '检查原文关联' }))
    await screen.findByText('来源编号：missing-source')
    expect(screen.getByText('索引来源暂不可读')).toBeInTheDocument()
    expect(screen.getByText(/源记录缺失，不能从索引还原原文/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /missing-source/ })).not.toBeInTheDocument()
    expect(screen.queryByText('已检查的数据库记录中没有发现缺失原文的关联。')).not.toBeInTheDocument()
  })
})

describe('bounded administrative mutations', () => {
  it('sends a correction as a new note with revision and request ID, not an overwrite', async () => {
    open()
    fireEvent.click(await screen.findByRole('button', { name: '添加纠错说明' }))
    fireEvent.change(screen.getByRole('textbox', { name: '纠错内容' }), { target: { value: '此案例仅适用于给定环境。' } })
    fireEvent.click(screen.getByRole('button', { name: '保存说明' }))
    await waitFor(() => expect(actions()).toHaveLength(1))
    expect(JSON.parse(String(actions()[0][1].body))).toEqual({ action: 'correct', expected_revision: 'record-r1', request_id: expect.any(String), content: '此案例仅适用于给定环境。' })
  })
  it('requires explicit archive confirmation and does not issue a delete when canceled', async () => {
    open()
    fireEvent.click(await screen.findByRole('button', { name: '停用这条经验' }))
    expect(screen.getByText('停用后不再参与检索，原文和历史证据仍保留。')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(actions()).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: '停用这条经验' }))
    fireEvent.click(screen.getByRole('button', { name: '确认停用' }))
    await waitFor(() => expect(actions()).toHaveLength(1))
    expect(JSON.parse(String(actions()[0][1].body))).toMatchObject({ action: 'archive', expected_revision: 'record-r1' })
  })
  it('does not replay a mutation after network failure and an unknown receipt', async () => {
    fetchMock.mockImplementation(async (path, options) => { if (pathname(path).endsWith('/actions')) throw new TypeError('network down'); return defaultFetch(path, options) })
    open()
    fireEvent.click(await screen.findByRole('button', { name: '停用这条经验' }))
    fireEvent.click(screen.getByRole('button', { name: '确认停用' }))
    await screen.findByText('操作结果尚未确认，请先核对状态；不会自动重复提交。')
    expect(screen.getByRole('button', { name: '确认停用' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '核对操作结果' }))
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path.includes('/operations/'))).toHaveLength(2))
    expect(fetchMock.mock.calls.filter(([path]) => path.includes('/operations/')).every(([path]) => path.includes('item_id=memory-one'))).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: /共享规范/ }))
    expect(screen.getByText('操作尚未完成或结果待核对，请先核对操作结果再离开。')).toBeInTheDocument()
    expect(screen.queryByText('已生效原文')).not.toBeInTheDocument()
    expect(actions()).toHaveLength(1)
  })
  it('warns before discarding unsaved correction notes when selecting another source', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    open()
    fireEvent.click(await screen.findByRole('button', { name: '添加纠错说明' }))
    fireEvent.change(screen.getByRole('textbox', { name: '纠错内容' }), { target: { value: '未保存说明' } })
    fireEvent.click(screen.getByRole('button', { name: /共享规范/ }))
    expect(confirm).toHaveBeenCalledWith('放弃未保存的说明？')
    expect(screen.getByRole('textbox', { name: '纠错内容' })).toHaveValue('未保存说明')
    expect(actions()).toHaveLength(0)
  })
  it('discards a late detail response for a previously selected item', async () => {
    let finish!: (value: Response) => void
    fetchMock.mockImplementation((path, options) => pathname(path).endsWith('/memory-one') ? new Promise<Response>(resolve => { finish = resolve }) : Promise.resolve(defaultFetch(path, options)))
    open()
    await waitFor(() => expect(finish).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: /共享规范/ }))
    await screen.findByText('已生效原文')
    await act(async () => finish(reply(memory)))
    expect(screen.queryByText('源任务保留的真实记录 <script>not executable</script>')).not.toBeInTheDocument()
  })
  it('aborts a read at its deadline instead of leaving the interface pending indefinitely', async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation((_path, options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))))
    const request = ragApi.catalog(initialRagFilters)
    const assertion = expect(request).rejects.toThrow('读取资料超时')
    await vi.advanceTimersByTimeAsync(15000)
    await assertion
  })
})

describe('separate current and historical data sources', () => {
  it('does not assume a complete current-only catalog when the source registry fails', async () => {
    fetchMock.mockImplementation(async path => pathname(path).endsWith('/sources') ? reply({ detail: '资料库配置暂不可读' }, 503) : defaultFetch(path))
    open()
    await screen.findByText(/无法读取资料库清单：资料库配置暂不可读/)
    expect(screen.queryByText('没有符合条件的资料。')).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path]) => pathname(path).startsWith('/api/admin/rag/catalog'))).toBe(false)
    expect(screen.queryByRole('button', { name: '新增 / 上传文档' })).not.toBeInTheDocument()
    fetchMock.mockImplementation(async (path, options) => defaultFetch(path, options))
    fireEvent.click(screen.getByRole('button', { name: '重新读取资料库' }))
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.getByRole('combobox', { name: '资料库' })).toHaveValue('current')
  })

  it('resets query when switching libraries and independently enforces history read-only', async () => {
    fetchMock.mockImplementation(async (path, options) => {
      if (sourceOf(path) !== 'legacy') return defaultFetch(path, options)
      if (pathname(path) === '/api/admin/rag/catalog') return reply({ ...metadata, scopes: [scopes[0]], items, total: 2, next_cursor: null })
      if (pathname(path).endsWith('/memory-one')) return reply({ ...memory, sections: [{ id: 'body', label: '历史原文', text: '另一资料库的同编号经验' }] })
      return defaultFetch(path, options)
    })
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fireEvent.change(screen.getByRole('searchbox', { name: '搜索标题或正文' }), { target: { value: '当前资料关键字' } })
    fireEvent.click(screen.getByRole('button', { name: '查找' }))
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.queryByRole('combobox', { name: '资料范围' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    await screen.findByText('另一资料库的同编号经验')
    expect(screen.getByText(/历史资料仅供查阅，不会在这里修改；显示的是历史来源的使用状态，不代表已加入当前服务的检索/)).toBeInTheDocument()
    expect(screen.getByRole('searchbox', { name: '搜索标题或正文' })).toHaveValue('')
    expect(screen.queryByRole('combobox', { name: '资料范围' })).not.toBeInTheDocument()
    expect(screen.queryByRole('option', { name: '研发空间' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '新增 / 上传文档' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '添加纠错说明' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '停用这条经验' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: '检索片段' }))
    await screen.findByText('该版本原文片段')
    expect(fetchMock.mock.calls.filter(([path]) => sourceOf(path) === 'legacy').every(([path]) => !path.includes('scope_id=team') && !path.includes('q=%'))).toBe(true)
    expect(fetchMock.mock.calls.some(([path]) => sourceOf(path) === 'legacy' && pathname(path).endsWith('/versions'))).toBe(true)
    expect(fetchMock.mock.calls.some(([path]) => sourceOf(path) === 'legacy' && pathname(path).endsWith('/chunks'))).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: /共享规范/ }))
    await screen.findByText('已生效原文')
    expect(screen.queryByRole('button', { name: '管理 / 编辑此文档' })).not.toBeInTheDocument()
    expect(actions()).toHaveLength(0)
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'current' } })
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.queryByText('另一资料库的同编号经验')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '添加纠错说明' })).toBeInTheDocument()
  })

  it('does not let a late same-ID response replace text from the newly selected library', async () => {
    let finish!: (value: Response) => void
    fetchMock.mockImplementation((path, options) => {
      if (pathname(path).endsWith('/memory-one')) return sourceOf(path) === 'current'
        ? new Promise<Response>(resolve => { finish = resolve })
        : Promise.resolve(reply({ ...memory, sections: [{ id: 'body', label: '历史原文', text: '独立历史原文' }] }))
      return Promise.resolve(defaultFetch(path, options))
    })
    open()
    await waitFor(() => expect(finish).toBeTypeOf('function'))
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    await screen.findByText('独立历史原文')
    await act(async () => finish(reply(memory)))
    expect(screen.getByText('独立历史原文')).toBeInTheDocument()
    expect(screen.queryByText('源任务保留的真实记录 <script>not executable</script>')).not.toBeInTheDocument()
  })

  it('preserves the unsaved-note and unknown-operation guards when selecting another library', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    fetchMock.mockImplementation(async (path, options) => { if (pathname(path).endsWith('/actions')) throw new TypeError('network down'); return defaultFetch(path, options) })
    open()
    fireEvent.click(await screen.findByRole('button', { name: '添加纠错说明' }))
    fireEvent.change(screen.getByRole('textbox', { name: '纠错内容' }), { target: { value: '尚未保存的说明' } })
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    expect(confirm).toHaveBeenCalledWith('放弃未保存的说明？')
    expect(screen.getByRole('combobox', { name: '资料库' })).toHaveValue('current')
    expect(screen.getByRole('textbox', { name: '纠错内容' })).toHaveValue('尚未保存的说明')
    fireEvent.click(screen.getByRole('button', { name: '保存说明' }))
    await screen.findByText('操作结果尚未确认，请先核对状态；不会自动重复提交。')
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    expect(screen.getByRole('combobox', { name: '资料库' })).toHaveValue('current')
    expect(screen.getByText('操作尚未完成或结果待核对，请先核对操作结果再离开。')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path]) => sourceOf(path) === 'legacy')).toBe(false)
    expect(actions()).toHaveLength(1)
    expect(actions().every(([path]) => sourceOf(path) === 'current')).toBe(true)
    expect(fetchMock.mock.calls.filter(([path]) => pathname(path).includes('/operations/')).every(([path]) => sourceOf(path) === 'current')).toBe(true)
  })

  it('shows an unavailable history source as an error without stale data or an empty-success message', async () => {
    fetchMock.mockImplementation(async (path, options) => sourceOf(path) === 'legacy' ? reply({ detail: '历史资料库无法连接，请重试。' }, 503) : defaultFetch(path, options))
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    await screen.findByRole('alert')
    expect(screen.getByRole('alert')).toHaveTextContent('历史资料库无法连接，请重试。')
    expect(screen.queryByText('源任务保留的真实记录 <script>not executable</script>')).not.toBeInTheDocument()
    expect(screen.queryByText('没有符合条件的资料。')).not.toBeInTheDocument()
    expect(screen.queryByText('共 2 份资料')).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'current' } })
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
  })

  it('does not accept an incomplete source registry as a current-only success', async () => {
    fetchMock.mockImplementation(async path => pathname(path).endsWith('/sources') ? reply({ sources: [] }) : defaultFetch(path))
    open()
    await screen.findByText(/资料库清单不完整，请重新读取/)
    expect(fetchMock.mock.calls.some(([path]) => pathname(path).startsWith('/api/admin/rag/catalog'))).toBe(false)
  })

  it('keeps source-level denial separate from revoked administrator access', async () => {
    fetchMock.mockImplementation(async (path, options) => sourceOf(path) === 'legacy'
      ? reply({ detail: '历史资料源未获授权或授权已失效', error_code: 'rag_source_forbidden' }, 403) : defaultFetch(path, options))
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    await screen.findByRole('alert')
    expect(screen.getByRole('alert')).toHaveTextContent('历史资料源未获授权或授权已失效')
    expect(onForbidden).not.toHaveBeenCalled()
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'current' } })
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
  })
})

describe('explicit historical-to-shared draft copies', () => {
  it('offers copying only with server capability and creates an editable draft after confirmation', async () => {
    fetchMock.mockImplementation(async (path, options) => {
      if (sourceOf(path) === 'legacy' && pathname(path).endsWith('/memory-one')) return reply({ ...memory, capabilities: ['view', 'versions', 'chunks', 'share_to_shared'] })
      if (pathname(path).endsWith('/share')) return reply({ id: 'shared-copy', title: memory.title, state: 'draft', published_version: null })
      return defaultFetch(path, options)
    })
    open()
    await screen.findByText('源任务保留的真实记录 <script>not executable</script>')
    expect(screen.queryByRole('button', { name: '加入共享知识库' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    fireEvent.click(await screen.findByRole('button', { name: '加入共享知识库' }))
    expect(screen.getByRole('alertdialog', { name: '加入共享知识库确认' })).toHaveTextContent('现在不会建立索引或向所有用户发布')
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(fetchMock.mock.calls.some(([path]) => pathname(path).endsWith('/share'))).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: '加入共享知识库' }))
    fireEvent.click(screen.getByRole('button', { name: '确认创建共享草稿' }))
    await waitFor(() => expect(onManage).toHaveBeenCalledWith('shared-copy'))
    const writes = fetchMock.mock.calls.filter(([path]) => pathname(path).endsWith('/share'))
    expect(writes).toHaveLength(1)
    expect(sourceOf(writes[0][0])).toBe('legacy')
    expect(JSON.parse(String(writes[0][1].body))).toEqual({ expected_revision: memory.revision, request_id: expect.any(String) })
    expect(actions()).toHaveLength(0)
    expect(fetchMock.mock.calls.some(([path]) => path.includes('/knowledge/confirm') || path.includes('/publish'))).toBe(false)
  })

  it('opens an existing shared version without reimporting or changing the historical record', async () => {
    fetchMock.mockImplementation(async (path, options) => pathname(path).endsWith('/memory-one')
      ? reply({ ...memory, capabilities: ['view', 'versions', 'chunks'], shared_document_id: 'existing-shared' }) : defaultFetch(path, options))
    open()
    fireEvent.click(await screen.findByRole('button', { name: '编辑共享版本' }))
    expect(onManage).toHaveBeenCalledWith('existing-shared')
    expect(screen.queryByRole('button', { name: '加入共享知识库' })).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([_path, options]) => options?.method === 'POST')).toBe(false)
  })

  it('holds an uncertain copy in place and reuses the same request only when asked', async () => {
    let interrupted = true
    fetchMock.mockImplementation(async (path, options) => {
      if (pathname(path).endsWith('/memory-one')) return reply({ ...memory, capabilities: ['view', 'share_to_shared'] })
      if (pathname(path).endsWith('/share')) {
        if (interrupted) throw new TypeError('transport lost')
        return reply({ id: 'shared-after-retry', state: 'draft', published_version: null })
      }
      return defaultFetch(path, options)
    })
    open()
    fireEvent.click(await screen.findByRole('button', { name: '加入共享知识库' }))
    fireEvent.click(screen.getByRole('button', { name: '确认创建共享草稿' }))
    await screen.findByText('尚未核对到共享草稿，结果仍未知；可继续核对或使用原请求重试。')
    fireEvent.change(screen.getByRole('combobox', { name: '资料库' }), { target: { value: 'legacy' } })
    expect(screen.getByRole('combobox', { name: '资料库' })).toHaveValue('current')
    const writes = () => fetchMock.mock.calls.filter(([path]) => pathname(path).endsWith('/share'))
    expect(writes()).toHaveLength(1)
    interrupted = false
    fireEvent.click(screen.getByRole('button', { name: '使用原请求重试导入' }))
    await waitFor(() => expect(onManage).toHaveBeenCalledWith('shared-after-retry'))
    expect(writes()).toHaveLength(2)
    expect(writes()[1][1].body).toEqual(writes()[0][1].body)
  })
})

function documentBodyScripts() { return window.document.querySelectorAll('.rag-reader script').length }
