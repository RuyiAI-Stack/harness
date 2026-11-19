import { BookOpenText, ChevronLeft, ChevronRight, FilePlus2, RefreshCw, Search, Download } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { AdminRequestError, adminDate } from './admin-api'
import { KnowledgeImage } from './KnowledgeImage'
import { RagVectors } from './RagVectors'
import { initialRagFilters, RagAction, RagCatalogPage, RagChunk, RagDetail, RagDataSource, RagFilters, RagItem, RagReconciliation, RagScope, RagSummary, RagVersion, ragApi, ragError } from './rag-api'
import './rag.css'

type NavigationGuard = () => boolean
type Props = { onForbidden: () => void; onManageDocument: (id: string | null) => void; active?: boolean; onGuardChange?: (guard: NavigationGuard | null) => void }
const kindLabels: Record<string, string> = { knowledge_document: '知识文档', memory_case: '历史经验' }
const lifecycleLabels: Record<string, string> = { active: '有效资料', archived: '已停用', trashed: '回收站' }
const indexLabels: Record<string, string> = { not_indexed: '未建立索引', queued: '等待处理', building: '处理中', ready: '索引就绪', failed: '处理失败', stale: '索引需更新', unknown: '索引状态待核对' }
const contentLabels: Record<string, string> = { available: '原文可读', missing: '原文缺失', parse_failed: '解析失败', review_required: '原文待审核' }
const cleanupLabels: Record<string, string> = { not_requested: '未请求清理', queued: '等待清理', running: '清理中', failed: '清理失败', complete: '清理完成', unknown: '尚未核对物理清理结果' }
const provenanceLabels: Record<string, string> = { source_run: '来源任务', confidence_grade: '原记录可信等级', evidence_status: '证据说明' }

function Status({ item }: { item: RagItem }) {
  const label = item.lifecycle_state !== 'active' ? lifecycleLabels[item.lifecycle_state] || item.lifecycle_state
    : item.content_state !== 'available' ? contentLabels[item.content_state] || item.content_state
      : item.retrieval_state === 'eligible' ? '可用于 RAG'
        : item.retrieval_state === 'unknown' ? '使用状态待核对'
          : item.retrieval_state === 'unavailable' ? '暂不可用' : '未启用'
  return <span className={`rag-badge ${item.retrieval_state === 'eligible' ? 'ready' : item.content_state === 'missing' || item.index_state === 'failed' ? 'attention' : ''}`}>{label}</span>
}

export function RagCatalog({ onForbidden, onManageDocument, active = true, onGuardChange }: Props) {
  const [sources, setSources] = useState<RagDataSource[]>([])
  const [origins, setOrigins] = useState<{ id: string; label: string }[]>([])
  const [sourceId, setSourceId] = useState('current')
  const [sourcesReady, setSourcesReady] = useState(false)
  const [sourcesLoading, setSourcesLoading] = useState(true)
  const [sourcesError, setSourcesError] = useState('')
  const [sourcesReload, setSourcesReload] = useState(0)
  const [filters, setFilters] = useState<RagFilters>(initialRagFilters)
  const [query, setQuery] = useState('')
  const [pages, setPages] = useState<(string | null)[]>([null])
  const [page, setPage] = useState(0)
  const [reload, setReload] = useState(0)
  const [catalog, setCatalog] = useState<RagCatalogPage | null>(null)
  const [summary, setSummary] = useState<RagSummary | null>(null)
  const [scopes, setScopes] = useState<RagScope[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [summaryError, setSummaryError] = useState('')
  const guard = useRef<NavigationGuard | null>(null)
  const registerGuard = useCallback((value: NavigationGuard | null) => { guard.current = value }, [])
  const mayLeave = useCallback(() => guard.current?.() ?? true, [])
  useEffect(() => { onGuardChange?.(mayLeave); return () => onGuardChange?.(null) }, [mayLeave, onGuardChange])
  const filterKey = JSON.stringify(filters)
  const cursor = pages[page]
  const dataSource = sources.find(source => source.id === sourceId)
  const readOnly = dataSource?.read_only !== false
  const refresh = useCallback(() => { setPages([null]); setPage(0); setReload(value => value + 1) }, [])

  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    setSourcesLoading(true); setSourcesReady(false); setSourcesError(''); setError(''); setSummaryError(''); setLoading(false); setCatalog(null); setSummary(null); setSelected(null); setScopes([])
    void ragApi.sources(controller.signal).then(result => {
      if (controller.signal.aborted) return
      if (!Array.isArray(result.sources) || !result.sources.length ||
          result.sources.some(source => !source.id || !source.label || typeof source.read_only !== 'boolean') ||
          new Set(result.sources.map(source => source.id)).size !== result.sources.length ||
          !result.sources.some(source => source.id === result.default_source_id)) throw new Error('资料库清单不完整，请重新读取。')
      setSources(result.sources)
      setOrigins(result.origin_filters || [])
      setSourceId(previous => result.sources.some(source => source.id === previous) ? previous : result.default_source_id)
      setFilters(initialRagFilters); setQuery(''); setPages([null]); setPage(0); setSourcesReady(true)
    }).catch(reason => { if (!controller.signal.aborted) { setSources([]); setSourcesError(ragError(reason, onForbidden)) } })
      .finally(() => { if (!controller.signal.aborted) setSourcesLoading(false) })
    return () => controller.abort()
  }, [active, sourcesReload, onForbidden])

  useEffect(() => {
    if (!active || !sourcesReady || !dataSource) return
    const controller = new AbortController()
    setLoading(true); setError(''); setSummaryError(''); setCatalog(null); setSummary(null)
    void Promise.allSettled([
      ragApi.catalog(filters, cursor, controller.signal, sourceId), ragApi.summary(filters, controller.signal, sourceId),
    ]).then(([listing, stats]) => {
      if (controller.signal.aborted) return
      if (listing.status === 'rejected') {
        setError(ragError(listing.reason, onForbidden)); setSelected(null)
      } else {
        setCatalog(listing.value); setScopes(listing.value.scopes)
        setSelected(previous => listing.value.items.some(item => item.item_id === previous) ? previous : listing.value.items[0]?.item_id || null)
      }
      if (stats.status === 'rejected') setSummaryError(ragError(stats.reason, onForbidden))
      else if (listing.status === 'fulfilled' && listing.value.catalog_revision === stats.value.catalog_revision) setSummary(stats.value)
      else if (listing.status === 'fulfilled') setSummaryError('统计期间资料已变化，请刷新查看一致的统计。')
    }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [active, sourcesReady, dataSource, sourceId, filterKey, cursor, reload, onForbidden])

  function changeSource(next: string) {
    if (next !== sourceId && sources.some(source => source.id === next) && mayLeave()) {
      setSourceId(next); setFilters(initialRagFilters); setQuery(''); setPages([null]); setPage(0); setSelected(null); setCatalog(null); setSummary(null); setScopes([]); setError(''); setSummaryError('')
    }
  }
  function changeFilters(next: RagFilters) { if (mayLeave()) { setFilters(next); setPages([null]); setPage(0); setSelected(null) } }
  const partial = catalog?.complete === false || summary?.complete === false
  const health = catalog?.source_health.filter(source => source.status !== 'ready') || []
  return <div className="rag-catalog" aria-busy={loading || sourcesLoading}>
    <div className="rag-heading">
      <div><h2><BookOpenText size={20} />RAG 资料总览</h2><p className="admin-muted">{dataSource?.read_only ? '历史资料，仅供查阅与审核导入' : '全体用户共享的知识资料'}</p></div>
      <div className="rag-heading-actions">{sourcesReady && !readOnly && <button className="secondary-button" onClick={() => { if (mayLeave()) onManageDocument(null) }}><FilePlus2 size={16} />新增 / 上传文档</button>}
        <button className="icon-button" title="刷新资料总览" aria-label="刷新资料总览" disabled={loading || sourcesLoading} onClick={() => { if (mayLeave()) setSourcesReload(value => value + 1) }}><RefreshCw size={16} /></button></div>
    </div>
    <div className="rag-data-source"><label>资料库<select value={sourcesReady ? sourceId : ''} disabled={sourcesLoading || !sourcesReady} onChange={event => changeSource(event.target.value)}>{!sourcesReady && <option value="">{sourcesLoading ? '正在读取资料库清单…' : '资料库清单未读取'}</option>}{sources.map(source => <option key={source.id} value={source.id}>{source.read_only ? `历史原件 · ${source.label.split(' · ')[0]}` : `${source.label}（统一管理）`}</option>)}</select></label>
      {sourcesReady && dataSource && <p className="admin-muted">当前查看：<strong>{dataSource.label}</strong>。{readOnly ? ' 历史资料仅供查阅，不会在这里修改；显示的是历史来源的使用状态，不代表已加入当前服务的检索。' : ' 发布后供全体用户检索；修改需更新 RAG 后生效，停用或删除后不再用于新的检索。'}</p>}</div>
    {sourcesError && <p className="admin-error" role="alert">无法读取资料库清单：{sourcesError} <button className="secondary-button" onClick={() => setSourcesReload(value => value + 1)}>重新读取资料库</button></p>}
    {sourcesLoading && <p className="admin-muted" role="status">正在读取资料库清单…</p>}
    {sourcesReady && <div className="rag-filters">
      <form onSubmit={event => { event.preventDefault(); changeFilters({ ...filters, q: query.trim() }) }}>
        <label>搜索标题或正文<input maxLength={200} type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="查找原文中的内容" /></label>
        <button className="secondary-button" type="submit"><Search size={16} />查找</button>
      </form>
      {readOnly && scopes.length > 1 && <label>资料范围<select value={filters.scope_id} onChange={event => changeFilters({ ...filters, scope_id: event.target.value })}><option value="all">全部授权范围</option>{scopes.map(scope => <option key={scope.id} value={scope.id}>{scope.label}</option>)}</select></label>}
      {!readOnly && origins.length > 0 && <label>资料分类<select value={filters.origin || ''} onChange={event => changeFilters({ ...filters, origin: event.target.value })}><option value="">全部来源</option>{origins.map(origin => <option key={origin.id} value={origin.id}>{origin.label}</option>)}</select></label>}
      <label>使用状态<select value={filters.status} onChange={event => changeFilters({ ...filters, status: event.target.value })}><option value="all">全部状态</option><option value="eligible">可用于 RAG</option><option value="pending">待处理</option><option value="attention">需处理异常</option><option value="excluded">未启用</option></select></label>
    </div>
    }
    {error && sourcesReady && <p className="admin-error" role="alert">{error} <button className="secondary-button" onClick={refresh}>重新读取</button></p>}
    {partial && <div className="rag-alert" role="status"><strong>当前为部分结果，尚未覆盖全部授权资料。</strong>{health.map((source, index) => <p key={`${source.scope_id}-${source.source_kind}-${index}`}>{scopes.find(scope => scope.id === source.scope_id)?.label || '已授权来源'} · {kindLabels[source.source_kind] || source.source_kind}：{source.reason || '暂时无法读取'}</p>)}</div>}
    {summary && <><dl className="rag-totals">{[['资料总数', summary.total], ['可用于 RAG', summary.eligible], ['待处理', summary.pending], ['需处理异常', summary.attention]].map(([label, value]) => <div key={label}><dt>{label}{partial ? '（部分）' : ''}</dt><dd>{value}</dd></div>)}</dl>
      <p className="rag-count-note admin-muted">可用资料可能同时有待处理的新草稿，状态数量不相加。统计时间：{adminDate(summary.as_of)}</p></>}
    {summaryError && <p className="admin-warning" role="status">资料统计暂不可用：{summaryError}</p>}
    <div className="rag-layout">
      <nav className="rag-sources" aria-label="资料分类">{(readOnly
        ? [['all', '全部资料'], ['memory_case', '历史经验']]
        : [['all', '全部资料'], ['knowledge_document', '知识文档'], ['trash', '回收站']]
      ).map(([kind, label]) => <button key={kind} aria-pressed={kind === 'trash' ? filters.view === 'trash' : filters.view !== 'trash' && filters.source_kind === kind} onClick={() => changeFilters({ ...filters, source_kind: kind === 'trash' ? 'all' : kind, view: kind === 'trash' ? 'trash' : 'active' })}>{label}</button>)}</nav>
      <section className="rag-list" aria-label="RAG 资料列表" tabIndex={0}>
        {loading && <p className="admin-muted" role="status">正在读取资料…</p>}
        {catalog && <p className="admin-muted rag-list-count">{partial ? '已读取' : '共'} {catalog.total} 份资料</p>}
        {catalog?.items.map(item => <button key={item.item_id} className="rag-document" aria-pressed={selected === item.item_id} onClick={() => { if (selected !== item.item_id && mayLeave()) setSelected(item.item_id) }}>
          <strong>{item.title}</strong><span className="rag-excerpt">{item.excerpt || (item.content_state === 'missing' ? '原始内容缺失，需核对来源。' : '暂无摘要')}</span><Status item={item} />
          <small>{kindLabels[item.source_kind]} · {item.scope.label}</small>
        </button>)}
        {!loading && catalog?.items.length === 0 && <p className="admin-muted">{partial ? '当前已读取来源中暂无匹配资料，其他来源尚未读取成功。' : '没有符合条件的资料。'}</p>}
        <nav className="admin-pagination" aria-label="资料目录分页"><span className="admin-muted">第 {page + 1} 页</span>
          <button className="icon-button" title="上一页资料" aria-label="上一页资料" disabled={loading || page === 0} onClick={() => { if (mayLeave()) setPage(value => value - 1) }}><ChevronLeft size={16} /></button>
          <button className="icon-button" title="下一页资料" aria-label="下一页资料" disabled={loading || !catalog?.next_cursor} onClick={() => { if (mayLeave()) { setPages(values => [...values.slice(0, page + 1), catalog!.next_cursor]); setPage(value => value + 1) } }}><ChevronRight size={16} /></button>
        </nav>
      </section>
      <div className="rag-main" role="region" aria-label="资料内容">{!loading && sourcesReady && catalog && selected ? <RagReader key={`${sourceId}:${selected}`} sourceId={sourceId} readOnly={readOnly} itemId={selected} reload={reload} onForbidden={onForbidden} onManageDocument={onManageDocument} onChanged={refresh} registerGuard={registerGuard} /> : <p className="rag-placeholder admin-muted">{loading ? '读取完成后可查看原文。' : '选择一份资料，查看原文和来源。'}</p>}</div>
    </div>
    {sourcesReady && filters.status === 'attention' && <IndexSourceCheck key={`${sourceId}:${filters.scope_id}`} sourceId={sourceId} scopeId={filters.scope_id} onForbidden={onForbidden} />}
  </div>
}

function IndexSourceCheck({ sourceId, scopeId, onForbidden }: { sourceId: string; scopeId: string; onForbidden: () => void }) {
  const [enabled, setEnabled] = useState(false)
  const [cursor, setCursor] = useState<string | null>(null)
  const [data, setData] = useState<RagReconciliation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    if (!enabled) return
    const controller = new AbortController(); setLoading(true); setError('')
    void ragApi.reconcile(scopeId, cursor, controller.signal, sourceId).then(result => {
      if (!controller.signal.aborted) setData(previous => ({ ...result, index_anomalies: cursor ? [...(previous?.index_anomalies || []), ...result.index_anomalies] : result.index_anomalies }))
    }).catch(reason => { if (!controller.signal.aborted) setError(ragError(reason, onForbidden)) }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [enabled, sourceId, scopeId, cursor, retry, onForbidden])
  return <section className="rag-index-check" aria-label="缺失原文检查"><h3>缺失原文的索引记录</h3><p className="admin-muted">检查当前资料范围内的数据库索引记录，不受标题搜索或资料分类筛选影响；没有扫描向量服务。索引记录不计作一份原文资料。</p>
    {!enabled && <button className="secondary-button" onClick={() => setEnabled(true)}>检查原文关联</button>}
    {error && <p role="alert" className="admin-error">{error} <button className="secondary-button" onClick={() => { setCursor(null); setData(null); setRetry(value => value + 1) }}>重新核对原文关联</button></p>}
    {loading && <p role="status">正在核对原文关联…</p>}
    {data && <><p>{data.index_scan_complete ? '已核对的异常' : '部分来源核对失败，目前发现的异常'}：{data.index_anomalies_total} 项 · {adminDate(data.as_of)}</p>
      {data.index_source_health.filter(row => row.status !== 'ready').map((row, index) => <p className="admin-warning" key={index}>{row.reason || '部分来源暂不可读'}</p>)}
      {data.index_anomalies.map(row => <article className="rag-original-section" key={`${row.scope.id}-${row.source_kind}-${row.source_id}`}><strong>{kindLabels[row.source_kind] || row.source_kind} · {row.scope.label}</strong><p>来源编号：{row.source_id}</p><p>源记录缺失，不能从索引还原原文。关联索引 {row.index_entries} 条、片段 {row.index_chunks} 个。</p></article>)}
      {!data.index_anomalies.length && <p>{data.index_scan_complete ? '已检查的数据库记录中没有发现缺失原文的关联。' : '已读取的部分中尚未发现异常，其余来源仍需核对。'}</p>}
      {data.index_anomalies_next_cursor && <button className="secondary-button" disabled={loading} onClick={() => setCursor(data.index_anomalies_next_cursor)}>继续读取异常</button>}</>}
  </section>
}

function RagReader({ sourceId, readOnly, itemId, reload, onForbidden, onManageDocument, onChanged, registerGuard }: Props & { sourceId: string; readOnly: boolean; itemId: string; reload: number; onChanged: () => void; registerGuard: (guard: NavigationGuard | null) => void }) {
  const [detail, setDetail] = useState<RagDetail | null>(null)
  const [versions, setVersions] = useState<RagVersion[]>([])
  const [versionCursor, setVersionCursor] = useState<string | null>(null)
  const [nextVersion, setNextVersion] = useState<string | null>(null)
  const [versionLoading, setVersionLoading] = useState(false)
  const [version, setVersion] = useState<string | null>(null)
  const [tab, setTab] = useState<'content' | 'chunks' | 'versions' | 'vectors'>('content')
  const [chunks, setChunks] = useState<RagChunk[]>([])
  const [chunkCursor, setChunkCursor] = useState<string | null>(null)
  const [nextChunk, setNextChunk] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [chunkLoading, setChunkLoading] = useState(false)
  const [error, setError] = useState('')
  const [versionError, setVersionError] = useState('')
  const [chunkError, setChunkError] = useState('')
  const [localRevision, setLocalRevision] = useState(0)
  const [action, setAction] = useState<'archive' | 'correct' | null>(null)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [uncertain, setUncertain] = useState<RagAction | null>(null)
  const [shareDialog, setShareDialog] = useState(false)
  const [shareUncertain, setShareUncertain] = useState(false)
  const shareRequest = useRef<{ expected_revision: string; request_id: string } | null>(null)
  const [notice, setNotice] = useState('')
  const operation = useRef(false)
  const editState = useRef({ text, busy, uncertain, shareUncertain })
  editState.current = { text, busy, uncertain, shareUncertain }
  const mayLeave = useCallback(() => {
    if (operation.current || editState.current.busy || editState.current.uncertain || editState.current.shareUncertain) { setNotice('操作尚未完成或结果待核对，请先核对操作结果再离开。'); return false }
    return !editState.current.text || window.confirm('放弃未保存的说明？')
  }, [])
  useEffect(() => {
    registerGuard(mayLeave)
    const warn = (event: BeforeUnloadEvent) => { if (editState.current.text || editState.current.busy || editState.current.uncertain || editState.current.shareUncertain) { event.preventDefault(); event.returnValue = '' } }
    window.addEventListener('beforeunload', warn)
    return () => { registerGuard(null); window.removeEventListener('beforeunload', warn) }
  }, [registerGuard, mayLeave])
  const lifetime = useRef<AbortController | null>(null)
  useEffect(() => { const controller = new AbortController(); lifetime.current = controller; return () => controller.abort() }, [])
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError(''); setDetail(null); setChunks([]); setChunkCursor(null); setNextChunk(null)
    void ragApi.detail(itemId, version, controller.signal, sourceId).then(result => { if (!controller.signal.aborted) setDetail(result) }).catch(reason => {
      if (!controller.signal.aborted) setError(ragError(reason, onForbidden))
    }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [sourceId, itemId, version, reload, localRevision, onForbidden])
  useEffect(() => {
    const controller = new AbortController(); if (!versionCursor) setVersions([]); setVersionError(''); setVersionLoading(true)
    void ragApi.versions(itemId, controller.signal, versionCursor, sourceId).then(result => { if (!controller.signal.aborted) { setVersions(previous => versionCursor ? [...previous, ...result.items] : result.items); setNextVersion(result.next_cursor || null) } }).catch(reason => {
      if (!controller.signal.aborted) setVersionError(ragError(reason, onForbidden))
    }).finally(() => { if (!controller.signal.aborted) setVersionLoading(false) })
    return () => controller.abort()
  }, [sourceId, itemId, reload, localRevision, versionCursor, onForbidden])
  useEffect(() => {
    if (tab !== 'chunks' || !detail) return
    const controller = new AbortController(); setChunkLoading(true); setChunkError('')
    void ragApi.chunks(itemId, detail.selected_version, chunkCursor, controller.signal, sourceId).then(result => {
      if (controller.signal.aborted) return
      if (result.version !== detail.selected_version) throw new Error('片段版本与原文不一致，请重新读取。')
      setChunks(previous => chunkCursor ? [...previous, ...result.items] : result.items); setNextChunk(result.next_cursor)
    }).catch(reason => { if (!controller.signal.aborted) setChunkError(ragError(reason, onForbidden)) }).finally(() => { if (!controller.signal.aborted) setChunkLoading(false) })
    return () => controller.abort()
  }, [sourceId, tab, detail, itemId, chunkCursor, onForbidden])
  function changed() { setAction(null); setText(''); setUncertain(null); onChanged() }
  async function reconcile(request: RagAction) {
    try {
      const result = await ragApi.operation(request.request_id, itemId, lifetime.current?.signal, sourceId)
      if (lifetime.current?.signal.aborted) return
      if (result.status === 'committed' && result.item_id === itemId && result.action === request.action) { changed(); return }
      setUncertain(request); setNotice('操作结果尚未确认，请先核对状态；不会自动重复提交。')
    } catch (reason) {
      if (lifetime.current?.signal.aborted) return
      setUncertain(request); setNotice('暂时无法核对操作结果；请保持当前页面，稍后重新核对。'); setError(ragError(reason, onForbidden))
    }
  }
  async function mutate() {
    if (readOnly || !detail || !action || operation.current || uncertain || (action === 'correct' && !text.trim())) return
    operation.current = true; setBusy(true); setError('')
    const request: RagAction = { action, expected_revision: detail.revision, request_id: crypto.randomUUID(), ...(action === 'correct' ? { content: text.trim() } : { reason: text.trim() || '管理员在资料总览中停用' }) }
    try { await ragApi.action(itemId, request, lifetime.current?.signal, sourceId); if (!lifetime.current?.signal.aborted) changed() }
    catch (reason) {
      if (lifetime.current?.signal.aborted) return
      if (reason instanceof AdminRequestError && reason.status < 500) setError(ragError(reason, onForbidden))
      else await reconcile(request)
    } finally { operation.current = false; if (!lifetime.current?.signal.aborted) setBusy(false) }
  }
  async function download() {
    if (!detail || operation.current) return
    operation.current = true; setBusy(true); setError('')
    try {
      const blob = await ragApi.original(itemId, detail.selected_version, lifetime.current?.signal, sourceId)
      if (lifetime.current?.signal.aborted) return
      const url = URL.createObjectURL(blob); const anchor = document.createElement('a')
      anchor.href = url; anchor.download = detail.source_file?.filename || 'source-document'; anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (reason) { if (!lifetime.current?.signal.aborted) setError(ragError(reason, onForbidden)) }
    finally { operation.current = false; if (!lifetime.current?.signal.aborted) setBusy(false) }
  }
  async function checkShare() {
    try {
      const latest = await ragApi.detail(itemId, null, lifetime.current?.signal, sourceId)
      if (lifetime.current?.signal.aborted) return
      if (latest.shared_document_id) {
        setDetail(latest); setShareUncertain(false); setShareDialog(false); shareRequest.current = null
        setNotice('已核对到共享草稿，可编辑共享版本；尚未自动发布。')
      } else setNotice('尚未核对到共享草稿，结果仍未知；可继续核对或使用原请求重试。')
    } catch (reason) { if (!lifetime.current?.signal.aborted) setError(ragError(reason, onForbidden)) }
  }
  async function share() {
    if (!detail || operation.current || !detail.capabilities.includes('share_to_shared')) return
    if (detail.shared_document_id) { onManageDocument(detail.shared_document_id); return }
    if (!shareRequest.current) shareRequest.current = { expected_revision: detail.revision, request_id: crypto.randomUUID() }
    operation.current = true; setBusy(true); setError(''); setNotice('')
    try {
      const document = await ragApi.share(itemId, shareRequest.current, lifetime.current?.signal, sourceId)
      if (lifetime.current?.signal.aborted) return
      if (!document.id) throw new Error('共享草稿回执不完整，请先核对结果。')
      setShareUncertain(false); setShareDialog(false); shareRequest.current = null
      setDetail(previous => previous ? { ...previous, shared_document_id: document.id } : previous)
      onManageDocument(document.id)
    } catch (reason) {
      if (lifetime.current?.signal.aborted) return
      if (reason instanceof AdminRequestError && reason.status < 500 && reason.status !== 408) { setError(ragError(reason, onForbidden)); shareRequest.current = null; setShareUncertain(false) }
      else { setShareUncertain(true); setNotice('导入结果尚未确认，不会自动重复创建；正在只读核对。'); await checkShare() }
    } finally { operation.current = false; if (!lifetime.current?.signal.aborted) setBusy(false) }
  }
  const currentChoice = detail ? { version: detail.selected_version, is_serving: detail.selected_version === detail.serving_version, is_latest: detail.selected_version === detail.latest_version } : null
  const choices = currentChoice && !versions.some(row => row.version === currentChoice.version) ? [currentChoice, ...versions] : versions
  return <article className="rag-reader" aria-label="原始资料详情" aria-busy={loading || busy}>
    {error && <p className="admin-error" role="alert">{error}</p>}
    {notice && <p className="admin-warning" role="status">{notice}</p>}
    {uncertain && <button className="secondary-button" disabled={busy} onClick={async () => { setBusy(true); await reconcile(uncertain); if (!lifetime.current?.signal.aborted) setBusy(false) }}>核对操作结果</button>}
    {loading && <p className="admin-muted" role="status">正在读取原文…</p>}
    {!loading && !detail && <button className="secondary-button" onClick={() => setLocalRevision(value => value + 1)}>重新读取原文</button>}
    {detail && <>
      <p className="admin-muted">{kindLabels[detail.source_kind]} · {detail.scope.label}</p><h3>{detail.selected_title ?? detail.title}</h3><Status item={detail} />
      <div className="rag-reader-actions">
        {!readOnly && detail.source_kind === 'knowledge_document' && detail.capabilities.some(capability => ['edit', 'delete', 'restore', 'publish', 'index', 'unpublish'].includes(capability)) && <button className="primary-button" disabled={busy} onClick={() => onManageDocument(detail.source_id)}>管理 / 编辑此文档</button>}
        {detail.shared_document_id && <button className="primary-button" disabled={busy || shareUncertain} onClick={() => { if (mayLeave()) onManageDocument(detail.shared_document_id!) }}>编辑共享版本</button>}
      </div>
      <dl className="rag-metadata"><div><dt>来源</dt><dd>{(detail.selected_source ?? detail.source) || '未注明来源'}</dd></div><div><dt>更新时间</dt><dd>{adminDate(detail.updated_at)}</dd></div><div><dt>最新版本索引记录</dt><dd>{indexLabels[detail.index_state] || detail.index_state}</dd></div></dl>
      <details className="rag-provenance"><summary>来源详情</summary><dl className="rag-metadata"><div><dt>资料编号</dt><dd>{detail.source_id}</dd></div><div><dt>最后核对时间</dt><dd>{detail.verified_at ? adminDate(detail.verified_at) : '尚未单独核对'}</dd></div>{Object.entries(provenanceLabels).map(([key, label]) => typeof detail.provenance[key] === 'string' && detail.provenance[key] ? <div key={key}><dt>{label}</dt><dd>{String(detail.provenance[key])}</dd></div> : null)}</dl></details>
      <div className="rag-use-state"><p>{detail.serving_version ? <>RAG 当前可使用版本：<strong>{detail.serving_version}</strong></> : '当前没有已确认可使用的版本。'}</p><p>{detail.retrieval_reason}</p>{detail.serving_version && detail.latest_version !== detail.serving_version && <p>最新保存版本与当前使用版本不同，修改不会直接替换原有版本。</p>}</div>
      {detail.lifecycle_state !== 'active' && <p className="admin-muted">索引清理：{cleanupLabels[detail.cleanup_state] || detail.cleanup_state}。停用不等于底层向量已物理删除。</p>}
      {detail.content_state !== 'available' && <p className="admin-warning">{contentLabels[detail.content_state] || detail.content_state}。{detail.content_state === 'missing' && '不能从向量还原完整原文。'}</p>}
      {detail.source_file && <div className="rag-source-file"><span>原文件：{detail.source_file.filename}</span>{detail.source_file.download_available && <button className="secondary-button" disabled={busy} onClick={() => void download()}><Download size={14} />下载该版本原文件</button>}{!detail.source_file.download_available && <p className="admin-warning">原文件暂不可下载；下方显示已保存的解析原文。</p>}{detail.source_file.warnings?.map((warning, index) => <p className="admin-warning" key={index}>{warning}</p>)}</div>}
      <label>查看原文版本<select value={detail.selected_version} disabled={loading || busy || !!action || shareDialog || shareUncertain} onChange={event => { setVersion(event.target.value); setTab('content') }}>{choices.map(choice => <option key={choice.version} value={choice.version}>{choice.version}{choice.is_serving ? ' · RAG 当前使用' : ''}{choice.is_latest ? ' · 最新保存' : ''}</option>)}</select></label>
      {detail.source_file?.preview_available && <KnowledgeImage
        src={`/api/admin/rag/catalog/${encodeURIComponent(itemId)}/preview?${new URLSearchParams({ version: detail.selected_version, source_id: sourceId })}`}
        filename={detail.source_file.filename} onForbidden={onForbidden} />}
      {versionError && <p className="admin-warning" role="status">版本列表暂不可用：{versionError}</p>}
      {nextVersion && <button className="secondary-button" disabled={versionLoading} onClick={() => setVersionCursor(nextVersion)}>继续加载历史版本</button>}
      <div className="rag-tabs" role="tablist" aria-label="原文与追溯">{([['content', '原文'], ['chunks', '检索片段'], ['vectors', '实际向量'], ['versions', '版本记录']] as const).map(([key, label]) => <button key={key} role="tab" id={`rag-${key}-tab`} aria-controls={`rag-${key}-panel`} aria-selected={tab === key} onClick={() => { if (key === 'chunks' && tab !== 'chunks') { setChunkCursor(null); setChunks([]); setNextChunk(null) }; setTab(key) }}>{label}</button>)}</div>
      <div role="tabpanel" id={`rag-${tab}-panel`} aria-labelledby={`rag-${tab}-tab`}>
        {tab === 'vectors' && <RagVectors key={`${sourceId}:${itemId}:${detail.selected_version}`} itemId={itemId} version={detail.selected_version} sourceId={sourceId} onForbidden={onForbidden} />}
        {tab === 'content' && <>{detail.sections.length ? detail.sections.map(section => <section className="rag-original-section" key={section.id}><h4>{section.label || section.locator || '原文段落'}</h4>{section.locator && section.locator !== section.label && <small>{section.locator}</small>}<div className="rag-original-text">{section.text}</div></section>) : detail.content ? <div className="rag-original-text">{detail.content}</div> : <p className="admin-muted">此版本没有可展示的原文。</p>}
          {!!detail.corrections?.length && <section className="rag-corrections"><h4>管理员纠错说明</h4><p className="admin-muted">独立说明，不覆盖原始任务证据，也不代表已经重新验证。</p>{detail.corrections.map(correction => <article key={correction.id}><small>{adminDate(correction.created_at)}</small><div className="rag-original-text">{correction.content}</div></article>)}</section>}</>}
        {tab === 'chunks' && <>{chunkError && <p className="admin-error" role="alert">{chunkError}</p>}{chunks.map(chunk => <section className="rag-original-section" key={JSON.stringify([detail.source_kind, detail.selected_version, chunk.generation ?? null, chunk.revision ?? null, chunk.kind ?? null, chunk.id, chunk.position])}><small>{chunk.locator || `片段 ${chunk.position + 1}`}{chunk.generation != null && ` · 索引代次 ${chunk.generation}`}{chunk.revision != null && ` · 索引修订 ${chunk.revision}`}</small><div className="rag-original-text">{chunk.text}</div></section>)}{chunkLoading && <p role="status">正在读取片段…</p>}{!chunkLoading && !chunkError && !chunks.length && <p className="admin-muted">此版本暂无可展示的文字片段。</p>}{nextChunk && <button className="secondary-button" disabled={chunkLoading} onClick={() => setChunkCursor(nextChunk)}>继续加载片段</button>}</>}
        {tab === 'versions' && <ul className="rag-versions">{versions.map(row => <li key={row.version}><div><strong>{row.version}</strong><p className="admin-muted">{row.is_serving ? 'RAG 当前使用 · ' : ''}{row.is_latest ? '最新保存 · ' : ''}{row.chunk_count} 个片段 · {adminDate(row.created_at)}</p></div><button className="secondary-button" disabled={busy || !!uncertain || !!action || shareDialog || shareUncertain} onClick={() => { setVersion(row.version); setTab('content') }}>查看该版本原文</button></li>)}</ul>}
      </div>
      <div className="rag-reader-actions">
        <button className="secondary-button" disabled={busy || !!uncertain} onClick={() => { if (mayLeave()) { setAction(null); setText(''); setVersionCursor(null); setLocalRevision(value => value + 1) } }}><RefreshCw size={14} />只读刷新</button>
        {!readOnly && detail.source_kind === 'memory_case' && detail.capabilities.includes('correct') && <button className="secondary-button" disabled={busy || !!uncertain} onClick={() => { if (mayLeave()) { setAction('correct'); setText('') } }}>添加纠错说明</button>}
        {!readOnly && detail.source_kind === 'memory_case' && detail.capabilities.includes('archive') && <button className="secondary-button" disabled={busy || !!uncertain} onClick={() => { if (mayLeave()) { setAction('archive'); setText('') } }}>停用这条经验</button>}
        {!detail.shared_document_id && detail.capabilities.includes('share_to_shared') && <button className="secondary-button" disabled={busy || shareUncertain} onClick={() => { if (mayLeave()) setShareDialog(true) }}>加入共享知识库</button>}
      </div>
      {shareDialog && <div className="rag-action-confirm" role="alertdialog" aria-label="加入共享知识库确认"><h4>复制到共享知识库</h4><p>将此资料的最新版本复制为共享草稿，并保留来源。历史原件不变；现在不会建立索引或向所有用户发布。请在编辑页检查内容，再点击“确认修改并更新 RAG”，成功后所有用户才可检索。</p>
        {shareUncertain ? <><button className="secondary-button" disabled={busy} onClick={async () => { if (operation.current) return; operation.current = true; setBusy(true); try { await checkShare() } finally { operation.current = false; if (!lifetime.current?.signal.aborted) setBusy(false) } }}>核对共享草稿</button><button className="secondary-button" disabled={busy} onClick={() => void share()}>使用原请求重试导入</button></> : <><button className="primary-button" disabled={busy} onClick={() => void share()}>确认创建共享草稿</button><button className="secondary-button" disabled={busy} onClick={() => setShareDialog(false)}>取消</button></>}
      </div>}
      {action && <form className="rag-action-confirm" onSubmit={event => { event.preventDefault(); void mutate() }} aria-label={action === 'correct' ? '纠错说明表单' : '停用经验确认'}><h4>{action === 'correct' ? '添加纠错说明' : '确认停用这条经验'}</h4><p>{action === 'correct' ? '原始任务事实保留。此说明不会自动标记为已验证的运行结果。' : '停用后不再参与检索，原文和历史证据仍保留。'}</p><label>{action === 'correct' ? '纠错内容' : '停用说明（可选）'}<textarea value={text} onChange={event => setText(event.target.value)} maxLength={action === 'correct' ? 10000 : 1000} required={action === 'correct'} disabled={busy || !!uncertain} /></label><div><button type="submit" className="primary-button" disabled={busy || !!uncertain || (action === 'correct' && !text.trim())}>{busy ? '处理中…' : action === 'correct' ? '保存说明' : '确认停用'}</button><button type="button" className="secondary-button" disabled={busy || !!uncertain} onClick={() => { if (mayLeave()) { setAction(null); setText('') } }}>取消</button></div></form>}
    </>}
  </article>
}
