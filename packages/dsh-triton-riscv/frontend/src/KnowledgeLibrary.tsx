import { BookOpen, ChevronLeft, ChevronRight, Database, Download, Eye, EyeOff, FilePlus2, RefreshCw, RotateCcw, Save, Search, Trash2, Upload, X } from 'lucide-react'
import { FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { KnowledgeImage } from './KnowledgeImage'
import {
  AdminRequestError, KnowledgeDetail, KnowledgeDocument, KnowledgeDraft, KnowledgeSearch, KnowledgeSync, adminApi, adminDate, adminError,
  characterCount, contentByteLimit, sectionCharacterLimit, validateDraft, validateSourceFile, validateUploadMetadata,
} from './admin-api'

type PermissionProps = { onForbidden: () => void }
const pendingStates = new Set(['parsing', 'pending', 'queued', 'indexing', 'running'])
const importStates = new Set(['parsing', 'review', 'parse_failed', 'ocr_required', 'ocr_empty'])
const stateLabels: Record<string, string> = {
  draft: '草稿', pending: '等待索引', queued: '等待索引', indexing: '索引中', running: '索引中', ready: '就绪', failed: '失败',
  parsing: '解析中', review: '待审核', parse_failed: '解析失败', ocr_required: '需要 OCR', ocr_empty: '未识别到文字', deleted: '已删除', stale: '需保存新版本后建索引',
}
const emptyDraft = { title: '', source: '', content: '' }
const syncing = (sync?: KnowledgeSync | null) => !!sync && ['pending', 'indexing'].includes(sync.state)
const isImported = (detail: KnowledgeDetail) => !!detail.source_file || detail.sections !== undefined || importStates.has(detail.state)

function detailDraft(detail: KnowledgeDetail): KnowledgeDraft {
  const metadata = { title: detail.title, source: detail.source }
  return isImported(detail)
    ? { ...metadata, sections: (detail.sections || []).map(({ id, text }) => ({ id, text })) }
    : { ...metadata, content: detail.content }
}

function useRequestScope() {
  const scope = useRef(new AbortController())
  useEffect(() => {
    const controller = new AbortController()
    scope.current = controller
    return () => controller.abort()
  }, [])
  return scope
}

function StateBadge({ state }: { state: string }) {
  return <span className={`knowledge-state ${state === 'ready' ? 'ready' : ['failed', 'parse_failed', 'ocr_required', 'ocr_empty'].includes(state) ? 'failed' : pendingStates.has(state) ? 'pending' : ''}`}>{stateLabels[state] || state}</span>
}

export function KnowledgeLibrary({ onForbidden, requestedDocumentId, navigationKey, onGuardChange }: PermissionProps & { requestedDocumentId?: string | null; navigationKey?: number; onGuardChange?: (guard: (() => boolean) | null) => void }) {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [enabled, setEnabled] = useState<boolean | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [offset, setOffset] = useState(0)
  const [pageLength, setPageLength] = useState(0)
  const [total, setTotal] = useState<number | undefined>()
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState({ q: '', status: 'all', view: 'active' })
  const [searchEpoch, setSearchEpoch] = useState(0)
  const scope = useRequestScope()
  const refreshId = useRef(0)
  const updatesDuringRefresh = useRef(new Map<string, KnowledgeDocument>())
  const dirty = useRef(false)
  const editorGuard = useRef<(() => boolean) | null>(null)
  const registerGuard = useCallback((guard: (() => boolean) | null) => { editorGuard.current = guard }, [])
  const mayLeave = useCallback(() => (editorGuard.current?.() ?? true) && (!dirty.current || window.confirm('放弃未保存的修改？')), [])
  useEffect(() => { onGuardChange?.(mayLeave); return () => onGuardChange?.(null) }, [mayLeave, onGuardChange])

  async function refresh(nextOffset = offset, nextFilters = filters) {
    const request = ++refreshId.current
    updatesDuringRefresh.current.clear()
    const signal = scope.current.signal
    setLoading(true)
    setError('')
    try {
      const result = await adminApi.documents(nextOffset, signal, nextFilters)
      if (signal.aborted || request !== refreshId.current) return
      setDocuments(result.documents.map(row => updatesDuringRefresh.current.get(row.id) || row))
      setEnabled(result.enabled)
      setOffset(nextOffset)
      setPageLength(result.documents.length)
      setTotal(result.total)
    } catch (reason) {
      if (!signal.aborted && request === refreshId.current) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted && request === refreshId.current) setLoading(false)
    }
  }
  useEffect(() => { void refresh() }, [])
  useEffect(() => {
    if (!navigationKey) return
    select(requestedDocumentId ?? null)
    void refresh(0)
  }, [navigationKey])

  function select(id: string | null) {
    if (id === selected) return
    if (!mayLeave()) return
    dirty.current = false
    setSelected(id)
  }

  function update(document: KnowledgeDocument) {
    updatesDuringRefresh.current.set(document.id, document)
    setDocuments(rows => rows.map(row => row.id === document.id ? document : row))
    if (!selected) {
      setSelected(document.id)
      void refresh(0)
    }
  }

  function filter(next: typeof filters) {
    if (!mayLeave()) return
    dirty.current = false
    setSelected(null)
    setFilters(next)
    void refresh(0, next)
  }

  function lifecycleChanged() {
    dirty.current = false
    setSelected(null)
    setSearchEpoch(value => value + 1)
    void refresh(0)
  }

  function committed() {
    setSearchEpoch(value => value + 1)
    void refresh()
  }

  return <div className="knowledge-layout">
    <aside className="knowledge-list" aria-label="知识文档" tabIndex={0}>
      <div className="knowledge-toolbar">
        <h2>共享知识库</h2>
        <span className="admin-muted">{total ?? documents.length}</span>
        <button className="icon-button" title="刷新文档列表" aria-label="刷新文档列表" disabled={loading} onClick={() => void refresh()}><RefreshCw size={16} /></button>
        <button className="icon-button" title="新建文档" aria-label="新建文档" disabled={!enabled} onClick={() => filters.view === 'trash' ? filter({ q: '', status: 'all', view: 'active' }) : select(null)}><FilePlus2 size={16} /></button>
      </div>
      <div className="knowledge-filters">
        <div className="workflow-modes" role="tablist" aria-label="资料范围">
          <button role="tab" aria-selected={filters.view === 'active'} onClick={() => filter({ ...filters, view: 'active' })}>资料库</button>
          <button role="tab" aria-selected={filters.view === 'trash'} onClick={() => filter({ ...filters, view: 'trash', status: 'all' })}>回收站</button>
        </div>
        <form onSubmit={event => { event.preventDefault(); filter({ ...filters, q: query.trim() }) }}>
          <input aria-label="查找资料" placeholder="标题或来源" maxLength={200} value={query} onChange={event => setQuery(event.target.value)} />
          <button className="icon-button" type="submit" title="查找资料" aria-label="查找资料" disabled={loading}><Search size={16} /></button>
        </form>
        <select aria-label="发布状态" value={filters.status} disabled={filters.view === 'trash'} onChange={event => filter({ ...filters, status: event.target.value })}>
          <option value="all">全部状态</option><option value="published">已发布</option><option value="unpublished">未发布</option>
        </select>
      </div>
      {error && <p role="alert" className="admin-error">{error}</p>}
      {loading && <p role="status" className="admin-muted">加载中...</p>}
      {enabled === false && <p role="status" className="admin-warning">知识检索未启用</p>}
      {!loading && enabled !== null && !documents.length && <p className="admin-muted">暂无文档。</p>}
      {documents.map(document => <button key={document.id} className="knowledge-document" aria-pressed={selected === document.id} onClick={() => select(document.id)}>
        <strong>{document.title}</strong>
        <StateBadge state={document.state} />
        <small>{document.filename || document.source || '未注明来源'}{document.format && ` · ${document.format.toUpperCase()}`}</small>
        <span>最新 v{document.latest_version} · 已发布 {document.published_version === null ? '无' : `v${document.published_version}`}</span>
        <small>{document.chunk_count} 个分块 · {adminDate(document.updated_at)}</small>
      </button>)}
      <nav className="admin-pagination" aria-label="文档分页">
        <span className="admin-muted">第 {Math.floor(offset / 50) + 1} 页</span>
        <button className="icon-button" title="上一页文档" aria-label="上一页文档" disabled={loading || offset === 0} onClick={() => void refresh(Math.max(0, offset - 50))}><ChevronLeft size={16} /></button>
        <button className="icon-button" title="下一页文档" aria-label="下一页文档" disabled={loading || !enabled || (total === undefined ? pageLength < 50 : offset + 50 >= total) || offset >= 100000} onClick={() => void refresh(offset + 50)}><ChevronRight size={16} /></button>
      </nav>
    </aside>
    <div className="knowledge-main" role="region" aria-label="文档内容" tabIndex={0}>
      <div className="knowledge-content">
        {enabled !== null && <DocumentEditor key={selected || 'new'} documentId={selected} enabled={enabled} onUpdate={update} onCommit={committed} onLifecycle={lifecycleChanged} onDirty={value => { dirty.current = value }} registerGuard={registerGuard} onForbidden={onForbidden} />}
        <SearchPreview revision={searchEpoch} enabled={enabled === true} onForbidden={onForbidden} />
      </div>
    </div>
  </div>
}

function DocumentEditor({ documentId, enabled, onUpdate, onCommit, onLifecycle, onDirty, registerGuard, onForbidden }: PermissionProps & {
  documentId: string | null
  enabled: boolean
  onUpdate: (document: KnowledgeDocument) => void
  onDirty: (dirty: boolean) => void
  onLifecycle: () => void
  onCommit: () => void
  registerGuard: (guard: (() => boolean) | null) => void
}) {
  const [detail, setDetail] = useState<KnowledgeDetail | null>(null)
  const [draft, setDraft] = useState<KnowledgeDraft>(emptyDraft)
  const [base, setBase] = useState<KnowledgeDraft>(emptyDraft)
  const [baseVersion, setBaseVersion] = useState<number | undefined>()
  const [loading, setLoading] = useState(Boolean(documentId))
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sync, setSync] = useState<KnowledgeSync | null>(null)
  const [unknownConfirmation, setUnknownConfirmation] = useState(false)
  const [syncPaused, setSyncPaused] = useState('')
  const [syncEpoch, setSyncEpoch] = useState(0)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [preview, setPreview] = useState<'content' | 'chunks' | 'versions'>('content')
  const [sectionIndex, setSectionIndex] = useState(0)
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const scope = useRequestScope()
  const operation = useRef(false)
  const readId = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const sourceFileInput = useRef<HTMLInputElement>(null)
  const uploadRequest = useRef<{ file: File; metadata: string; id: string } | null>(null)
  const saveRequest = useRef<{ payload: string; id: string } | null>(null)
  const indexRequest = useRef<{ version: number; id: string } | null>(null)
  const confirmRequest = useRef<{ id: string; payload: KnowledgeDraft } | null>(null)
  const confirmationState = useRef({ busy, unknownConfirmation })
  confirmationState.current = { busy, unknownConfirmation }
  const parsingVersion = useRef<number | null>(null)
  const draftDirty = JSON.stringify(draft) !== JSON.stringify(base)
  const dirty = draftDirty || !!sourceFile
  const currentEditor = useRef({ dirty: draftDirty, version: baseVersion })
  currentEditor.current = { dirty: draftDirty, version: baseVersion }
  const stale = !!detail && detail.latest_version !== baseVersion
  const validation = validateDraft(draft) || (draft.sections && detail?.sections
    && JSON.stringify(draft.sections.map(section => section.id)) !== JSON.stringify(detail.sections.map(section => section.id))
    ? '审核块已更新，请重新读取。' : '')
  const uploadValidation = sourceFile ? validateUploadMetadata(sourceFile, draft) : ''
  const imported = !!detail && isImported(detail)
  const deleted = detail?.state === 'deleted'
  const parseLocked = !!detail && ['parsing', 'parse_failed', 'ocr_required', 'ocr_empty'].includes(detail.state)
  const canIndex = !!detail && ['draft', 'failed'].includes(detail.state) && (!imported || detail.source_file?.reviewed === true)
  const canPublish = detail?.state === 'ready' && (!imported || detail.source_file?.reviewed === true)
  const activeSection = draft.sections?.[sectionIndex]
  const originalSection = detail?.sections?.find(section => section.id === activeSection?.id)
  const sectionEdited = !!originalSection && (originalSection.edited || originalSection.text !== activeSection?.text)
  const syncActive = syncing(sync)
  useEffect(() => {
    registerGuard(() => {
      if (['confirm', 'receipt'].includes(confirmationState.current.busy) || confirmationState.current.unknownConfirmation) {
        setNotice('操作尚未完成或结果待核对，请先核对更新结果再离开。')
        return false
      }
      return true
    })
    return () => registerGuard(null)
  }, [registerGuard])
  useEffect(() => { onDirty(dirty) }, [dirty, onDirty])
  useEffect(() => {
    if (!dirty && !busy && !unknownConfirmation) return
    const preventClose = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', preventClose)
    return () => window.removeEventListener('beforeunload', preventClose)
  }, [dirty, busy, unknownConfirmation])

  async function load(replaceDraft: boolean) {
    if (!documentId) return
    const signal = scope.current.signal
    const request = ++readId.current
    if (replaceDraft) setLoading(true)
    try {
      const result = await adminApi.document(documentId, signal)
      if (signal.aborted || request !== readId.current) return
      setDetail(result)
      if (!unknownConfirmation) setSync(result.sync || null)
      onUpdate(result)
      // Parsing fills an initially empty editor, but background reads never replace local edits.
      const parsed = parsingVersion.current === result.latest_version && result.state !== 'parsing'
        && currentEditor.current.version === result.latest_version && !currentEditor.current.dirty
      if (replaceDraft || parsed) {
        const fields = detailDraft(result)
        setDraft(fields)
        setBase(fields)
        setBaseVersion(result.latest_version)
        setSectionIndex(0)
        parsingVersion.current = result.state === 'parsing' ? result.latest_version : null
      }
      setError('')
    } catch (reason) {
      if (!signal.aborted && request === readId.current) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted && request === readId.current && replaceDraft) setLoading(false)
    }
  }
  useEffect(() => { void load(true) }, [documentId])
  useEffect(() => {
    if (!detail || !pendingStates.has(detail.state) || busy || loading || syncActive || unknownConfirmation) return
    let stopped = false
    let timer: number
    // Schedule after each response so slow reads cannot overtake manual refreshes.
    const poll = async () => {
      await load(false)
      if (!stopped) timer = window.setTimeout(() => void poll(), 3000)
    }
    timer = window.setTimeout(() => void poll(), 3000)
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [detail?.state, busy, loading, documentId, syncActive, unknownConfirmation])

  function acceptConfirmation(summary: KnowledgeDocument, payload?: KnowledgeDraft) {
    const receipt = summary.sync
    if (!receipt?.request_id || !receipt.state) throw new Error('更新回执不完整，请核对结果。')
    setSync(receipt); setUnknownConfirmation(false); setSyncPaused(''); setError('')
    if (payload) {
      const fields: KnowledgeDraft = payload.sections
        ? { title: payload.title, source: payload.source, sections: payload.sections }
        : { title: payload.title, source: payload.source, content: payload.content }
      setBase(fields); setBaseVersion(receipt.version)
      onDirty(false)
    }
    setDetail(previous => ({ ...(previous || { content: payload?.content || '', chunks: [], versions: [] }), ...summary }))
    confirmRequest.current = null
    onUpdate(summary)
    if (receipt.state === 'published') { setNotice(`v${receipt.version} 已更新到共享 RAG，所有用户后续检索可使用。`); onCommit() }
    else if (receipt.state === 'superseded') setNotice('本次确认已被更新版本替代，未自动发布旧版本。')
    else if (receipt.state === 'withdrawn') setNotice('本次更新已撤销，未继续发布。')
    if (documentId && !syncing(receipt)) void load(false)
  }

  async function checkConfirmation() {
    const requestId = confirmRequest.current?.id || sync?.request_id
    if (!requestId || operation.current) return
    operation.current = true; setBusy('receipt'); setError(''); setSyncPaused('')
    const signal = scope.current.signal
    try {
      const summary = await adminApi.confirmation(requestId, signal)
      if (signal.aborted) return
      if (summary.sync?.request_id !== requestId) throw new Error('更新回执不匹配，请稍后核对。')
      acceptConfirmation(summary, confirmRequest.current?.payload)
      setSyncEpoch(value => value + 1)
    } catch (reason) {
      if (!signal.aborted) {
        setSyncPaused(reason instanceof AdminRequestError && reason.status === 404
          ? '尚未查到确认请求；结果仍未知，可继续核对，或使用同一请求重试。' : adminError(reason, onForbidden))
      }
    } finally { operation.current = false; if (!signal.aborted) setBusy('') }
  }

  useEffect(() => {
    if (!syncActive || !sync || busy || unknownConfirmation || syncPaused) return
    const requestId = sync.request_id
    const controller = new AbortController()
    const started = Date.now()
    let timer: number
    let count = 0
    const poll = async () => {
      if (controller.signal.aborted) return
      if (++count > 30 || Date.now() - started > 120000) { setSyncPaused('自动核对已暂停，后台仍可能处理中。请稍后核对更新结果。'); return }
      try {
        const summary = await adminApi.confirmation(requestId, controller.signal)
        if (controller.signal.aborted) return
        if (summary.sync?.request_id !== requestId) throw new Error('更新回执不匹配，请稍后核对。')
        acceptConfirmation(summary)
        if (syncing(summary.sync)) timer = window.setTimeout(() => void poll(), 3000)
      } catch (reason) { if (!controller.signal.aborted) setSyncPaused(adminError(reason, onForbidden)) }
    }
    timer = window.setTimeout(() => void poll(), 3000)
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [sync?.request_id, sync?.state, syncActive, busy, unknownConfirmation, syncPaused, syncEpoch])

  async function confirmUpdate(retry = false) {
    if (!enabled || deleted || operation.current || loading || sourceFile || parseLocked || syncActive || (documentId && !detail)) return
    if (!retry && (unknownConfirmation || stale || validation)) { if (validation) setError(validation); return }
    const request = retry ? confirmRequest.current : {
      id: crypto.randomUUID(), payload: { ...draft, ...(documentId ? { document_id: documentId, expected_version: baseVersion, catalog_revision: detail?.catalog_revision ?? 0 } : {}) },
    }
    if (!request) return
    confirmRequest.current = request
    operation.current = true; readId.current += 1; setBusy('confirm'); setError(''); setNotice(''); setSyncPaused('')
    const signal = scope.current.signal
    let reconcile = false
    try {
      const summary = await adminApi.confirm(request.payload, request.id, signal)
      if (signal.aborted) return
      if (summary.sync?.request_id !== request.id) throw new Error('更新回执不匹配，请核对结果。')
      acceptConfirmation(summary, request.payload)
    } catch (reason) {
      if (signal.aborted) return
      if (reason instanceof AdminRequestError && reason.status < 500 && reason.status !== 408) {
        setError(adminError(reason, onForbidden)); confirmRequest.current = null; setUnknownConfirmation(false)
      } else {
        setUnknownConfirmation(true); setNotice('确认结果尚未明确，不会自动重复提交；正在核对原请求。'); reconcile = true
      }
    } finally {
      operation.current = false
      if (!signal.aborted) setBusy('')
    }
    if (reconcile && !signal.aborted) await checkConfirmation()
  }

  async function mutate(action: 'save' | 'index' | 'publish' | 'unpublish') {
    if (!enabled || deleted || operation.current || loading || syncActive || unknownConfirmation || (documentId && !detail)) return
    if (action === 'save' && (sourceFile || parseLocked || stale)) return
    if (action === 'save' && validation) { setError(validation); return }
    if (action !== 'save' && !detail) return
    if ((action === 'index' || action === 'publish') && (dirty || stale || !enabled)) return
    if (action === 'index' && !canIndex) return
    if (action === 'publish' && !canPublish) return
    operation.current = true
    readId.current += 1
    const signal = scope.current.signal
    setBusy(action)
    setError('')
    setNotice('')
    try {
      let summary: KnowledgeDocument
      if (action === 'save') {
        const payload: KnowledgeDraft = { ...draft, ...(documentId ? { document_id: documentId, expected_version: baseVersion, catalog_revision: detail?.catalog_revision ?? 0 } : {}) }
        const fingerprint = JSON.stringify(payload)
        if (saveRequest.current?.payload !== fingerprint) saveRequest.current = { payload: fingerprint, id: crypto.randomUUID() }
        summary = await adminApi.save(payload, saveRequest.current.id, signal)
        if (signal.aborted) return
        setBase(draft)
        setBaseVersion(summary.latest_version)
        setDetail({ ...detail, ...summary, content: draft.content ?? '',
          ...(draft.sections ? {
            source_file: detail?.source_file ? { ...detail.source_file, reviewed: true } : undefined,
            sections: detail?.sections?.map(section => {
              const text = draft.sections!.find(item => item.id === section.id)?.text ?? section.text
              return { ...section, text, edited: section.edited || text !== section.text }
            }),
          } : {}),
          chunks: [], versions: [
          { version: summary.latest_version, state: summary.state, chunk_count: summary.chunk_count, error: summary.error },
          ...(detail?.versions.filter(item => item.version !== summary.latest_version) || []),
        ] })
        saveRequest.current = null
        setNotice(`已保存 v${summary.latest_version}`)
      } else {
        const current = detail!
        if (action === 'index') {
          if (indexRequest.current?.version !== current.latest_version) indexRequest.current = { version: current.latest_version, id: crypto.randomUUID() }
          summary = await adminApi.index(current.id, current.latest_version, indexRequest.current.id, signal, current.catalog_revision ?? 0)
          if (signal.aborted) return
          indexRequest.current = null
        } else if (action === 'publish') {
          summary = await adminApi.publish(current.id, current.latest_version, signal, current.catalog_revision ?? 0)
        } else {
          summary = await adminApi.unpublish(current.id, signal, current.catalog_revision ?? 0)
        }
        if (signal.aborted) return
        setDetail(previous => previous ? { ...previous, ...summary } : null)
        setNotice(action === 'index'
          ? pendingStates.has(summary.state) ? '索引已排队' : `索引状态：${stateLabels[summary.state] || summary.state}`
          : action === 'publish' ? `已发布 v${summary.published_version}` : '已取消发布')
      }
      onUpdate(summary)
      if (documentId) onCommit()
      if (action !== 'save' || documentId) await load(false)
    } catch (reason) {
      if (!signal.aborted) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted) { operation.current = false; setBusy('') }
    }
  }

  async function upload(file?: File) {
    if (!file || operation.current || !enabled || loading || imported || sourceFile) return
    setError('')
    setNotice('')
    if (!/\.(txt|md)$/i.test(file.name)) { setError('仅支持 .txt 和 .md 文件。'); return }
    if (file.size > contentByteLimit) { setError('文件不能超过 256 KB。'); return }
    if (draftDirty && draft.content && !window.confirm('替换未保存的正文？')) return
    const signal = scope.current.signal
    operation.current = true
    setBusy('upload')
    try {
      const content = await file.text()
      if (signal.aborted) return
      const fields = { title: draft.title || file.name.replace(/\.(txt|md)$/i, ''), source: draft.source, content }
      const invalid = validateDraft(fields)
      if (invalid) { setError(invalid); return }
      setDraft(fields)
    } catch (reason) {
      if (!signal.aborted) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted) { operation.current = false; setBusy('') }
    }
  }

  function chooseSourceFile(file?: File) {
    if (!file || !enabled || operation.current || documentId) return
    setError('')
    setNotice('')
    const invalid = validateSourceFile(file)
    if (invalid) { setError(invalid); return }
    if (draft.content && !window.confirm('改为上传文件并放弃当前正文？')) return
    setSourceFile(file)
    setDraft({ ...draft, title: draft.title || file.name.replace(/\.(pdf|docx|png|jpe?g)$/i, '') })
  }

  async function uploadSource() {
    if (!sourceFile || !enabled || operation.current || loading || documentId) return
    const invalid = validateSourceFile(sourceFile) || validateUploadMetadata(sourceFile, draft)
    if (invalid) { setError(invalid); return }
    const metadata = JSON.stringify({ title: draft.title, source: draft.source })
    if (uploadRequest.current?.file !== sourceFile || uploadRequest.current.metadata !== metadata) {
      uploadRequest.current = { file: sourceFile, metadata, id: crypto.randomUUID() }
    }
    operation.current = true
    readId.current += 1
    const signal = scope.current.signal
    setBusy('parse')
    setError('')
    setNotice('')
    try {
      const summary = await adminApi.upload(sourceFile, draft, uploadRequest.current.id, signal)
      if (signal.aborted) return
      onDirty(false)
      onUpdate(summary)
    } catch (reason) {
      if (!signal.aborted) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted) { operation.current = false; setBusy('') }
    }
  }

  async function downloadOriginal() {
    if (!detail?.source_file || operation.current) return
    operation.current = true
    const signal = scope.current.signal
    setBusy('download')
    setError('')
    try {
      const blob = await adminApi.original(detail.id, signal)
      if (signal.aborted) return
      const url = URL.createObjectURL(blob)
      try {
        const link = window.document.createElement('a')
        link.href = url
        link.download = detail.source_file.filename
        window.document.body.appendChild(link)
        link.click()
        link.remove()
      } finally {
        URL.revokeObjectURL(url)
      }
    } catch (reason) {
      if (!signal.aborted) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted) { operation.current = false; setBusy('') }
    }
  }

  const locked = !enabled || loading || !!busy || syncActive || unknownConfirmation || (!!documentId && !detail)
  async function changeLifecycle(action: 'delete' | 'restore') {
    if (!detail || locked || operation.current || stale) return
    operation.current = true
    readId.current += 1
    setBusy(action)
    setError('')
    const signal = scope.current.signal
    try {
      await adminApi.lifecycle(detail.id, action, detail.latest_version, detail.catalog_revision ?? 0, signal)
      if (!signal.aborted) onLifecycle()
    } catch (reason) {
      if (!signal.aborted) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted) { operation.current = false; setBusy(''); setConfirmDelete(false) }
    }
  }
  return <section className="knowledge-editor" aria-label="文档编辑器" aria-busy={loading || !!busy}>
    <div className="section-title-row">
      <h2>{documentId ? '共享文档详情' : '新建共享文档'}</h2>
      {documentId && <button className="icon-button" title="重新读取文档" aria-label="重新读取文档" disabled={loading || !!busy || unknownConfirmation} onClick={() => {
        if (!dirty || window.confirm('放弃未保存的修改并重新读取？')) void load(true)
      }}><RefreshCw size={16} /></button>}
    </div>
    <p className="admin-muted">这里管理所有用户共用的知识资料。确认后将自动建立索引并发布；只保存草稿不会影响当前检索内容。</p>
    {loading && <p role="status">加载中...</p>}
    {error && <p role="alert" className="admin-error">{error}</p>}
    {notice && <p role="status" className="admin-success">{notice}</p>}
    {(sync || unknownConfirmation) && <div className="knowledge-sync" aria-label="共享 RAG 更新状态" role="status">
      {syncActive && <p>正在自动更新共享 RAG：{sync?.state === 'pending' ? '等待处理' : '正在建立索引'}。成功发布后所有用户后续检索可使用；如有已发布版本，更新期间会继续保留。</p>}
      {sync?.state === 'failed' && <p className="admin-error">{typeof detail?.published_version === 'number' ? '本次更新失败，原有已发布版本仍可使用。' : '本次更新失败，此资料尚未发布到共享 RAG。'}{sync.error || '请检查资料后重新确认。'}</p>}
      {sync?.state === 'published' && <p className="admin-success">v{sync.version} 已发布到共享 RAG，所有用户后续检索可使用。</p>}
      {sync?.state === 'superseded' && <p>本次确认已被更新版本替代，未自动发布旧版本。</p>}
      {sync?.state === 'withdrawn' && <p>本次更新已撤销，未继续发布。</p>}
      {unknownConfirmation && <p className="admin-warning">确认结果未知，请先核对；修改内容和原请求编号已在当前页面保留。</p>}
      {syncPaused && <p className="admin-warning">{syncPaused}</p>}
      {(syncActive || unknownConfirmation || syncPaused) && <button className="secondary-button" disabled={!!busy} onClick={() => void checkConfirmation()}>核对更新结果</button>}
      {unknownConfirmation && <button className="secondary-button" disabled={!!busy} onClick={() => void confirmUpdate(true)}>使用原请求重试</button>}
    </div>}
    {deleted && <p className="admin-warning">已移入回收站，不参与检索。恢复后仍需手动发布。</p>}
    {confirmDelete && <div className="knowledge-delete-confirm" role="alertdialog" aria-label="删除知识资料" aria-describedby="knowledge-delete-description">
      <h3>删除「{detail?.title}」？</h3>
      <p id="knowledge-delete-description">将取消发布并移入回收站，保留原文和版本记录。{dirty ? '未保存的修改将被放弃。' : ''}</p>
      <button className="secondary-button" disabled={!!busy} onClick={() => setConfirmDelete(false)}>取消</button>
      <button className="secondary-button danger" disabled={!!busy} onClick={() => void changeLifecycle('delete')}><Trash2 size={15} />确认删除</button>
    </div>}
    {detail && <div className="knowledge-metadata">
      <StateBadge state={detail.state} />
      <span>最新版本 v{detail.latest_version}</span>
      <span>已发布版本 {detail.published_version === null ? '无' : `v${detail.published_version}`}</span>
      <span>{detail.chunk_count} 个分块</span>
      <span>{adminDate(detail.updated_at)}</span>
    </div>}
    {detail?.error && <p className="admin-error" role="alert">{detail.error}</p>}
    {stale && <p className="admin-warning" role="status">文档已有更新，请重新读取后继续。</p>}
    {detail?.source_file && <div className="knowledge-source" aria-label="源文件">
      <button className="icon-button knowledge-download" type="button" title="下载原文件" aria-label="下载原文件" disabled={loading || !!busy} onClick={() => void downloadOriginal()}><Download size={16} /></button>
      <dl>
        <div><dt>文件</dt><dd>{detail.source_file.filename}</dd></div>
        <div><dt>格式</dt><dd>{detail.source_file.format}</dd></div>
        <div><dt>解析器</dt><dd>{detail.source_file.parser_version}</dd></div>
        <div><dt>审核</dt><dd>{detail.source_file.reviewed ? '已保存审核稿' : '未保存审核稿'}</dd></div>
      </dl>
      <details><summary>SHA-256</summary><code>{detail.source_file.sha256}</code></details>
      {detail.source_file.warnings.length > 0 && <ul className="admin-warning" aria-label="解析警告">
        {detail.source_file.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
      </ul>}
    </div>}
    {detail?.source_file?.preview_available && <KnowledgeImage
      src={`/api/admin/knowledge/${encodeURIComponent(detail.id)}/preview?version=${detail.latest_version}`}
      filename={detail.source_file.filename} section={originalSection} onForbidden={onForbidden} />}
    <form onSubmit={(event: FormEvent) => { event.preventDefault(); void mutate('save') }}>
      <fieldset disabled={locked || parseLocked || deleted}>
        <div className="knowledge-fields">
          <label>标题<input required={!sourceFile} value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} /></label>
          <label>来源<input value={draft.source} onChange={event => setDraft({ ...draft, source: event.target.value })} /></label>
        </div>
        {!documentId && <div className="knowledge-upload">
          <input ref={sourceFileInput} type="file" accept=".pdf,.docx,.png,.jpg,.jpeg" aria-label="选择文档或图片" hidden onChange={event => {
            const file = event.currentTarget.files?.[0]
            event.currentTarget.value = ''
            chooseSourceFile(file)
          }} />
          <button className="icon-button" type="button" title="选择文档或图片" aria-label="选择文档或图片" onClick={() => sourceFileInput.current?.click()}><Upload size={16} /></button>
          <span className="knowledge-filename">{sourceFile ? sourceFile.name : 'PDF / DOCX / PNG / JPG'}</span>
          {sourceFile && <button className="icon-button" type="button" title="移除待上传文件" aria-label="移除待上传文件" onClick={() => setSourceFile(null)}><X size={16} /></button>}
          <button className="primary-button" type="button" disabled={!sourceFile || !!uploadValidation} onClick={() => void uploadSource()}><Upload size={15} />{busy === 'parse' ? '上传中...' : '上传并解析'}</button>
        </div>}
        {imported ? <div className="knowledge-sections" aria-label="分块审核">
          {!!draft.sections?.length && <>
            <nav className="knowledge-section-nav" aria-label="审核块分页">
              <div className="knowledge-review-field">
                <label htmlFor="knowledge-review-section">审核块</label>
                <select id="knowledge-review-section" value={sectionIndex} onChange={event => setSectionIndex(Number(event.target.value))}>
                  {draft.sections.map((section, index) => <option key={section.id} value={index}>
                    {index + 1} · {detail?.sections?.[index]?.locator || section.id}
                  </option>)}
                </select>
              </div>
              <span>{sectionIndex + 1} / {draft.sections.length}</span>
              <button className="icon-button" type="button" title="上一审核块" aria-label="上一审核块" disabled={sectionIndex === 0} onClick={() => setSectionIndex(index => index - 1)}><ChevronLeft size={16} /></button>
              <button className="icon-button" type="button" title="下一审核块" aria-label="下一审核块" disabled={sectionIndex >= draft.sections.length - 1} onClick={() => setSectionIndex(index => index + 1)}><ChevronRight size={16} /></button>
            </nav>
            {activeSection && <>
              <div className="knowledge-section-meta" id="knowledge-section-source">
                <span>{originalSection?.locator || activeSection.id}</span>
                <span className={sectionEdited ? 'admin-warning' : 'admin-muted'}>{sectionEdited ? '已编辑' : '原文'}</span>
                {originalSection?.ocr_confidence !== undefined && <span className={originalSection.ocr_confidence < 70 ? 'admin-warning' : 'admin-muted'}>OCR 置信分 {originalSection.ocr_confidence.toFixed(1)} / 100 · 待人工核对</span>}
              </div>
              <div className="knowledge-review-field">
                <label htmlFor="knowledge-review-text">审核正文</label>
                <textarea id="knowledge-review-text" rows={10} aria-describedby="knowledge-section-source" value={activeSection.text} onChange={event => {
                  const text = event.target.value
                  setDraft(current => current.sections ? { ...current, sections: current.sections.map(section => section.id === activeSection.id ? { ...section, text } : section) } : current)
                }} />
              </div>
              <p className="admin-muted">{characterCount(activeSection.text)} / {sectionCharacterLimit} 字符</p>
            </>}
          </>}
          {!draft.sections?.length && <p className="admin-muted">{detail?.state === 'parsing' ? '解析中...' : '暂无审核正文。'}</p>}
        </div> : !sourceFile && <>
          <div className="knowledge-review-field">
            <label htmlFor="knowledge-content-text">正文</label>
            <textarea id="knowledge-content-text" required rows={10} value={draft.content ?? ''} onChange={event => setDraft({ title: draft.title, source: draft.source, content: event.target.value })} />
          </div>
          <div className="knowledge-toolbar">
            <span className="admin-muted">{characterCount(draft.content ?? '')} / 60000 字符</span>
            <input ref={fileInput} type="file" accept=".txt,.md,text/plain,text/markdown" aria-label="上传文本文件" hidden onChange={event => {
              const file = event.currentTarget.files?.[0]
              event.currentTarget.value = ''
              void upload(file)
            }} />
            <button className="icon-button" type="button" title="上传文本文件" aria-label="上传文本文件" onClick={() => fileInput.current?.click()}><Upload size={16} /></button>
          </div>
        </>}
        {detail?.state === 'stale' && <p className="admin-warning">此版本的索引已失效。请先保存为新版本，再建立索引；原文不会被删除。</p>}
        {!sourceFile && <div className="knowledge-actions">
          <button className="primary-button" type="button" disabled={!!validation || stale || (!dirty && !!detail && detail.published_version === detail.latest_version)} onClick={() => void confirmUpdate()}><Save size={15} />{busy === 'confirm' ? '正在确认…' : documentId ? '确认修改并更新 RAG' : '确认加入共享 RAG'}</button>
          <button className="secondary-button" type="submit" disabled={!!validation || stale || (!dirty && !!detail && !['review', 'stale'].includes(detail.state))}><Save size={15} />{imported ? '保存审核稿' : '保存草稿'}</button>
        </div>}
        {(sourceFile ? uploadValidation : !parseLocked && validation) && draft.title && <p className="admin-error" role="alert">{sourceFile ? uploadValidation : validation}</p>}
      </fieldset>
    </form>
    {detail && <>
      <div className="knowledge-actions">
        {!deleted && <>
        <button className="secondary-button" disabled={locked || dirty || stale || !canIndex} onClick={() => void mutate('index')}><Database size={15} />{busy === 'index' ? '排队中...' : '建立索引'}</button>
        <button className="secondary-button" disabled={locked || dirty || stale || !canPublish || detail.published_version === detail.latest_version} onClick={() => void mutate('publish')}><Eye size={15} />发布最新版本</button>
        <button className="secondary-button" disabled={locked || detail.published_version === null} onClick={() => void mutate('unpublish')}><EyeOff size={15} />取消发布</button>
        <button className="icon-button danger" title="删除资料" aria-label="删除资料" disabled={locked || stale} onClick={() => setConfirmDelete(true)}><Trash2 size={16} /></button>
        </>}
        {deleted && <button className="secondary-button" disabled={locked} onClick={() => void changeLifecycle('restore')}><RotateCcw size={15} />恢复资料</button>}
        {dirty && <span className="admin-warning">未保存</span>}
      </div>
      <div className="knowledge-preview">
        <div className="workflow-modes" role="tablist" aria-label="文档预览">
          {([['content', '文本预览'], ['chunks', '分块预览'], ['versions', '版本记录']] as const).map(([value, label]) => <button key={value} role="tab" id={`knowledge-${value}-tab`} aria-controls={`knowledge-${value}`} aria-selected={preview === value} onClick={() => setPreview(value)}>{label}</button>)}
        </div>
        <div id={`knowledge-${preview}`} role="tabpanel" aria-labelledby={`knowledge-${preview}-tab`}>
          {preview === 'content' && <>{activeSection && <small>{originalSection?.locator || activeSection.id}</small>}<pre>{(imported ? activeSection?.text : draft.content) || '暂无正文。'}</pre></>}
          {preview === 'chunks' && (detail.chunks.length ? [...detail.chunks].sort((a, b) => a.position - b.position).map(chunk => <article className="knowledge-chunk" key={chunk.id}><small>#{chunk.position} · {chunk.id}{chunk.locator && ` · ${chunk.locator}`}</small><pre>{chunk.text}</pre></article>) : <p className="admin-muted">暂无分块。</p>)}
          {preview === 'versions' && <ul className="knowledge-versions">{detail.versions.map(version => <li key={version.version}><strong>v{version.version}</strong><StateBadge state={version.state} /><span>{version.chunk_count} 个分块</span>{detail.published_version === version.version && <span className="admin-success">已发布</span>}{version.error && <p className="admin-error">{version.error}</p>}</li>)}</ul>}
        </div>
      </div>
    </>}
  </section>
}

function SearchPreview({ enabled, revision, onForbidden }: PermissionProps & { enabled: boolean; revision: number }) {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<KnowledgeSearch | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const scope = useRequestScope()
  const pending = useRef(false)
  const currentRevision = useRef(revision)
  currentRevision.current = revision
  useEffect(() => { setResult(null); setError('') }, [revision])
  async function search(event: FormEvent) {
    event.preventDefault()
    if (!enabled || !query.trim() || pending.current) return
    const signal = scope.current.signal
    const startedRevision = currentRevision.current
    pending.current = true
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const value = await adminApi.search(query.trim(), signal)
      if (!signal.aborted && startedRevision === currentRevision.current) setResult(value)
    } catch (reason) {
      if (!signal.aborted && startedRevision === currentRevision.current) setError(adminError(reason, onForbidden))
    } finally {
      if (!signal.aborted) { pending.current = false; setBusy(false) }
    }
  }
  return <section className="knowledge-search" aria-label="搜索预览">
    <h2><BookOpen size={16} />搜索预览</h2>
    <form onSubmit={event => void search(event)}>
      <label>查询<input value={query} disabled={busy || !enabled} onChange={event => { setQuery(event.target.value); setResult(null); setError('') }} /></label>
      <button className="icon-button" type="submit" title="搜索知识库" aria-label="搜索知识库" disabled={busy || !enabled || !query.trim()}><Search size={17} /></button>
    </form>
    {busy && <p role="status">检索中...</p>}
    {error && <p role="alert" className="admin-error">{error}</p>}
    {result?.status === 'unavailable' && <p role="status" className="admin-warning">知识检索暂不可用。</p>}
    {result?.status === 'empty' && <p role="status" className="admin-muted">未找到匹配的已发布内容。</p>}
    {result?.status === 'ready' && <>
      {result.items.map(item => <article className="knowledge-hit" key={`${item.document_id}:${item.version}:${item.chunk_id}`}>
        <h3>{item.title} <small>v{item.version}</small></h3>
        <small>{item.source || '未注明来源'}{item.locator && ` · ${item.locator}`} · {item.chunk_id} · {item.score.toFixed(3)}</small>
        <pre>{item.text}</pre>
      </article>)}
      <details><summary>检索上下文</summary><pre>{result.context}</pre></details>
    </>}
  </section>
}
