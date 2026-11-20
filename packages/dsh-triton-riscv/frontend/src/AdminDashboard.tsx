import { ArrowLeft, BookOpen, ChevronLeft, ChevronRight, RefreshCw, Users } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { AdminOverview, adminApi, adminDate, adminError } from './admin-api'
import { KnowledgeLibrary } from './KnowledgeLibrary'
import { RagCatalog } from './RagCatalog'
import { accountStatus } from './usage-api'
import './admin.css'

export function AdminDashboard({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<'rag' | 'overview' | 'knowledge'>('rag')
  const [documentRequest, setDocumentRequest] = useState<{ id: string | null; key: number }>({ id: null, key: 0 })
  const [forbidden, setForbidden] = useState(false)
  const ragGuard = useRef<(() => boolean) | null>(null)
  const setRagGuard = useCallback((guard: (() => boolean) | null) => { ragGuard.current = guard }, [])
  const knowledgeGuard = useRef<(() => boolean) | null>(null)
  const setKnowledgeGuard = useCallback((guard: (() => boolean) | null) => { knowledgeGuard.current = guard }, [])
  const mayLeave = () => tab === 'rag' ? (ragGuard.current?.() ?? true) : tab === 'knowledge' ? (knowledgeGuard.current?.() ?? true) : true
  const deny = useCallback(() => setForbidden(true), [])
  const manageDocument = useCallback((id: string | null) => {
    setDocumentRequest(request => ({ id, key: request.key + 1 }))
    setTab('knowledge')
  }, [])
  return (
    <main className="admin-shell" aria-label="管理后台">
      <header className="admin-header">
        <button className="icon-button" title="返回对话" aria-label="返回对话" onClick={() => { if (forbidden || mayLeave()) onClose() }}>
          <ArrowLeft size={18} />
        </button>
        <div className="admin-heading"><h1>管理后台</h1><p>账号使用情况与共享知识资料</p></div>
        {!forbidden && <div className="workflow-modes" role="tablist" aria-label="管理视图">
          <button role="tab" id="admin-rag-tab" aria-controls="admin-rag" aria-selected={tab === 'rag'} onClick={() => { if (mayLeave()) setTab('rag') }}>
            <BookOpen size={16} />RAG 资料
          </button>
          <button role="tab" id="admin-overview-tab" aria-controls="admin-overview" aria-selected={tab === 'overview'} onClick={() => { if (mayLeave()) setTab('overview') }}>
            <Users size={16} />账号概览
          </button>
          <button role="tab" id="admin-knowledge-tab" aria-controls="admin-knowledge" aria-selected={tab === 'knowledge'} onClick={() => { if (mayLeave()) setTab('knowledge') }}>
            <BookOpen size={16} />文档管理
          </button>
        </div>}
      </header>
      {forbidden ? <p className="admin-error admin-content" role="alert">无权访问管理后台，管理员权限可能已变更。</p> : <div className="admin-body admin-tabs-body">
        <section id="admin-rag" role="tabpanel" aria-labelledby="admin-rag-tab" hidden={tab !== 'rag'}>
          <RagCatalog onForbidden={deny} onManageDocument={manageDocument} active={tab === 'rag'} onGuardChange={setRagGuard} />
        </section>
        <section id="admin-overview" role="tabpanel" aria-labelledby="admin-overview-tab" hidden={tab !== 'overview'}>
          <Overview onForbidden={deny} />
        </section>
        <section id="admin-knowledge" role="tabpanel" aria-labelledby="admin-knowledge-tab" hidden={tab !== 'knowledge'}>
          <KnowledgeLibrary onForbidden={deny} requestedDocumentId={documentRequest.id} navigationKey={documentRequest.key} onGuardChange={setKnowledgeGuard} />
        </section>
      </div>}
    </main>
  )
}

function Overview({ onForbidden }: { onForbidden: () => void }) {
  const [overview, setOverview] = useState<AdminOverview | null>(null)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setOverview(null)
    setError('')
    void adminApi.overview(offset, controller.signal).then(result => {
      if (!controller.signal.aborted) setOverview(result)
    }).catch(reason => {
      if (!controller.signal.aborted) setError(adminError(reason, onForbidden))
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [revision, offset, onForbidden])
  function changePage(nextOffset: number) {
    setOverview(null)
    setLoading(true)
    setOffset(nextOffset)
  }
  return <div className="admin-content" aria-busy={loading}>
    <div className="section-title-row">
      <div><h2>账号概览 <span className="usage-scope">全部账号</span></h2><p className="admin-muted">管理员可查看本服务全部账号的使用统计；普通用户只能查看自己的记录。</p></div>
      <button className="icon-button" title="刷新账号概览" aria-label="刷新账号概览" disabled={loading} onClick={() => setRevision(value => value + 1)}><RefreshCw size={16} /></button>
    </div>
    {error && <p className="admin-error" role="alert">{error}</p>}
    {loading && !overview && <p role="status">加载中...</p>}
    {overview && <>
      <dl className="admin-totals">
        <div><dt>账号总数</dt><dd>{overview.totals.users}</dd></div>
        <div><dt>会话记录</dt><dd>{overview.totals.sessions}</dd></div>
        <div><dt>运行记录</dt><dd>{overview.totals.runs}</dd></div>
        <div><dt>失败运行</dt><dd className={overview.totals.failed_runs ? 'admin-error' : ''}>{overview.totals.failed_runs}</dd></div>
      </dl>
      <div className="admin-table-scroll" tabIndex={0} aria-label="账号统计表">
        <table className="admin-table">
          <thead><tr><th>用户名</th><th>状态</th><th>最近使用</th><th>会话</th><th>运行</th><th>失败</th></tr></thead>
          <tbody>{overview.users.map(user => <tr key={user.id}>
            <th scope="row"><span>{user.username}</span><small>{user.id}</small></th>
            <td><span className={`knowledge-state ${user.status === 'active' ? 'ready' : ''}`}>{accountStatus(user.status)}</span></td><td>{adminDate(user.last_used_at)}</td><td>{user.session_count}</td><td>{user.run_count}</td><td>{user.failed_run_count}</td>
          </tr>)}</tbody>
        </table>
      </div>
      {!overview.users.length && <p className="admin-muted">暂无账号。</p>}
    </>}
    <nav className="admin-pagination" aria-label="账号分页">
      <span className="admin-muted">第 {Math.floor(offset / 100) + 1} 页{overview ? ` · 共 ${overview.totals.users} 个账号` : ''}</span>
      <button className="icon-button" title="上一页账号" aria-label="上一页账号" disabled={loading || offset === 0} onClick={() => changePage(Math.max(0, offset - 100))}><ChevronLeft size={16} /></button>
      <button className="icon-button" title="下一页账号" aria-label="下一页账号" disabled={loading || !overview || !overview.users.length || offset + overview.users.length >= overview.totals.users || offset >= 100000} onClick={() => changePage(offset + 100)}><ChevronRight size={16} /></button>
    </nav>
    <p className="usage-footnote admin-muted">统计当前数据库保留的会话与运行记录，包含所有运行状态；暂不统计模型 Token 或费用。</p>
  </div>
}
