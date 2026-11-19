import { Activity, ArrowLeft, Clock3, MessageSquare, RefreshCw, ShieldCheck, TriangleAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { adminDate } from './admin-api'
import { AccountUsage, accountStatus, readAccountUsage, runStatus } from './usage-api'
import './admin.css'

export function UsageDashboard({ accountId, onClose }: { accountId: string; onClose: () => void }) {
  const [usage, setUsage] = useState<AccountUsage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setUsage(null)
    setError('')
    void readAccountUsage(accountId, controller.signal).then(result => {
      if (!controller.signal.aborted) setUsage(result)
    }).catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '使用情况暂时无法读取，请重试。')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [accountId, revision])

  return <main className="admin-shell usage-shell" aria-label="我的使用情况">
    <header className="admin-header">
      <button className="icon-button" aria-label="返回对话" title="返回对话" onClick={onClose}><ArrowLeft size={18} /></button>
      <div className="admin-heading"><h1>我的使用情况</h1><p>查看当前账号的活动记录</p></div>
      <span className="usage-scope"><ShieldCheck size={14} />当前账号</span>
    </header>
    <div className="admin-body">
      <div className="admin-content usage-content" aria-busy={loading}>
        <div className="section-title-row usage-title-row">
          <div><h2>使用概览</h2><p className="admin-muted">仅统计当前账号；不包含其他用户的数据。</p></div>
          <button className="secondary-button" aria-label="刷新我的使用情况" disabled={loading} onClick={() => setRevision(value => value + 1)}><RefreshCw size={15} />刷新</button>
        </div>
        {loading && <div className="usage-empty" role="status">正在读取你的使用情况…</div>}
        {error && <div className="usage-empty usage-error" role="alert"><TriangleAlert size={21} /><p>{error}</p><button className="secondary-button" onClick={() => setRevision(value => value + 1)}>重新读取</button></div>}
        {usage && <>
          <div className="usage-account"><strong>{usage.account.username}</strong><span className="knowledge-state ready">{accountStatus(usage.account.status)}</span><span className="admin-muted">截至 {adminDate(usage.as_of)}</span></div>
          <dl className="admin-totals usage-totals">
            <div><dt><MessageSquare size={16} />会话记录</dt><dd>{usage.totals.sessions.toLocaleString('zh-CN')}</dd><small>当前保存的会话记录</small></div>
            <div><dt><Activity size={16} />运行记录</dt><dd>{usage.totals.runs.toLocaleString('zh-CN')}</dd><small>所有状态的运行记录</small></div>
            <div><dt><TriangleAlert size={16} />失败运行</dt><dd className={usage.totals.failed_runs ? 'admin-error' : ''}>{usage.totals.failed_runs.toLocaleString('zh-CN')}</dd><small>运行状态为失败</small></div>
            <div><dt><Clock3 size={16} />最近活动</dt><dd className="usage-date">{adminDate(usage.last_used_at)}</dd><small>最近会话或运行更新</small></div>
          </dl>
          <section className="usage-recent" aria-labelledby="usage-recent-title">
            <div className="section-title-row"><h2 id="usage-recent-title">最近运行</h2><span className="admin-muted">最多显示 10 条</span></div>
            {usage.recent_runs.length ? <ul className="usage-run-list">{usage.recent_runs.map(run => <li key={run.id}>
              <div className="usage-run-name"><span className={`knowledge-state ${run.status === 'failed' ? 'failed' : ['running', 'queued'].includes(run.status) ? 'pending' : ''}`}>{runStatus(run.status)}</span><code title={run.id}>{run.id}</code></div>
              <dl><div><dt>创建</dt><dd>{adminDate(run.created_at)}</dd></div><div><dt>更新</dt><dd>{adminDate(run.updated_at)}</dd></div></dl>
            </li>)}</ul> : <div className="usage-empty"><Activity size={25} /><p>还没有运行记录</p><span className="admin-muted">在工作台开始任务后，可在这里查看使用情况。</span></div>}
          </section>
          <p className="usage-footnote admin-muted">这里展示会话与运行记录，暂不统计模型 Token 或费用。</p>
        </>}
      </div>
    </div>
  </main>
}
