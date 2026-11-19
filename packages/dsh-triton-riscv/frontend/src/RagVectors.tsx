import { RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { adminDate } from './admin-api'
import { RagVectorCheck, ragApi, ragError } from './rag-api'

const labels: Record<string, string> = {
  present: '实际向量数量与记录一致', missing: '未找到该版本的向量', mismatch: '实际向量数量与记录不一致',
  unavailable: '向量服务暂不可核对', not_checked: '尚未核对实际向量',
}

export function RagVectors({ itemId, version, sourceId, onForbidden }: {
  itemId: string; version: string; sourceId: string; onForbidden: () => void
}) {
  const [result, setResult] = useState<RagVectorCheck | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [reload, setReload] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError(''); setResult(null)
    void ragApi.vectors(itemId, version, controller.signal, sourceId).then(data => {
      if (controller.signal.aborted) return
      if (data.version !== version || !Array.isArray(data.checks)) throw new Error('向量核对结果与当前版本不一致。')
      setResult(data)
    }).catch(reason => { if (!controller.signal.aborted) setError(ragError(reason, onForbidden)) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [itemId, version, sourceId, reload, onForbidden])
  return <section aria-label="实际向量核对">
    <button className="secondary-button" disabled={loading} onClick={() => setReload(value => value + 1)}><RefreshCw size={14} />重新核对向量</button>
    {loading && <p role="status">正在核对向量数据库…</p>}
    {error && <p className="admin-error" role="alert">{error}</p>}
    {result && <>
      <p className="admin-muted">版本 {result.version} · {adminDate(result.checked_at)}</p>
      {result.reason && <p>{result.reason}</p>}
      {!result.complete && <p className="admin-warning">本次仅完成部分核对，达到数量或耗时上限。</p>}
      {result.checks.map((check, position) => <section className="rag-original-section" key={`${check.generation}:${position}`}>
        <h4>{labels[check.state] || '核对状态未知'}</h4>
        <dl className="rag-metadata">
          <div><dt>索引代次</dt><dd>{check.generation}</dd></div>
          <div><dt>记录数量</dt><dd>{check.expected_count}</dd></div>
          <div><dt>实际向量数量</dt><dd>{check.actual_count ?? '未知'}</dd></div>
          <div><dt>索引维度</dt><dd>{check.dimensions ?? '未知'}</dd></div>
        </dl>
        {check.reason && <p className="admin-warning">{check.reason}</p>}
      </section>)}
    </>}
  </section>
}
