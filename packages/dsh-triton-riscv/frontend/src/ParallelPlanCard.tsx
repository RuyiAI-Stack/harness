import { Check, Play, RefreshCw, Square, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { ModelRouteSummary } from './format'
import type { ParallelPlan, ParallelPlanStatus } from './types'

const statusLabels: Record<ParallelPlanStatus, string> = {
  pending: '等待审批',
  approved: '已批准，等待执行',
  rejected: '已拒绝',
  running: '运行中',
  completed: '已完成',
  partial: '部分完成',
  cancelled: '已取消',
  unknown: '状态未知',
}

export function ParallelPlanCard({
  sessionId,
  planId,
  summary,
  disabled = false,
  onContinue,
}: {
  sessionId: string
  planId: string
  summary?: string
  disabled?: boolean
  onContinue: (planId: string) => Promise<void>
}) {
  const [plan, setPlan] = useState<ParallelPlan | null>(null)
  const [pending, setPending] = useState(true)
  const [continued, setContinued] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const revision = useRef(0)
  const locked = useRef(false)

  const checked = (item: ParallelPlan) => {
    if (item.session_id !== sessionId || item.plan_id !== planId || !item.digest) {
      throw new Error('计划标识或摘要不匹配，请刷新后重新审查。')
    }
    return item
  }

  const refresh = async () => {
    if (locked.current) return
    locked.current = true
    const current = ++revision.current
    setPending(true)
    setError(null)
    try {
      const item = checked(await api.getParallelPlan(sessionId, planId))
      if (current === revision.current) setPlan(item)
    } catch (reason) {
      if (current === revision.current) {
        setPlan(null)
        setError(reason instanceof Error ? reason.message : '读取并行计划失败')
      }
    } finally {
      if (current === revision.current) {
        locked.current = false
        setPending(false)
      }
    }
  }

  useEffect(() => {
    locked.current = false
    setPlan(null)
    setContinued(false)
    void refresh()
    return () => {
      revision.current += 1
    }
  }, [sessionId, planId])

  useEffect(() => {
    if (pending || !(plan?.status === 'running' || (plan?.status === 'approved' && continued))) return
    const timer = window.setTimeout(() => void refresh(), 3000)
    return () => window.clearTimeout(timer)
  }, [plan, pending, continued])

  const decide = async (approved: boolean) => {
    if (!plan || plan.status !== 'pending' || locked.current || (approved && disabled)) return
    locked.current = true
    const current = ++revision.current
    setPending(true)
    setError(null)
    let decisionRecorded = false
    try {
      const updated = checked(await api.decideParallelPlan(sessionId, planId, approved, plan.digest))
      if (current !== revision.current) return
      if (updated.digest !== plan.digest) throw new Error('计划摘要已变更，请刷新后重新审查。')
      setPlan(updated)
      decisionRecorded = true
      if (approved && updated.status === 'approved') {
        await onContinue(planId)
        if (current === revision.current) setContinued(true)
      } else if (updated.status !== (approved ? 'approved' : 'rejected')) {
        setError('计划状态已变更，未发送执行请求。')
      }
    } catch (reason) {
      if (current === revision.current) {
        // A failed decision may be stale or already recorded; never retry it blindly.
        if (!decisionRecorded) setPlan(null)
        setError(reason instanceof Error ? reason.message : '提交并行计划审批失败')
      }
    } finally {
      if (current === revision.current) {
        locked.current = false
        setPending(false)
      }
    }
  }

  const continueApproved = async () => {
    if (!plan || plan.status !== 'approved' || locked.current || disabled || continued) return
    locked.current = true
    const current = ++revision.current
    setPending(true)
    setError(null)
    try {
      const updated = checked(await api.getParallelPlan(sessionId, planId))
      if (current !== revision.current) return
      setPlan(updated)
      if (updated.status !== 'approved' || updated.digest !== plan.digest) {
        setError('计划状态或摘要已变更，未发送执行请求。')
        return
      }
      await onContinue(planId)
      if (current === revision.current) setContinued(true)
    } catch (reason) {
      if (current === revision.current) setError(reason instanceof Error ? reason.message : '发送执行请求失败')
    } finally {
      if (current === revision.current) {
        locked.current = false
        setPending(false)
      }
    }
  }

  const cancel = async () => {
    if (!plan || !['approved', 'running'].includes(plan.status) || locked.current) return
    locked.current = true
    const current = ++revision.current
    setPending(true)
    setError(null)
    try {
      const updated = checked(await api.cancelParallelPlan(sessionId, planId))
      if (current === revision.current) setPlan(updated)
    } catch (reason) {
      if (current === revision.current) {
        setPlan(null)
        setError(reason instanceof Error ? reason.message : '取消并行计划失败')
      }
    } finally {
      if (current === revision.current) {
        locked.current = false
        setPending(false)
      }
    }
  }

  return (
    <section className="proposal-review parallel-plan-review" aria-label="并行子任务计划" aria-busy={pending}>
      <div className="section-title-row">
        <h3>并行子任务计划</h3>
        <span className={`run-state ${plan?.status || 'unknown'}`}>
          {plan ? statusLabels[plan.status] || statusLabels.unknown : pending ? '加载中' : '待刷新'}
        </span>
        <button
          className="parallel-refresh secondary-button"
          type="button"
          title="刷新并行计划"
          aria-label="刷新并行计划"
          disabled={pending}
          onClick={() => void refresh()}
        >
          <RefreshCw size={15} />
        </button>
      </div>
      <code>{planId}</code>
      <p>{plan?.reason || summary}</p>
      {plan && (
        <>
          <ol className="parallel-tasks">
            {plan.tasks.map(task => (
              <li key={task.task_id}>
                <code>{task.task_id}</code>
                <span>{task.goal}</span>
                <div>
                  <ModelRouteSummary
                    route={
                      task.route ?? {
                        model: plan.model,
                        provider: plan.provider,
                        timeout_seconds: plan.timeout_seconds,
                        reason: '旧计划未提供路由信息，沿用计划配置',
                      }
                    }
                  />
                </div>
              </li>
            ))}
          </ol>
          <dl className="fact-list">
            <div>
              <dt>默认模型</dt>
              <dd>{plan.model}</dd>
            </div>
            <div>
              <dt>最大并发</dt>
              <dd>{plan.max_parallel}</dd>
            </div>
            <div>
              <dt>单个子任务最长时限</dt>
              <dd>{plan.timeout_seconds} 秒</dd>
            </div>
            <div>
              <dt>只读工具范围</dt>
              <dd>{plan.tools.join(', ') || '无工具'}</dd>
            </div>
          </dl>
          <details className="parallel-context">
            <summary>委派内容与背景</summary>
            <dl>
              <dt>原始用户请求</dt>
              <dd>
                <pre>{plan.user_request || '未提供原始用户请求'}</pre>
              </dd>
              <dt>委派背景</dt>
              <dd>
                <pre>{plan.context || '无额外背景'}</pre>
              </dd>
              {plan.provider && (
                <>
                  <dt>父级模型提供方</dt>
                  <dd>{plan.provider}</dd>
                </>
              )}
              {plan.base_url && (
                <>
                  <dt>父级模型端点</dt>
                  <dd>{plan.base_url}</dd>
                </>
              )}
              {plan.max_tool_calls !== undefined && (
                <>
                  <dt>工具调用上限</dt>
                  <dd>{plan.max_tool_calls}</dd>
                </>
              )}
            </dl>
          </details>
          {plan.results.length > 0 && <p>已返回 {plan.results.length} 条子任务结果</p>}
          {plan.status === 'approved' && continued && <p>继续执行消息已发送，等待执行状态。</p>}
          <div className="proposal-actions">
            {plan.status === 'pending' && (
              <>
                <button
                  className="primary-button"
                  type="button"
                  disabled={pending || disabled}
                  onClick={() => void decide(true)}
                >
                  <Check size={15} /> 批准并继续
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  disabled={pending}
                  onClick={() => void decide(false)}
                >
                  <X size={15} /> 拒绝
                </button>
              </>
            )}
            {plan.status === 'approved' && !continued && (
              <button
                className="primary-button"
                type="button"
                disabled={pending || disabled}
                onClick={() => void continueApproved()}
              >
                <Play size={15} /> 继续已批准计划
              </button>
            )}
            {['approved', 'running'].includes(plan.status) && (
              <button className="secondary-button" type="button" disabled={pending} onClick={() => void cancel()}>
                <Square size={15} /> 取消计划
              </button>
            )}
          </div>
        </>
      )}
      {error && (
        <p className="proposal-error" role="alert">
          {error} 请刷新计划确认最新状态。
        </p>
      )}
    </section>
  )
}
