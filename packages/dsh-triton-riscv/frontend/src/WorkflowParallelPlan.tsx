import { Check, RefreshCw, X } from 'lucide-react'
import { ModelRouteSummary } from './format'
import type { ParallelPlanStatus } from './types'
import type { WorkflowParallelPlanView } from './workflow-api'

const statuses: Record<ParallelPlanStatus, string> = {
  pending: '等待并行审批',
  approved: '已批准并行，等待执行',
  rejected: '已选择不并行，继续顺序处理',
  running: '并行运行中',
  completed: '并行子任务已完成',
  partial: '并行子任务部分完成',
  cancelled: '并行已取消',
  unknown: '并行状态未知',
}

export function WorkflowParallelPlan({
  plan,
  waiting,
  busy,
  disabled,
  onDecide,
  onRefresh,
}: {
  plan?: WorkflowParallelPlanView | null
  waiting: boolean
  busy: boolean
  disabled: boolean
  onDecide: (approved: boolean) => void
  onRefresh: () => void
}) {
  const fallback = plan?.workflow_status === 'fallback'
  return (
    <section className={`workflow-approval parallel-plan-review${waiting ? ' awaiting-decision' : ''}`} aria-label="工作流并行计划" aria-busy={busy}>
      <div className="section-title-row">
        <h3>可选并行子任务</h3>
        <span className={`run-state ${fallback ? 'fallback' : plan?.status || 'unknown'}`}>
          {fallback ? '已回退为顺序处理' : plan ? statuses[plan.status] || statuses.unknown : '待刷新'}
        </span>
        <button
          className="parallel-refresh secondary-button"
          type="button"
          title="刷新工作流并行计划"
          aria-label="刷新工作流并行计划"
          disabled={busy}
          onClick={onRefresh}
        >
          <RefreshCw size={15} />
        </button>
      </div>
      {!plan ? (
        <p role="alert">并行计划尚未加载，请刷新后审查。</p>
      ) : (
        <>
          {waiting && !fallback && plan.status === 'pending' && (
            <>
              <p>{plan.tasks.length} 个只读子任务 · 最大并发 {plan.max_parallel} · 不含代码写入或测试授权</p>
              <div className="proposal-actions">
                <button
                  className="primary-button"
                  type="button"
                  disabled={busy || disabled}
                  onClick={() => onDecide(true)}
                >
                  <Check size={15} /> 批准只读并行
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy || disabled}
                  onClick={() => onDecide(false)}
                >
                  <X size={15} /> 不并行，继续处理
                </button>
              </div>
            </>
          )}
          <code>{plan.plan_id}</code>
          {fallback && <p role="status">并行回退原因：{plan.fallback_reason || '未提供原因'}</p>}
          <p>{plan.reason}</p>
          <ol className="parallel-tasks">
            {plan.tasks.map(task => (
              <li key={task.task_id}>
                <code>{task.task_id}</code> · {task.goal}
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
            <div>
              <dt>工具调用上限</dt>
              <dd>{plan.max_tool_calls ?? '未提供'}</dd>
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
            </dl>
          </details>
          {plan.results.length > 0 && (
            <details className="parallel-context">
              <summary>已返回 {plan.results.length} 条子任务结果</summary>
              <pre>{JSON.stringify(plan.results, null, 2)}</pre>
            </details>
          )}
        </>
      )}
    </section>
  )
}
