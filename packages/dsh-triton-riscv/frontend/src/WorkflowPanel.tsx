import { Check, GitBranch, RefreshCw, Send, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { MessageText } from './format'
import { workflowApi, type Workflow, type WorkflowSubmission } from './workflow-api'

const phases: Record<string, string> = {
  preparing: '确认需求',
  planning_tests: '设计验收测试',
  implementing: '提出实现',
  reviewing: '独立审查',
  waiting_approval: '等待方案审批',
  applying: '写入代码',
  waiting_validation_approval: '等待验证审批',
  waiting_validation: '验证中',
  succeeded: '验证通过',
  failed: '本次尝试失败',
  blocked: '需要补充信息',
  cancelled: '已取消',
  needs_reconciliation: '需要核对执行记录',
}
const roles: Record<string, string> = {
  main: '主 Agent',
  semantic: '语义 Agent',
  implementer: '实现 Agent',
  reviewer: '测试与审查 Agent',
}

export function WorkflowPanel({ sessionId, enabled }: { sessionId: string; enabled: boolean }) {
  const [rows, setRows] = useState<Workflow[]>([])
  const [task, setTask] = useState<Workflow | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [kind, setKind] = useState<'create' | 'repair'>('create')
  const [operator, setOperator] = useState('')
  const [failure, setFailure] = useState('')
  const [text, setText] = useState('')
  const [autoRepair, setAutoRepair] = useState(true)
  const [budget, setBudget] = useState(3)
  const [reviewer, setReviewer] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef<{ fingerprint: string; body: WorkflowSubmission } | null>(null)
  const generation = useRef(0)

  useEffect(() => {
    const current = ++generation.current
    let timer: ReturnType<typeof setTimeout>
    let active = true
    const refresh = async () => {
      try {
        const items = await workflowApi.list(sessionId)
        if (!active) return
        setRows(items)
        const id = selected || items[0]?.id
        if (id) {
          const detail = await workflowApi.get(id)
          if (active && current === generation.current) setTask(detail)
        } else setTask(null)
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : '读取任务失败')
      } finally {
        if (active) timer = setTimeout(() => void refresh(), 2000)
      }
    }
    void refresh()
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [sessionId, selected])

  const act = async (action: () => Promise<unknown>, reload = true) => {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      await action()
      if (task && reload) setTask(await workflowApi.get(task.id))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '操作失败')
      if (task) setTask(await workflowApi.get(task.id).catch(() => task))
    } finally {
      setBusy(false)
    }
  }

  const submit = async () => {
    const body: WorkflowSubmission = {
      request: {
        session_id: sessionId,
        kind,
        operator: operator.trim(),
        requirement: text.trim(),
        auto_publish: true,
        auto_repair: autoRepair,
        max_repairs: budget,
      },
      request_id: '',
      ...(kind === 'repair' ? { failure_run_id: failure.trim() } : {}),
    }
    const fingerprint = JSON.stringify(body)
    if (pending.current?.fingerprint !== fingerprint)
      pending.current = { fingerprint, body: { ...body, request_id: crypto.randomUUID() } }
    const job = await workflowApi.submit(pending.current.body)
    setSelected(job.payload.workflow_id)
    setTask(await workflowApi.get(job.payload.workflow_id))
    pending.current = null
    setText('')
  }

  const proposal = task?.proposal
  const plan = task?.validation_plan
  const blocked = Boolean(
    task?.artifact_error || task?.result.lifecycle?.operation || task?.state === 'needs_reconciliation',
  )
  const authorized = reviewer.trim().length > 0 && !busy && !blocked
  const command = (name: string) => task && act(() => workflowApi.action(task, name))

  const decide = async (approve: boolean) => {
    if (!task) return
    if (task.state === 'waiting_approval' && proposal) {
      await api.decideProposal(
        task.kind === 'create' ? 'development' : 'repair',
        proposal.proposal_id,
        approve,
        reviewer.trim(),
      )
      if (approve) {
        let next = await workflowApi.get(task.id)
        next = await workflowApi.action(next, 'apply')
        await workflowApi.action(next, 'plan-validation')
      }
    } else if (plan) {
      await api.decideValidationPlan(plan.run_id, approve, reviewer.trim())
      if (approve) await workflowApi.action(await workflowApi.get(task.id), 'enqueue-validation')
    }
  }

  return (
    <section className="workflow-panel" aria-label="多智能体任务">
      <div className="workflow-toolbar">
        <GitBranch size={17} aria-hidden="true" />
        <select
          aria-label="已保存任务"
          disabled={busy}
          value={selected || task?.id || ''}
          onChange={event => {
            setTask(null)
            setSelected(event.target.value)
          }}
        >
          {!rows.length && <option value="">暂无任务</option>}
          {rows.map(row => (
            <option key={row.id} value={row.id}>
              {row.operator} · 修复 {row.repair_count}/{row.max_repairs} · {phases[row.state]}
            </option>
          ))}
        </select>
        <span className={`run-state ${task?.state}`}>
          {enabled ? phases[task?.state || ''] || '就绪' : '多智能体未启用'}
        </span>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          {error}
          <button title="关闭错误" aria-label="关闭错误" onClick={() => setError('')}>
            <X size={15} />
          </button>
        </div>
      )}
      <div className="workflow-content">
        {task && (
          <>
            <h3>{task.operator}</h3>
            <div className="workflow-requirement">
              <MessageText value={task.request.requirement} />
            </div>
            <dl className="workflow-facts">
              <div>
                <dt>任务 ID</dt>
                <dd>{task.id}</dd>
              </div>
              <div>
                <dt>审查返工</dt>
                <dd>
                  {task.result.rework_count || 0}/{task.max_repairs}
                </dd>
              </div>
              <div>
                <dt>执行修复</dt>
                <dd>
                  {task.repair_count}/{task.max_repairs}
                </dd>
              </div>
            </dl>
            <ol className="timeline">
              {task.steps.map(step => (
                <li key={step.id}>
                  <strong>
                    {roles[step.role]} · {phases[step.state]}
                  </strong>
                  <span>{step.result.summary || step.status}</span>
                  {step.result.artifact_id && (
                    <a
                      href={`/api/jobs/${step.job_id}/artifacts/${step.result.artifact_id}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      输入与输出记录
                    </a>
                  )}
                </li>
              ))}
            </ol>
            {(task.result.reason || task.result.lifecycle?.reason || task.job?.result.reason) && (
              <p role="status">{task.result.reason || task.result.lifecycle?.reason || task.job?.result.reason}</p>
            )}
            {task.result.lifecycle?.child_workflow_id && (
              <button
                type="button"
                disabled={busy}
                className="secondary-button"
                onClick={() => {
                  setTask(null)
                  setSelected(task.result.lifecycle!.child_workflow_id!)
                }}
              >
                <GitBranch size={15} />
                查看自动修复任务
              </button>
            )}
            {task.result.parent_workflow_id && (
              <button
                type="button"
                disabled={busy}
                className="secondary-button"
                onClick={() => {
                  setTask(null)
                  setSelected(task.result.parent_workflow_id!)
                }}
              >
                <GitBranch size={15} />
                查看上次失败
              </button>
            )}
            {task.state === 'waiting_approval' && !proposal && !blocked && (
              <button className="primary-button" disabled={busy} onClick={() => void command('publish')}>
                <Check size={15} />
                发布待审提案
              </button>
            )}
            {task.state === 'waiting_approval' && proposal && (
              <div className="workflow-approval">
                <h3>代码方案 · {proposal.status}</h3>
                <p>{proposal.rationale}</p>
                {task.acceptance && (
                  <details open>
                    <summary>算子要求与验收条件</summary>
                    <pre>{JSON.stringify(task.acceptance, null, 2)}</pre>
                  </details>
                )}
                <details open>
                  <summary>查看代码差异</summary>
                  <pre>{proposal.diff || '无代码差异'}</pre>
                </details>
                {proposal.status === 'pending_approval' && (
                  <>
                    <label>
                      审查人
                      <input aria-label="审查人" value={reviewer} onChange={event => setReviewer(event.target.value)} />
                    </label>
                    <div className="proposal-actions">
                      <button
                        className="primary-button"
                        disabled={!authorized}
                        onClick={() => void act(() => decide(true))}
                      >
                        <Check size={15} />
                        批准并应用
                      </button>
                      <button
                        className="secondary-button"
                        disabled={!authorized}
                        onClick={() => void act(() => decide(false))}
                      >
                        <X size={15} />
                        拒绝
                      </button>
                    </div>
                  </>
                )}
                {proposal.status === 'approved' && (
                  <button className="primary-button" disabled={busy || blocked} onClick={() => void command('apply')}>
                    <Check size={15} />
                    应用已批准方案
                  </button>
                )}
              </div>
            )}
            {task.state === 'waiting_validation_approval' && !plan && (
              <button
                className="primary-button"
                disabled={busy || blocked}
                onClick={() => void command('plan-validation')}
              >
                <Check size={15} />
                生成验证计划
              </button>
            )}
            {task.state === 'waiting_validation_approval' && plan && (
              <div className="workflow-approval">
                <h3>验证命令 · {plan.execution_target === 'remote' ? '远程' : '本机'}</h3>
                <pre>{plan.command}</pre>
                {plan.approval.status === 'pending_approval' && (
                  <>
                    <label>
                      审查人
                      <input aria-label="审查人" value={reviewer} onChange={event => setReviewer(event.target.value)} />
                    </label>
                    <div className="proposal-actions">
                      <button
                        className="primary-button"
                        disabled={!authorized}
                        onClick={() => void act(() => decide(true))}
                      >
                        <Check size={15} />
                        批准并执行
                      </button>
                      <button
                        className="secondary-button"
                        disabled={!authorized}
                        onClick={() => void act(() => decide(false))}
                      >
                        <X size={15} />
                        拒绝
                      </button>
                    </div>
                  </>
                )}
                {plan.approval.status === 'approved' && (
                  <button
                    className="primary-button"
                    disabled={busy || blocked}
                    onClick={() => void command('enqueue-validation')}
                  >
                    <Check size={15} />
                    执行已批准计划
                  </button>
                )}
                {plan.approval.status === 'rejected' && <p>验证计划已拒绝，未执行。</p>}
              </div>
            )}
            {task.state === 'waiting_validation' && (
              <button className="secondary-button" disabled={busy} onClick={() => void command('collect-validation')}>
                <RefreshCw size={15} />
                核对已保存结果
              </button>
            )}
            {blocked && (
              <p role="alert">执行结果尚未核清，已停止自动重试。请核对任务产物和执行日志，不要重复应用方案。</p>
            )}
            {task.job?.status === 'needs-reconciliation' && (
              <p role="alert">模型调用结果未知，请先核对 Worker 记录。</p>
            )}
          </>
        )}
      </div>
      <form
        className="workflow-composer"
        onSubmit={event => {
          event.preventDefault()
          void act(submit, false)
        }}
      >
        <div className="workflow-fields">
          <label>
            任务类型
            <select
              aria-label="任务类型"
              value={kind}
              onChange={event => setKind(event.target.value as 'create' | 'repair')}
            >
              <option value="create">创建算子</option>
              <option value="repair">修复算子</option>
            </select>
          </label>
          <label>
            算子名称
            <input
              aria-label="算子名称"
              required
              pattern="[A-Za-z_][A-Za-z0-9_]{0,127}"
              value={operator}
              onChange={event => setOperator(event.target.value)}
            />
          </label>
          {kind === 'repair' && (
            <label>
              失败回执 ID
              <input
                aria-label="失败回执 ID"
                required
                value={failure}
                onChange={event => setFailure(event.target.value)}
              />
            </label>
          )}
          <label>
            重试上限
            <input
              aria-label="重试上限"
              type="number"
              min={1}
              max={20}
              value={budget}
              onChange={event => setBudget(Number(event.target.value))}
            />
          </label>
        </div>
        <label className="workflow-toggle">
          <input type="checkbox" checked={autoRepair} onChange={event => setAutoRepair(event.target.checked)} />
          验证失败后自动提出修复方案
        </label>
        <div className="workflow-send">
          <textarea
            aria-label="多智能体需求"
            placeholder="公式、输入输出、dtype、shape、误差与边界要求"
            required
            maxLength={8000}
            rows={3}
            value={text}
            onChange={event => setText(event.target.value)}
          />
          <button
            className="send-button"
            type="submit"
            aria-label="发送多智能体任务"
            title="发送多智能体任务"
            disabled={!enabled || busy || !text.trim() || !operator.trim()}
          >
            <Send size={18} />
          </button>
        </div>
      </form>
    </section>
  )
}
