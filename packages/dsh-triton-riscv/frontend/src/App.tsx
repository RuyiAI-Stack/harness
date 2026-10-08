import { Bot, Check, Menu, PanelRight, Pin, PinOff, Play, Plus, Send, Trash2, UserRound, X } from 'lucide-react'
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { MessageText, shortDate } from './format'
import type { Bootstrap, Run, RunEvent, ProposalReview, Session, SessionBundle, ValidationPlanReview } from './types'

const terminalStatuses = new Set(['completed', 'failed', 'cancelled'])

function runStatusLabel(status?: string): string {
  const labels: Record<string, string> = {
    'awaiting-confirmation': '等待确认',
    queued: '排队中',
    running: 'Agent 运行中',
    completed: 'Agent 回合已结束',
    failed: 'Agent 运行失败',
    cancelled: 'Agent 已取消',
  }
  return status ? labels[status] || status : '空闲'
}

function evidenceLabel(verdict?: string): string {
  const labels: Record<string, string> = {
    'not-observed': '尚未执行验证',
    planned: '验证等待审批',
    'verified-passed': '证据确认通过',
    'verified-failed': '证据确认失败',
    invalid: '验证证据无效',
  }
  return verdict ? labels[verdict] || verdict : '尚未产生证据'
}

function timelineTitle(event: RunEvent): string {
  if (event.event_type === 'agent-step') {
    const labels: Record<string, string> = {
      analysis: '分析任务',
      'model-progress': '模型进展',
      'tool-call': '调用工具',
      'tool-result': '工具结果',
      'final-answer': '完成回答',
    }
    return labels[event.payload.kind || ''] || 'Agent 步骤'
  }
  const labels: Record<string, string> = {
    planned: '创建任务',
    scheduled: '进入队列',
    started: '启动 Harness',
    completed: 'Agent 回合结束',
    failed: '任务失败',
    cancelled: '任务取消',
    'validation-evidence': '验证证据审计',
  }
  return labels[event.event_type] || event.event_type
}

function uniqueApprovals(rows: RunEvent[]): RunEvent[] {
  const byProposal = new Map<string, RunEvent>()
  rows.forEach(event => {
    const key = event.payload.proposal_id || event.payload.validation_run_id || String(event.id)
    byProposal.set(key, event)
  })
  return [...byProposal.values()]
}

function EmptyConversation() {
  return (
    <div className="empty-state">
      <span className="eyebrow">Natural language workspace</span>
      <h3>从一句任务描述开始</h3>
      <p>DeepSeek Harness 会理解任务、选择 Triton-RISCV 工具，并把模型、工具调用和结果记录到会话时间线。</p>
    </div>
  )
}

function ProposalApproval({
  event,
  onResolved,
  onContinue,
}: {
  event: RunEvent
  onResolved: (eventId: string) => void
  onContinue: (proposalType: 'development' | 'repair', proposalId: string) => Promise<void>
}) {
  const proposalId = event.payload.proposal_id
  const proposalType = event.payload.proposal_type
  const [proposal, setProposal] = useState<ProposalReview | null>(null)
  const [reviewer, setReviewer] = useState('ada-cl25')
  const [pending, setPending] = useState(false)
  const [continued, setContinued] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!proposalId || !proposalType) return
    void api
      .getProposal(proposalType, proposalId)
      .then(item => {
        if (['pending_approval', 'approved'].includes(item.status)) setProposal(item)
        else onResolved(event.id)
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : '读取提案失败'))
  }, [event.id, proposalId, proposalType])

  if (!proposalId || !proposalType) return null

  const decide = async (approve: boolean) => {
    if (!reviewer.trim()) return
    setPending(true)
    setError(null)
    try {
      const updated = await api.decideProposal(proposalType, proposalId, approve, reviewer.trim())
      if (approve) setProposal(updated)
      else onResolved(event.id)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '提交审查失败')
    } finally {
      setPending(false)
    }
  }

  return (
    <section className="proposal-review">
      <div className="section-title-row">
        <h3>{proposalType === 'development' ? '新算子提案' : '修复提案'}</h3>
        <span className={`run-state ${proposal?.status || ''}`}>{proposal?.status || '加载中'}</span>
      </div>
      <code>{proposalId}</code>
      {proposal && (
        <p>
          {proposal.operator}：{proposal.rationale}
        </p>
      )}
      {proposal?.status === 'pending_approval' && (
        <>
          <label>
            审查人
            <input value={reviewer} maxLength={120} onChange={item => setReviewer(item.target.value)} />
          </label>
          <div className="proposal-actions">
            <button
              className="primary-button"
              type="button"
              disabled={pending || !reviewer.trim()}
              onClick={() => void decide(true)}
            >
              <Check size={15} /> 批准
            </button>
            <button className="secondary-button" type="button" disabled={pending} onClick={() => void decide(false)}>
              <X size={15} /> 拒绝
            </button>
          </div>
        </>
      )}
      {proposal?.status === 'approved' && (
        <div className="proposal-actions approved-actions">
          <span>{continued ? '应用任务已启动。' : '提案已批准，尚未应用到仓库。'}</span>
          {!continued && (
            <button
              className="primary-button"
              type="button"
              disabled={pending}
              onClick={() => {
                setPending(true)
                void onContinue(proposalType, proposalId)
                  .then(() => setContinued(true))
                  .finally(() => setPending(false))
              }}
            >
              <Check size={15} /> 继续应用
            </button>
          )}
        </div>
      )}
      {error && <p className="proposal-error">{error}</p>}
    </section>
  )
}

function CommandApproval({
  event,
  onResolved,
  onExecute,
}: {
  event: RunEvent
  onResolved: (eventId: string) => void
  onExecute: (validationRunId: string) => Promise<void>
}) {
  const validationRunId = event.payload.validation_run_id
  const [plan, setPlan] = useState<ValidationPlanReview | null>(null)
  const [reviewer, setReviewer] = useState('ada-cl25')
  const [pending, setPending] = useState(false)
  const [launched, setLaunched] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!validationRunId) return
    void api
      .getValidationPlan(validationRunId)
      .then(item => {
        if (item.approval.status === 'rejected' || item.approval.execution_run_id) {
          onResolved(event.id)
          return
        }
        setPlan(item)
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : '读取验证计划失败'))
  }, [event.id, validationRunId])

  if (!validationRunId) return null

  const decide = async (approve: boolean) => {
    if (!reviewer.trim()) return
    setPending(true)
    setError(null)
    try {
      const updated = await api.decideValidationPlan(validationRunId, approve, reviewer.trim())
      if (!approve) {
        onResolved(event.id)
        return
      }
      setPlan(updated)
      await onExecute(validationRunId)
      setLaunched(true)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '提交命令审查失败')
    } finally {
      setPending(false)
    }
  }

  const executeApproved = async () => {
    setPending(true)
    setError(null)
    try {
      await onExecute(validationRunId)
      setLaunched(true)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '启动验证失败')
    } finally {
      setPending(false)
    }
  }

  return (
    <section className="proposal-review command-review">
      <div className="section-title-row">
        <h3>执行验证命令</h3>
        <span className={`run-state ${plan?.approval.status || ''}`}>{plan?.approval.status || '加载中'}</span>
      </div>
      <p>{plan?.operator || event.payload.operator || '算子'}</p>
      <pre>
        <code>{plan?.command || event.payload.command || '正在读取命令...'}</code>
      </pre>
      {plan?.approval.status === 'pending_approval' && (
        <>
          <label>
            审查人
            <input value={reviewer} maxLength={120} onChange={item => setReviewer(item.target.value)} />
          </label>
          <div className="proposal-actions">
            <button
              className="primary-button"
              type="button"
              disabled={pending || !reviewer.trim()}
              onClick={() => void decide(true)}
            >
              <Play size={15} /> 批准并执行
            </button>
            <button className="secondary-button" type="button" disabled={pending} onClick={() => void decide(false)}>
              <X size={15} /> 拒绝
            </button>
          </div>
        </>
      )}
      {plan?.approval.status === 'approved' && (
        <div className="proposal-actions approved-actions">
          <span>{launched ? '验证任务已启动。' : '命令已批准，尚未执行。'}</span>
          {!launched && (
            <button className="primary-button" type="button" disabled={pending} onClick={() => void executeApproved()}>
              <Play size={15} /> 执行已批准命令
            </button>
          )}
        </div>
      )}
      {error && <p className="proposal-error">{error}</p>}
    </section>
  )
}

export default function App() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [bundle, setBundle] = useState<SessionBundle | null>(null)
  const [activeRun, setActiveRun] = useState<Run | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [approvalEvents, setApprovalEvents] = useState<RunEvent[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [drawer, setDrawer] = useState<'sessions' | 'inspector' | null>(null)
  const [sessionMenu, setSessionMenu] = useState<{
    sessionId: string
    x: number
    y: number
  } | null>(null)
  const eventSource = useRef<EventSource | null>(null)
  const messagesEnd = useRef<HTMLDivElement | null>(null)

  const closeStream = () => {
    eventSource.current?.close()
    eventSource.current = null
  }

  const refreshSessions = async () => {
    const rows = await api.listSessions()
    setSessions(rows)
    return rows
  }

  const refreshApprovals = async (sessionId: string) => {
    const rows = uniqueApprovals(await api.getSessionApprovals(sessionId))
    const unresolved = await Promise.all(
      rows.map(async event => {
        try {
          if (event.payload.validation_run_id) {
            const plan = await api.getValidationPlan(event.payload.validation_run_id)
            return plan.approval.status !== 'rejected' && !plan.approval.execution_run_id
          }
          if (event.payload.proposal_id && event.payload.proposal_type) {
            const proposal = await api.getProposal(event.payload.proposal_type, event.payload.proposal_id)
            return ['pending_approval', 'approved'].includes(proposal.status)
          }
        } catch {
          return true
        }
        return false
      }),
    )
    setApprovalEvents(rows.filter((_, index) => unresolved[index]))
  }

  const loadSession = async (sessionId: string) => {
    closeStream()
    setStreamingText('')
    setActiveSessionId(sessionId)
    const nextBundle = await api.getSession(sessionId)
    setBundle(nextBundle)
    const run = nextBundle.runs[0] || null
    setActiveRun(run)
    setEvents([])
    setDrawer(null)
    await refreshApprovals(sessionId)
    if (run) await loadRun(run, sessionId)
  }

  const createSession = async () => {
    setError(null)
    const session = await api.createSession()
    await refreshSessions()
    await loadSession(session.id)
  }

  const openStream = (runId: string, sessionId: string, after: number) => {
    closeStream()
    const stream = new EventSource(`/api/runs/${runId}/events/stream?after=${after}`)
    eventSource.current = stream
    stream.onmessage = async message => {
      const item = JSON.parse(message.data) as RunEvent
      setEvents(current => [...current, item])
      if (item.event_type === 'assistant-delta' && item.payload.text) {
        setStreamingText(current => current + item.payload.text)
      }
      if (item.event_type === 'agent-step' && item.payload.kind === 'tool-call') {
        setStreamingText('')
      }
      if (item.event_type === 'approval-required') {
        setApprovalEvents(current => uniqueApprovals([...current, item]))
      }
      if (['completed', 'failed', 'cancelled'].includes(item.event_type)) {
        try {
          const run = await api.getRun(runId)
          setActiveRun(run)
          if (terminalStatuses.has(run.status)) {
            setStreamingText('')
            closeStream()
            setBundle(await api.getSession(sessionId))
            await refreshApprovals(sessionId)
          }
        } catch (streamError) {
          setError(streamError instanceof Error ? streamError.message : '读取运行状态失败')
          closeStream()
        }
      }
    }
    stream.onerror = () => {
      setError('任务状态连接已中断，可以重新打开该会话继续查看结果。')
      closeStream()
    }
  }

  const loadRun = async (run: Run, sessionId: string) => {
    const [currentRun, runEvents] = await Promise.all([api.getRun(run.id), api.getEvents(run.id)])
    setActiveRun(currentRun)
    setEvents(runEvents)
    if (['queued', 'running'].includes(currentRun.status)) {
      const after = runEvents.at(-1)?.sequence ?? -1
      openStream(run.id, sessionId, after)
    } else if (terminalStatuses.has(currentRun.status)) {
      setBundle(await api.getSession(sessionId))
    }
  }

  useEffect(() => {
    let cancelled = false
    const start = async () => {
      try {
        const [initialBootstrap, initialSessions] = await Promise.all([api.bootstrap(), api.listSessions()])
        if (cancelled) return
        setBootstrap(initialBootstrap)
        setSessions(initialSessions)
        if (initialSessions.length) await loadSession(initialSessions[0].id)
        else await createSession()
      } catch (startError) {
        if (!cancelled) setError(startError instanceof Error ? startError.message : '平台启动失败')
      }
    }
    void start()
    return () => {
      cancelled = true
      closeStream()
    }
  }, [])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({
      behavior: streamingText ? 'auto' : 'smooth',
      block: 'end',
    })
  }, [bundle?.messages.length, streamingText])

  useEffect(() => {
    if (!sessionMenu) return
    const closeMenu = () => setSessionMenu(null)
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') closeMenu()
    }
    window.addEventListener('click', closeMenu)
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      window.removeEventListener('click', closeMenu)
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [sessionMenu])

  const timeline = events.reduce<RunEvent[]>((items, event) => {
    if (['output', 'approval-required', 'harness-event', 'assistant-delta'].includes(event.event_type)) return items
    return [...items, event]
  }, [])
  const validationEvidence = [...events].reverse().find(event => event.event_type === 'validation-evidence')
  const evidenceVerdict =
    typeof validationEvidence?.payload.verdict === 'string' ? validationEvidence.payload.verdict : undefined
  const evidenceReasons = Array.isArray(validationEvidence?.payload.reasons)
    ? validationEvidence.payload.reasons.map(String)
    : []
  const agentWorking = Boolean(activeRun && ['queued', 'running'].includes(activeRun.status))

  const sendMessage = async (content: string) => {
    if (!content.trim() || !activeSessionId || busy) return
    setBusy(true)
    setStreamingText('')
    setError(null)
    try {
      const result = await api.sendMessage(activeSessionId, content.trim())
      setBundle(current =>
        current
          ? {
              ...current,
              messages: [
                ...current.messages,
                result.user_message,
                ...(result.assistant_message ? [result.assistant_message] : []),
              ],
              runs: result.run ? [result.run, ...current.runs] : current.runs,
            }
          : current,
      )
      if (result.run) {
        await loadRun(result.run, activeSessionId)
      }
      await refreshApprovals(activeSessionId)
      await refreshSessions()
    } catch (sendError) {
      setError(sendError instanceof Error ? sendError.message : '发送失败')
    } finally {
      setBusy(false)
    }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const content = input
    setInput('')
    void sendMessage(content)
  }

  const inputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  const setPinned = async (session: Session) => {
    setSessionMenu(null)
    setError(null)
    try {
      await api.setSessionPinned(session.id, !Boolean(session.pinned))
      await refreshSessions()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '更新会话失败')
    }
  }

  const deleteSession = async (session: Session) => {
    setSessionMenu(null)
    if (!window.confirm(`删除会话“${session.title}”？该操作不可撤销。`)) return
    setError(null)
    try {
      await api.deleteSession(session.id)
      const rows = await refreshSessions()
      if (session.id === activeSessionId) {
        closeStream()
        setBundle(null)
        setActiveRun(null)
        setEvents([])
        setApprovalEvents([])
        if (rows.length) await loadSession(rows[0].id)
        else await createSession()
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '删除会话失败')
    }
  }

  const dismissApproval = (eventId: string) => {
    setApprovalEvents(current => current.filter(event => event.id !== eventId))
  }

  const continueProposal = async (proposalType: 'development' | 'repair', proposalId: string) => {
    const action =
      proposalType === 'development'
        ? '请调用 apply_operator_implementation 应用已批准的新算子提案，然后生成 execute=false 的验证计划。'
        : '请调用 apply_repair 应用已批准的修复提案，然后生成 execute=false 的验证计划。'
    await sendMessage(`提案 ${proposalId} 已由用户批准。${action}`)
  }

  const executeValidation = async (validationRunId: string) => {
    await sendMessage(
      `验证计划 ${validationRunId} 已由用户批准。请调用 validate_operator，设置 execute=true 且 approved_run_id="${validationRunId}"，只执行该计划对应的精确命令。` +
        '执行后如果失败，立即调用 diagnose_failure；若 source_repair_allowed=true，则按 repair_strategy 生成修复提案，等待用户审批后继续验证，直到通过或达到停止条件。',
    )
  }

  return (
    <div className={`app-shell ${drawer ? `show-${drawer}` : ''}`}>
      <header className="topbar">
        <button
          className="mobile-tool"
          type="button"
          title="打开任务会话"
          aria-label="打开任务会话"
          onClick={() => setDrawer('sessions')}
        >
          <Menu size={18} />
        </button>
        <div className="brand-block">
          <div className="brand-mark">RV</div>
          <div>
            <h1>Triton-RISCV Agent</h1>
            <p>算子开发与编译验证工作台</p>
          </div>
        </div>
        <div className="system-state">
          <span className="status-dot" />
          <span>{bootstrap?.harness.api_configured ? 'Harness 已配置' : '等待模型配置'}</span>
        </div>
      </header>

      <aside className="sessions-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Workspace</span>
            <h2>任务会话</h2>
          </div>
          <button
            className="icon-button"
            type="button"
            title="新建任务"
            aria-label="新建任务"
            onClick={() => void createSession()}
          >
            <Plus size={18} />
          </button>
        </div>
        <div className="session-list">
          {sessions.map(session => (
            <button
              className={`session-button ${session.id === activeSessionId ? 'active' : ''}`}
              key={session.id}
              type="button"
              onClick={() => void loadSession(session.id)}
              onContextMenu={event => {
                event.preventDefault()
                setSessionMenu({
                  sessionId: session.id,
                  x: Math.min(event.clientX, window.innerWidth - 170),
                  y: Math.min(event.clientY, window.innerHeight - 100),
                })
              }}
            >
              <strong>
                {Boolean(session.pinned) && <Pin size={11} aria-label="已置顶" />}
                {session.title}
              </strong>
              <span>{shortDate(session.updated_at)}</span>
            </button>
          ))}
        </div>
        <div className="inventory-strip">
          <div>
            <strong>{bootstrap?.operators.operators || 0}</strong>
            <span>算子</span>
          </div>
          <div>
            <strong>{bootstrap?.project.total_targets || 0}</strong>
            <span>验证目标</span>
          </div>
        </div>
      </aside>

      <main className="chat-panel">
        <div className="chat-heading">
          <div>
            <span className="eyebrow">Conversation</span>
            <h2>{bundle?.session.title || '新对话'}</h2>
          </div>
          <div className="chat-heading-actions">
            <span className="mode-badge">DeepSeek Harness</span>
            <button
              className="icon-button"
              type="button"
              title={drawer === 'inspector' ? '关闭任务详情' : '打开任务详情'}
              aria-label={drawer === 'inspector' ? '关闭任务详情' : '打开任务详情'}
              aria-expanded={drawer === 'inspector'}
              onClick={() => setDrawer(current => (current === 'inspector' ? null : 'inspector'))}
            >
              <PanelRight size={18} />
            </button>
          </div>
        </div>
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button type="button" title="关闭错误提示" aria-label="关闭错误提示" onClick={() => setError(null)}>
              <X size={16} />
            </button>
          </div>
        )}
        <div className="messages" aria-live="polite">
          {!bundle?.messages.length ? (
            <EmptyConversation />
          ) : (
            bundle.messages.map(message => {
              return (
                <article className={`message ${message.role}`} key={message.id}>
                  {message.role === 'assistant' && (
                    <div className="message-avatar agent-avatar" role="img" aria-label="Agent">
                      <Bot size={17} />
                    </div>
                  )}
                  <div className="message-content">
                    <div className="message-body">
                      <MessageText value={message.content} />
                    </div>
                  </div>
                  {message.role === 'user' && (
                    <div className="message-avatar user-avatar" role="img" aria-label="用户">
                      <UserRound size={17} />
                    </div>
                  )}
                </article>
              )
            })
          )}
          {streamingText && (
            <article className="message assistant streaming-message" aria-label="Agent 正在回复">
              <div className="message-avatar agent-avatar" role="img" aria-label="Agent">
                <Bot size={17} />
              </div>
              <div className="message-content">
                <div className="message-body">
                  <MessageText value={streamingText} />
                  <span className="typing-cursor" aria-hidden="true" />
                </div>
              </div>
            </article>
          )}
          {agentWorking && !streamingText && (
            <article className="message assistant streaming-message" aria-label="Agent 正在处理任务">
              <div className="message-avatar agent-avatar" role="img" aria-label="Agent">
                <Bot size={17} />
              </div>
              <div className="message-content">
                <div className="message-body typing-indicator" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </article>
          )}
          <div ref={messagesEnd} />
        </div>
        {approvalEvents.length > 0 && (
          <div className="approval-dock" aria-label="待审批提案">
            {approvalEvents.map(event =>
              event.payload.validation_run_id ? (
                <CommandApproval
                  event={event}
                  key={event.id}
                  onResolved={dismissApproval}
                  onExecute={executeValidation}
                />
              ) : (
                <ProposalApproval
                  event={event}
                  key={event.id}
                  onResolved={dismissApproval}
                  onContinue={continueProposal}
                />
              ),
            )}
          </div>
        )}
        <div className="suggestions">
          {bootstrap?.suggestions.map(suggestion => (
            <button
              className="suggestion-button"
              type="button"
              title={suggestion}
              key={suggestion}
              onClick={() => setInput(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
        <form className="composer" onSubmit={submit}>
          <textarea
            value={input}
            rows={2}
            maxLength={12000}
            placeholder="描述你想检查、开发或分析的 Triton-RISCV 任务..."
            aria-label="任务描述"
            onChange={event => setInput(event.target.value)}
            onKeyDown={inputKeyDown}
          />
          <button
            className="send-button"
            type="submit"
            title="发送任务"
            aria-label="发送任务"
            disabled={busy || !input.trim()}
          >
            <Send size={18} />
          </button>
        </form>
      </main>

      <aside className="inspector-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Agent State</span>
            <h2>任务详情</h2>
          </div>
          <button
            className="icon-button"
            type="button"
            title="关闭任务详情"
            aria-label="关闭任务详情"
            onClick={() => setDrawer(null)}
          >
            <X size={18} />
          </button>
        </div>
        <section className="inspector-section">
          <h3>Agent 运行时</h3>
          <dl className="fact-list">
            <div>
              <dt>框架</dt>
              <dd>{bootstrap?.harness.runtime || '加载中'}</dd>
            </div>
            <div>
              <dt>领域插件</dt>
              <dd>
                {bootstrap?.harness.plugin.loaded
                  ? `${bootstrap.harness.plugin.name} ${bootstrap.harness.plugin.version}`
                  : '未加载'}
              </dd>
            </div>
            <div>
              <dt>模型</dt>
              <dd>{bootstrap?.harness.model || '未配置'}</dd>
            </div>
            <div>
              <dt>API</dt>
              <dd>{bootstrap?.harness.api_configured ? '已配置' : '未配置'}</dd>
            </div>
            <div>
              <dt>RISC-V</dt>
              <dd>{bootstrap?.harness.remote_configured ? '已配置' : '未配置'}</dd>
            </div>
          </dl>
        </section>
        <section className="inspector-section">
          <div className="section-title-row">
            <h3>可信执行结论</h3>
            <span className={`run-state ${evidenceVerdict || 'not-observed'}`}>{evidenceLabel(evidenceVerdict)}</span>
          </div>
          <dl className="fact-list evidence-facts">
            <div>
              <dt>Agent</dt>
              <dd>{runStatusLabel(activeRun?.status)}</dd>
            </div>
            <div>
              <dt>算子验证</dt>
              <dd>{evidenceLabel(evidenceVerdict)}</dd>
            </div>
            {Boolean(validationEvidence?.payload.operator) && (
              <div>
                <dt>算子</dt>
                <dd>{String(validationEvidence?.payload.operator)}</dd>
              </div>
            )}
            {Boolean(validationEvidence?.payload.run_id) && (
              <div>
                <dt>证据 ID</dt>
                <dd>{String(validationEvidence?.payload.run_id)}</dd>
              </div>
            )}
          </dl>
          {evidenceReasons.length > 0 && <p className="evidence-reason">{evidenceReasons.join('；')}</p>}
        </section>
        <section className="inspector-section">
          <div className="section-title-row">
            <h3>运行时间线</h3>
            <span className={`run-state ${activeRun?.status || ''}`}>{runStatusLabel(activeRun?.status)}</span>
          </div>
          <ol className="timeline">
            {!timeline.length ? (
              <li className="muted">发送任务后显示运行阶段</li>
            ) : (
              timeline.map((event, index) => {
                const label = event.payload.message || event.payload.command || event.event_type
                const active = index === timeline.length - 1 && activeRun?.status === 'running'
                return (
                  <li className={active ? 'active' : 'done'} key={event.id}>
                    <strong>{timelineTitle(event)}</strong>
                    <span>{String(label)}</span>
                    {event.payload.detail && <code>{event.payload.detail}</code>}
                  </li>
                )
              })
            )}
          </ol>
        </section>
      </aside>

      {sessionMenu &&
        (() => {
          const session = sessions.find(item => item.id === sessionMenu.sessionId)
          if (!session) return null
          return (
            <div
              className="session-menu"
              role="menu"
              style={{ left: sessionMenu.x, top: sessionMenu.y }}
              onClick={event => event.stopPropagation()}
            >
              <button type="button" role="menuitem" onClick={() => void setPinned(session)}>
                {Boolean(session.pinned) ? <PinOff size={15} /> : <Pin size={15} />}
                {Boolean(session.pinned) ? '取消置顶' : '置顶会话'}
              </button>
              <button className="danger" type="button" role="menuitem" onClick={() => void deleteSession(session)}>
                <Trash2 size={15} /> 删除会话
              </button>
            </div>
          )
        })()}

      {drawer && <button className="scrim" type="button" aria-label="关闭侧栏" onClick={() => setDrawer(null)} />}
    </div>
  )
}
