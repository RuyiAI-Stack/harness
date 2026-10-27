import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { api } from './api'
import { workflowApi } from './workflow-api'
import type { Bootstrap, Run, RunEvent, Session, SessionBundle } from './types'

vi.mock('./api', () => ({
  api: {
    bootstrap: vi.fn(),
    listSessions: vi.fn(),
    createSession: vi.fn(),
    getSession: vi.fn(),
    setSessionPinned: vi.fn(),
    deleteSession: vi.fn(),
    getSessionApprovals: vi.fn(),
    sendMessage: vi.fn(),
    getRun: vi.fn(),
    getEvents: vi.fn(),
    getProposal: vi.fn(),
    decideProposal: vi.fn(),
    getValidationPlan: vi.fn(),
    decideValidationPlan: vi.fn(),
  },
}))
vi.mock('./workflow-api', () => ({ workflowApi: { config: vi.fn(), list: vi.fn(), get: vi.fn() } }))

class EventStream {
  static instances: EventStream[] = []
  onmessage: ((message: { data: string }) => void | Promise<void>) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()
  constructor(public url: string) {
    EventStream.instances.push(this)
  }
  async emit(event: RunEvent) {
    await this.onmessage?.({ data: JSON.stringify(event) })
  }
}

const session: Session = {
  id: 'session-one',
  title: 'relu_and_mul',
  pinned: false,
  created_at: '2026-10-11T10:00:00',
  updated_at: '2026-10-11T10:00:00',
}
const run: Run = { id: 'run-one', session_id: session.id, status: 'running', created_at: '', updated_at: '' }
const bootstrap: Bootstrap = {
  operators: { operators: 261 },
  project: { total_targets: 682 },
  harness: {
    runtime: 'Harness',
    plugin: { name: 'dsh-triton-riscv', version: '0.1.0', loaded: true },
    provider: 'test',
    model: 'test-model',
    api_configured: true,
    remote_configured: true,
  },
  suggestions: ['检查 relu_and_mul'],
}
const bundle = (value = session): SessionBundle => ({
  session: value,
  runs: [],
  messages: [
    { id: 'user-one', session_id: value.id, role: 'user', content: '检查实现和测试', metadata: {}, created_at: '' },
    {
      id: 'assistant-one',
      session_id: value.id,
      role: 'assistant',
      content: '找到 `relu_and_mul.py`，尚未执行测试。',
      metadata: {},
      created_at: '',
    },
  ],
})
const event = (event_type: string, payload: RunEvent['payload'] = {}): RunEvent => ({
  id: `event-${event_type}`,
  run_id: run.id,
  sequence: 1,
  event_type,
  payload,
  created_at: '',
})

async function mount() {
  const view = render(<App />)
  await screen.findByText('找到', { exact: false })
  await waitFor(() => expect(api.getSessionApprovals).toHaveBeenCalledWith(session.id))
  return view
}

describe('workbench interactions', () => {
  const originalScroll = HTMLElement.prototype.scrollIntoView
  beforeEach(() => {
    vi.resetAllMocks()
    EventStream.instances = []
    vi.stubGlobal('EventSource', EventStream)
    HTMLElement.prototype.scrollIntoView = vi.fn()
    vi.mocked(api.bootstrap).mockResolvedValue(bootstrap)
    vi.mocked(api.listSessions).mockResolvedValue([session])
    vi.mocked(api.getSession).mockResolvedValue(bundle())
    vi.mocked(api.getSessionApprovals).mockResolvedValue([])
    vi.mocked(api.getEvents).mockResolvedValue([])
    vi.mocked(api.getRun).mockResolvedValue(run)
    vi.mocked(workflowApi.config).mockResolvedValue({ enabled: true })
    vi.mocked(workflowApi.list).mockResolvedValue([])
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    HTMLElement.prototype.scrollIntoView = originalScroll
  })

  it('retains left/right messages, inspector controls and multi-agent mode', async () => {
    const { container } = await mount()
    expect(container.querySelector('.message.user')).toHaveTextContent('检查实现和测试')
    expect(container.querySelector('.message.assistant')).toHaveTextContent('尚未执行测试')
    fireEvent.click(screen.getByRole('button', { name: '打开任务详情' }))
    expect(container.firstChild).toHaveClass('show-inspector')
    expect(screen.getByText('test-model')).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: '关闭任务详情' })[0])
    expect(container.firstChild).not.toHaveClass('show-inspector')
    fireEvent.click(screen.getByRole('tab', { name: '多智能体' }))
    expect(await screen.findByLabelText('多智能体需求')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: '对话' }))
    expect(screen.getByLabelText('任务描述')).toBeInTheDocument()
    expect(api.sendMessage).not.toHaveBeenCalled()
  })

  it('creates a session without starting a model run', async () => {
    await mount()
    const created = { ...session, id: 'session-two', title: '新对话' }
    vi.mocked(api.createSession).mockResolvedValue(created)
    vi.mocked(api.listSessions).mockResolvedValue([created, session])
    vi.mocked(api.getSession).mockResolvedValue({ session: created, messages: [], runs: [] })
    fireEvent.click(screen.getByRole('button', { name: '新建任务' }))
    expect(await screen.findByText('从一句任务描述开始')).toBeInTheDocument()
    expect(api.getSession).toHaveBeenLastCalledWith('session-two')
    expect(api.sendMessage).not.toHaveBeenCalled()
  })

  it('handles a rejected session creation without an unhandled promise or model call', async () => {
    await mount()
    vi.mocked(api.createSession).mockRejectedValue(new Error('Account changed'))
    fireEvent.click(screen.getByRole('button', { name: '新建任务' }))
    expect(await screen.findByText('Account changed')).toBeInTheDocument()
    expect(api.sendMessage).not.toHaveBeenCalled()
  })

  it('handles a rejected session read from the sidebar', async () => {
    await mount()
    vi.mocked(api.getSession).mockRejectedValue(new Error('Session access denied'))
    fireEvent.click(screen.getByRole('button', { name: /relu_and_mul 10-11/ }))
    expect(await screen.findByText('Session access denied')).toBeInTheDocument()
  })

  it('pins and unpins via the session context menu', async () => {
    await mount()
    vi.mocked(api.listSessions).mockResolvedValue([{ ...session, pinned: true }])
    fireEvent.contextMenu(screen.getByRole('button', { name: /relu_and_mul 10-11/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: '置顶会话' }))
    expect(await screen.findByLabelText('已置顶')).toBeInTheDocument()
    expect(api.setSessionPinned).toHaveBeenCalledWith(session.id, true)
    fireEvent.contextMenu(screen.getByRole('button', { name: /relu_and_mul 10-11/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: '取消置顶' }))
    await waitFor(() => expect(api.setSessionPinned).toHaveBeenLastCalledWith(session.id, false))
  })

  it('only deletes a session after confirmation and loads the remaining conversation', async () => {
    await mount()
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    const openDelete = () => {
      fireEvent.contextMenu(screen.getByRole('button', { name: /relu_and_mul 10-11/ }))
      fireEvent.click(screen.getByRole('menuitem', { name: '删除会话' }))
    }
    openDelete()
    expect(api.deleteSession).not.toHaveBeenCalled()
    confirm.mockReturnValue(true)
    const other = { ...session, id: 'session-two', title: '其他任务' }
    vi.mocked(api.listSessions).mockResolvedValue([other])
    vi.mocked(api.getSession).mockResolvedValue(bundle(other))
    openDelete()
    await waitFor(() => expect(api.getSession).toHaveBeenLastCalledWith(other.id))
    expect(api.deleteSession).toHaveBeenCalledWith(session.id)
  })

  it('fills suggestions, sends, streams the answer and records completion', async () => {
    await mount()
    const submit = screen.getByRole('button', { name: '发送任务' })
    expect(submit).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: bootstrap.suggestions[0] }))
    expect(screen.getByLabelText('任务描述')).toHaveValue(bootstrap.suggestions[0])
    vi.mocked(api.sendMessage).mockResolvedValue({
      user_message: { ...bundle().messages[0], id: 'user-two' },
      assistant_message: null,
      run,
    })
    fireEvent.click(submit)
    await waitFor(() => expect(EventStream.instances).toHaveLength(1))
    const stream = EventStream.instances[0]
    expect(api.sendMessage).toHaveBeenCalledWith(session.id, bootstrap.suggestions[0], expect.any(String))
    expect(stream.url).toContain(`/api/runs/${run.id}/events/stream`)
    await act(() => stream.emit(event('assistant-delta', { text: '正在读取测试源码' })))
    expect(screen.getByLabelText('Agent 正在回复')).toHaveTextContent('正在读取测试源码')
    vi.mocked(api.getRun).mockResolvedValue({ ...run, status: 'completed' })
    await act(() => stream.emit(event('completed')))
    expect(stream.close).toHaveBeenCalled()
    expect(screen.queryByLabelText('Agent 正在回复')).not.toBeInTheDocument()
  })

  it('keeps proposal approval separate from applying code', async () => {
    const proposal = {
      proposal_id: 'proposal-one',
      operator: 'relu_and_mul',
      status: 'pending_approval',
      rationale: '修复边界条件',
    }
    vi.mocked(api.getSessionApprovals).mockResolvedValue([
      event('approval-required', { proposal_id: proposal.proposal_id, proposal_type: 'repair' }),
    ])
    vi.mocked(api.getProposal).mockResolvedValue(proposal)
    vi.mocked(api.decideProposal).mockResolvedValue({ ...proposal, status: 'approved' })
    await mount()
    fireEvent.click(await screen.findByRole('button', { name: '批准' }))
    expect(await screen.findByRole('button', { name: '继续应用' })).toBeInTheDocument()
    expect(api.decideProposal).toHaveBeenCalledWith('repair', proposal.proposal_id, true, 'ada-cl25')
    expect(api.sendMessage).not.toHaveBeenCalled()
    expect(api.decideValidationPlan).not.toHaveBeenCalled()
  })

  it('shows the exact validation command and can reject without execution', async () => {
    const plan = {
      run_id: 'validation-one',
      operator: 'relu_and_mul',
      status: 'planned',
      command: 'ssh sg2044 pytest test_relu_and_mul.py',
      execution_target: 'remote' as const,
      receipt_path: 'receipt.json',
      approval: { status: 'pending_approval' as const },
    }
    vi.mocked(api.getSessionApprovals).mockResolvedValue([
      event('approval-required', { validation_run_id: plan.run_id }),
    ])
    vi.mocked(api.getValidationPlan).mockResolvedValue(plan)
    vi.mocked(api.decideValidationPlan).mockResolvedValue({ ...plan, approval: { status: 'rejected' } })
    await mount()
    expect(await screen.findByText(plan.command)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '拒绝' }))
    await waitFor(() => expect(api.decideValidationPlan).toHaveBeenCalledWith(plan.run_id, false, 'ada-cl25'))
    expect(api.sendMessage).not.toHaveBeenCalled()
  })

  it('shows and dismisses connection errors without marking a run successful', async () => {
    vi.mocked(api.getSession).mockResolvedValue({ ...bundle(), runs: [run] })
    await mount()
    await waitFor(() => expect(EventStream.instances).toHaveLength(1))
    act(() => EventStream.instances[0].onerror?.())
    expect(await screen.findByRole('alert')).toHaveTextContent('连接已中断')
    expect(screen.queryByText('证据确认通过')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '关闭错误提示' }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(EventStream.instances[0].close).toHaveBeenCalled()
  })
})
