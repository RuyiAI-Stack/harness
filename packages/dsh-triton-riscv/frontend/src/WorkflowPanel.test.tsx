import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'
import { WorkflowPanel } from './WorkflowPanel'
import { workflowApi, type Workflow } from './workflow-api'

vi.mock('./workflow-api', () => ({ workflowApi: { list: vi.fn(), get: vi.fn(), submit: vi.fn(), action: vi.fn() } }))
vi.mock('./api', () => ({ api: { decideProposal: vi.fn(), decideValidationPlan: vi.fn() } }))

const task = (extra: Partial<Workflow> = {}): Workflow => ({
  id: 'task-one',
  session_id: 'session-one',
  operator: 'square_new',
  kind: 'create',
  state: 'waiting_approval',
  revision: 8,
  repair_count: 0,
  max_repairs: 3,
  recovery_action: '',
  request: { requirement: 'Return x squared with fixed float32 tests.' },
  result: {},
  steps: [],
  acceptance: { formula: 'y=x*x', dtype: 'float32' },
  proposal: {
    proposal_id: 'proposal-one',
    operator: 'square_new',
    status: 'pending_approval',
    rationale: 'Review candidate',
    diff: '+ return x*x',
  },
  ...extra,
})

function mount(value: Workflow | null = task(), enabled = true) {
  vi.mocked(workflowApi.list).mockResolvedValue(value ? [value] : [])
  vi.mocked(workflowApi.get).mockResolvedValue(value || task())
  return render(<WorkflowPanel sessionId="session-one" enabled={enabled} />)
}

describe('multi-agent workbench', () => {
  beforeEach(() => vi.resetAllMocks())
  afterEach(() => cleanup())

  it('shows frozen requirements and needs an explicit reviewer before writing', async () => {
    mount()
    expect(await screen.findByText('算子要求与验收条件')).toBeInTheDocument()
    expect(screen.getByText(/"formula": "y=x\*x"/)).toBeInTheDocument()
    const approve = screen.getByRole('button', { name: '批准并应用' })
    expect(approve).toBeDisabled()
    fireEvent.change(screen.getByLabelText('审查人'), { target: { value: 'human-reviewer' } })
    vi.mocked(workflowApi.action).mockResolvedValue(task({ state: 'waiting_validation_approval', revision: 9 }))
    fireEvent.click(approve)
    await waitFor(() => expect(workflowApi.action).toHaveBeenCalledTimes(2))
    expect(api.decideProposal).toHaveBeenCalledWith('development', 'proposal-one', true, 'human-reviewer')
    expect(vi.mocked(workflowApi.action).mock.calls.map(c => c[1])).toEqual(['apply', 'plan-validation'])
    expect(api.decideValidationPlan).not.toHaveBeenCalled()
  })

  it('requires separate validation approval and queues only the bound plan', async () => {
    mount(
      task({
        state: 'waiting_validation_approval',
        validation_plan: {
          run_id: 'plan-one',
          operator: 'square_new',
          status: 'planned',
          command: 'pytest fixed_test.py',
          execution_target: 'remote',
          receipt_path: 'receipt.json',
          approval: { status: 'pending_approval' },
        },
      }),
    )
    expect(await screen.findByText('pytest fixed_test.py')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('审查人'), { target: { value: 'reviewer' } })
    fireEvent.click(screen.getByRole('button', { name: '批准并执行' }))
    await waitFor(() =>
      expect(workflowApi.action).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'task-one' }),
        'enqueue-validation',
      ),
    )
    expect(api.decideValidationPlan).toHaveBeenCalledWith('plan-one', true, 'reviewer')
    expect(api.decideProposal).not.toHaveBeenCalled()
  })

  it('rejects a proposal without applying or running tests', async () => {
    mount()
    await screen.findByText('代码方案 · pending_approval')
    fireEvent.change(screen.getByLabelText('审查人'), { target: { value: 'reviewer' } })
    fireEvent.click(screen.getByRole('button', { name: '拒绝' }))
    await waitFor(() =>
      expect(api.decideProposal).toHaveBeenCalledWith('development', 'proposal-one', false, 'reviewer'),
    )
    expect(workflowApi.action).not.toHaveBeenCalled()
  })

  it('retains the submission id across a network failure', async () => {
    mount(null)
    fireEvent.change(screen.getByLabelText('算子名称'), { target: { value: 'square_new' } })
    fireEvent.change(screen.getByLabelText('多智能体需求'), {
      target: { value: 'Compute x squared with float32 and six acceptance cases.' },
    })
    vi.mocked(workflowApi.submit)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({ payload: { workflow_id: 'task-one' } })
    fireEvent.click(screen.getByRole('button', { name: '发送多智能体任务' }))
    await screen.findByText('offline')
    fireEvent.click(screen.getByRole('button', { name: '发送多智能体任务' }))
    await waitFor(() => expect(workflowApi.submit).toHaveBeenCalledTimes(2))
    const calls = vi.mocked(workflowApi.submit).mock.calls
    expect(calls[0][0]).toEqual(calls[1][0])
    expect(calls[0][0].request).toMatchObject({
      session_id: 'session-one',
      auto_publish: true,
      auto_repair: true,
      max_repairs: 3,
    })
    expect(calls[0][0].request_id.length).toBeGreaterThan(10)
    expect(api.decideProposal).not.toHaveBeenCalled()
  })

  it('blocks review buttons while artifacts cannot be verified', async () => {
    mount(task({ artifact_error: 'Missing evidence' }))
    await screen.findByText(/执行结果尚未核清/)
    fireEvent.change(screen.getByLabelText('审查人'), { target: { value: 'reviewer' } })
    expect(screen.getByRole('button', { name: '批准并应用' })).toBeDisabled()
    expect(workflowApi.action).not.toHaveBeenCalled()
  })

  it('keeps the same submission id if acknowledgement succeeds but detail loading fails', async () => {
    mount(null)
    fireEvent.change(screen.getByLabelText('算子名称'), { target: { value: 'square_new' } })
    fireEvent.change(screen.getByLabelText('多智能体需求'), {
      target: { value: 'Compute x squared under the supplied contract.' },
    })
    vi.mocked(workflowApi.submit).mockResolvedValue({ payload: { workflow_id: 'task-one' } })
    vi.mocked(workflowApi.get).mockRejectedValue(new Error('detail unavailable'))
    fireEvent.click(screen.getByRole('button', { name: '发送多智能体任务' }))
    await screen.findByText('detail unavailable')
    vi.mocked(workflowApi.get).mockResolvedValue(task())
    fireEvent.click(screen.getByRole('button', { name: '发送多智能体任务' }))
    await waitFor(() => expect(workflowApi.submit).toHaveBeenCalledTimes(2))
    expect(vi.mocked(workflowApi.submit).mock.calls[0][0].request_id).toBe(
      vi.mocked(workflowApi.submit).mock.calls[1][0].request_id,
    )
  })

  it('navigates to a repair child without hiding the failed attempt', async () => {
    mount(task({ state: 'failed', result: { lifecycle: { child_workflow_id: 'task-child' } } }))
    fireEvent.click(await screen.findByRole('button', { name: '查看自动修复任务' }))
    await waitFor(() => expect(workflowApi.get).toHaveBeenCalledWith('task-child'))
    expect(workflowApi.action).not.toHaveBeenCalled()
  })

  it('requires failure evidence for repair and keeps disabled installations read-only', async () => {
    mount(null, false)
    fireEvent.change(screen.getByLabelText('任务类型'), { target: { value: 'repair' } })
    expect(screen.getByLabelText('失败回执 ID')).toBeRequired()
    expect(screen.getByRole('button', { name: '发送多智能体任务' })).toBeDisabled()
    expect(screen.getByText('多智能体未启用')).toBeInTheDocument()
  })
})
