import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'
import { WorkflowPanel } from './WorkflowPanel'
import { workflowApi, type Workflow, type WorkflowParallelPlanView } from './workflow-api'

vi.mock('./workflow-api', () => ({
  workflowApi: { list: vi.fn(), get: vi.fn(), submit: vi.fn(), action: vi.fn(), decideParallelPlan: vi.fn() },
}))
vi.mock('./api', () => ({
  api: {
    decideProposal: vi.fn(),
    decideValidationPlan: vi.fn(),
    getParallelPlan: vi.fn(),
    decideParallelPlan: vi.fn(),
    cancelParallelPlan: vi.fn(),
    sendMessage: vi.fn(),
  },
}))

const plan: WorkflowParallelPlanView = {
  plan_id: 'parallel-one',
  session_id: 'session-one',
  workflow_id: 'workflow-one',
  digest: 'reviewed-digest',
  status: 'pending',
  reason: 'Independent read-only implementation and test reviews',
  context: 'Do not edit code or execute validation.',
  user_request: 'Review the square operator.',
  tasks: [
    {
      task_id: 'implementation',
      goal: 'Inspect operator source',
      route: {
        model: 'child-model',
        provider: 'child-provider',
        tier: 'strong',
        reason: 'code_or_review_role',
        timeout_seconds: 90,
        input_bytes: 512,
        policy_fingerprint: 'policy-one',
      },
    },
    { task_id: 'tests', goal: 'Inspect coverage' },
  ],
  model: 'parent-model',
  provider: 'parent-provider',
  max_parallel: 2,
  timeout_seconds: 120,
  max_tool_calls: 8,
  tools: ['read_file', 'search_code'],
  results: [],
}

const task = (extra: Partial<Workflow> = {}): Workflow => ({
  id: 'workflow-one',
  session_id: 'session-one',
  operator: 'square_new',
  kind: 'create',
  state: 'waiting_parallel_approval',
  revision: 7,
  repair_count: 0,
  max_repairs: 3,
  recovery_action: '',
  request: { requirement: 'Review the square operator.' },
  result: {},
  steps: [],
  parallel_plan: plan,
  ...extra,
})

function mount(value = task(), enabled = true) {
  vi.mocked(workflowApi.list).mockResolvedValue([value])
  vi.mocked(workflowApi.get).mockResolvedValue(value)
  return render(<WorkflowPanel sessionId="session-one" enabled={enabled} />)
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => {
    resolve = done
  })
  return { promise, resolve }
}

function expectNoOtherActions() {
  expect(workflowApi.action).not.toHaveBeenCalled()
  expect(workflowApi.submit).not.toHaveBeenCalled()
  for (const method of Object.values(api)) expect(method).not.toHaveBeenCalled()
}

beforeEach(() => vi.resetAllMocks())
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('optional workflow parallel approval', () => {
  it('puts pending decisions before long requirements, history and plan details without auto-approving', async () => {
    mount(task({ request: { requirement: 'Long operator specification.\n'.repeat(200) } }))
    const region = await screen.findByRole('region', { name: '工作流并行计划' })
    const requirement = screen.getByText('原始需求').closest('details')!
    expect(requirement.open).toBe(false)
    expect(region.compareDocumentPosition(requirement) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    const approve = within(region).getByRole('button', { name: '批准只读并行' })
    expect(approve.compareDocumentPosition(within(region).getByText(plan.reason)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(approve).toBeEnabled()
    expect(within(region).getByRole('button', { name: '不并行，继续处理' })).toBeEnabled()
    fireEvent.click(screen.getByText('原始需求'))
    expect(requirement.open).toBe(true)
    expect(requirement).toHaveTextContent('Long operator specification.')
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
    expectNoOtherActions()
  })

  it('shows the reviewed scope, real child routes and bounded read-only limits without taking action', async () => {
    mount()
    const region = await screen.findByRole('region', { name: '工作流并行计划' })
    expect(within(region).getByText('等待并行审批')).toBeVisible()
    expect(within(region).getByText(plan.reason)).toBeVisible()
    const children = within(region).getAllByRole('listitem')
    expect(children[0]).toHaveTextContent('Inspect operator source')
    expect(children[0]).toHaveTextContent('模型：child-model（强力）')
    expect(children[0]).toHaveTextContent('提供方：child-provider')
    expect(children[0]).toHaveTextContent('超时：90 秒')
    expect(children[1]).toHaveTextContent('模型：parent-model（未提供档位）')
    expect(children[1]).toHaveTextContent('超时：120 秒')
    expect(within(region).getByText('默认模型').parentElement).toHaveTextContent(plan.model)
    expect(within(region).queryByText('父级模型（旧版默认）')).not.toBeInTheDocument()
    expect(within(region).getByText('最大并发').parentElement).toHaveTextContent('2')
    expect(within(region).getByText('工具调用上限').parentElement).toHaveTextContent('8')
    expect(within(region).getByText('只读工具范围').parentElement).toHaveTextContent('read_file, search_code')
    fireEvent.click(within(region).getByText('委派内容与背景'))
    expect(within(region).getByText(plan.context)).toBeVisible()
    expect(within(region).getByText(plan.user_request)).toBeVisible()
    expect(screen.queryByLabelText('审查人')).not.toBeInTheDocument()
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
    expectNoOtherActions()
  })

  it.each([true, false])(
    'records approved=%s once, disables controls pending HTTP, and uses the returned workflow',
    async approved => {
      mount()
      const decision = deferred<Workflow>()
      vi.mocked(workflowApi.decideParallelPlan).mockReturnValue(decision.promise)
      const button = await screen.findByRole('button', {
        name: approved ? '批准只读并行' : '不并行，继续处理',
      })
      fireEvent.click(button)
      fireEvent.click(button)
      expect(screen.getByRole('button', { name: '批准只读并行' })).toBeDisabled()
      expect(screen.getByRole('button', { name: '不并行，继续处理' })).toBeDisabled()
      expect(screen.getByRole('button', { name: '刷新工作流并行计划' })).toBeDisabled()
      expect(screen.getByLabelText('已保存任务')).toBeDisabled()
      expect(workflowApi.decideParallelPlan).toHaveBeenCalledExactlyOnceWith(task(), approved)

      await act(async () =>
        decision.resolve(
          task({
            revision: 8,
            state: 'preparing',
            parallel_plan: { ...plan, status: approved ? 'approved' : 'rejected' },
          }),
        ),
      )
      expect(screen.getByText(approved ? '已批准并行，等待执行' : '已选择不并行，继续顺序处理')).toBeVisible()
      expect(screen.queryByRole('button', { name: '批准只读并行' })).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: '不并行，继续处理' })).not.toBeInTheDocument()
      expect(workflowApi.get).toHaveBeenCalledTimes(1)
      expectNoOtherActions()
    },
  )

  it.each([
    [true, 'stale workflow revision'],
    [false, 'stale workflow revision'],
    [true, 'digest mismatch'],
    [false, 'digest mismatch'],
  ] as const)(
    'requires refresh and explicit new consent after approved=%s fails: %s',
    async (approved, message) => {
      mount()
      vi.mocked(workflowApi.decideParallelPlan).mockRejectedValueOnce(new Error(message))
      fireEvent.click(
        await screen.findByRole('button', { name: approved ? '批准只读并行' : '不并行，继续处理' }),
      )
      expect(await screen.findByRole('alert')).toHaveTextContent(message)
      expect(screen.getByRole('button', { name: '批准只读并行' })).toBeDisabled()
      expect(screen.getByRole('button', { name: '不并行，继续处理' })).toBeDisabled()
      const refreshed = task({
        revision: 10,
        parallel_plan: { ...plan, reason: 'Revised scope', digest: 'new-digest' },
      })
      vi.mocked(workflowApi.get).mockResolvedValue(refreshed)
      fireEvent.click(screen.getByRole('button', { name: '刷新工作流并行计划' }))
      expect(await screen.findByText('Revised scope')).toBeVisible()
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
      expect(workflowApi.decideParallelPlan).toHaveBeenCalledTimes(1)
      vi.mocked(workflowApi.decideParallelPlan).mockResolvedValue({
        ...refreshed,
        revision: 11,
        state: 'preparing',
        parallel_plan: { ...refreshed.parallel_plan!, status: 'rejected' },
      })
      fireEvent.click(screen.getByRole('button', { name: '不并行，继续处理' }))
      await screen.findByText('已选择不并行，继续顺序处理')
      expect(workflowApi.decideParallelPlan).toHaveBeenLastCalledWith(refreshed, false)
      expectNoOtherActions()
    },
  )

  it('keeps decisions disabled if the required refresh fails', async () => {
    mount()
    vi.mocked(workflowApi.decideParallelPlan).mockRejectedValueOnce(new Error('digest mismatch'))
    fireEvent.click(await screen.findByRole('button', { name: '批准只读并行' }))
    await screen.findByRole('alert')
    vi.mocked(workflowApi.get).mockRejectedValue(new Error('offline'))
    fireEvent.click(screen.getByRole('button', { name: '刷新工作流并行计划' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('offline'))
    expect(screen.getByRole('button', { name: '批准只读并行' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '不并行，继续处理' })).toBeDisabled()
    expect(workflowApi.decideParallelPlan).toHaveBeenCalledTimes(1)
  })

  it('coalesces repeated refresh clicks and keeps decisions locked until the latest view is read', async () => {
    vi.useFakeTimers()
    mount()
    await act(async () => {})
    const refreshed = task({
      revision: 8,
      parallel_plan: { ...plan, reason: 'Refreshed scope', digest: 'new-digest' },
    })
    const refresh = deferred<Workflow>()
    vi.mocked(workflowApi.get).mockReturnValueOnce(refresh.promise)
    const button = screen.getByRole('button', { name: '刷新工作流并行计划' })
    fireEvent.click(button)
    fireEvent.click(button)
    expect(button).toBeDisabled()
    expect(screen.getByRole('button', { name: '批准只读并行' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '不并行，继续处理' })).toBeDisabled()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(workflowApi.get).toHaveBeenCalledTimes(2)
    expect(workflowApi.list).toHaveBeenCalledTimes(1)

    await act(async () => refresh.resolve(refreshed))
    expect(screen.getByText('Refreshed scope')).toBeVisible()
    expect(button).toBeEnabled()
    expect(screen.getByRole('button', { name: '批准只读并行' })).toBeEnabled()
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()

    const second = deferred<Workflow>()
    vi.mocked(workflowApi.get).mockReturnValueOnce(second.promise)
    fireEvent.click(button)
    fireEvent.click(button)
    expect(workflowApi.get).toHaveBeenCalledTimes(3)
    await act(async () => second.resolve({ ...refreshed, revision: 9 }))
    expect(button).toBeEnabled()
    expectNoOtherActions()
  })

  it.each(['pending', 'approved', 'running'] as const)(
    'shows workflow fallback over the underlying %s plan without offering parallel decisions',
    async status => {
      mount(
        task({
          parallel_plan: {
            ...plan,
            status,
            workflow_status: 'fallback',
            fallback_reason: '计划已过期或模型配置已变更',
          },
        }),
      )
      const region = await screen.findByRole('region', { name: '工作流并行计划' })
      expect(within(region).getByText('已回退为顺序处理')).toBeVisible()
      expect(within(region).getByRole('status')).toHaveTextContent('并行回退原因：计划已过期或模型配置已变更')
      expect(within(region).queryByText('等待并行审批')).not.toBeInTheDocument()
      expect(within(region).queryByText('已批准并行，等待执行')).not.toBeInTheDocument()
      expect(within(region).queryByText('并行运行中')).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: '批准只读并行' })).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: '不并行，继续处理' })).not.toBeInTheDocument()
      expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
      expectNoOtherActions()
    },
  )

  it('shows fallback returned directly by approval instead of claiming it is queued', async () => {
    mount()
    vi.mocked(workflowApi.decideParallelPlan).mockResolvedValue(
      task({
        revision: 8,
        state: 'preparing',
        parallel_plan: {
          ...plan,
          status: 'approved',
          workflow_status: 'fallback',
          fallback_reason: 'parallel configuration changed',
        },
      }),
    )
    fireEvent.click(await screen.findByRole('button', { name: '批准只读并行' }))
    expect(await screen.findByText('已回退为顺序处理')).toBeVisible()
    expect(screen.getByRole('status')).toHaveTextContent('parallel configuration changed')
    expect(screen.queryByText('已批准并行，等待执行')).not.toBeInTheDocument()
    expect(workflowApi.decideParallelPlan).toHaveBeenCalledTimes(1)
    expectNoOtherActions()
  })

  it('does not invent a fallback reason when the server omits it', async () => {
    mount(task({ parallel_plan: { ...plan, status: 'running', workflow_status: 'fallback' } }))
    expect(await screen.findByText('已回退为顺序处理')).toBeVisible()
    expect(screen.getByRole('status')).toHaveTextContent('并行回退原因：未提供原因')
    expect(screen.queryByText('并行运行中')).not.toBeInTheDocument()
  })

  it('accepts legacy views without an optional workflow binding', async () => {
    mount(task({ parallel_plan: { ...plan, workflow_id: undefined } }))
    expect(await screen.findByRole('button', { name: '批准只读并行' })).toBeEnabled()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it.each([{ id: 'other-workflow' }, { session_id: 'other-session' }, { revision: 6 }])(
    'does not replace the reviewed workflow with an inconsistent response: %j',
    async extra => {
      mount()
      vi.mocked(workflowApi.decideParallelPlan).mockResolvedValue(task(extra))
      fireEvent.click(await screen.findByRole('button', { name: '批准只读并行' }))
      expect(await screen.findByRole('alert')).toHaveTextContent('任务标识或版本不匹配')
      expect(screen.getByRole('button', { name: '批准只读并行' })).toBeDisabled()
      expect(screen.getByText('workflow-one')).toBeVisible()
      expect(workflowApi.decideParallelPlan).toHaveBeenCalledTimes(1)
      expectNoOtherActions()
    },
  )

  it('keeps code approval separate from completed parallel reviews', async () => {
    mount(
      task({
        state: 'waiting_approval',
        parallel_plan: { ...plan, status: 'completed' },
        proposal: {
          proposal_id: 'proposal-one',
          operator: 'square_new',
          status: 'pending_approval',
          rationale: 'Review code',
          diff: '+ return x*x',
        },
      }),
    )
    expect(await screen.findByRole('button', { name: '批准并应用' })).toBeDisabled()
    fireEvent.change(screen.getByLabelText('审查人'), { target: { value: 'reviewer' } })
    vi.mocked(workflowApi.action).mockResolvedValue(
      task({ revision: 8, state: 'waiting_validation_approval' }),
    )
    fireEvent.click(screen.getByRole('button', { name: '批准并应用' }))
    await waitFor(() => expect(workflowApi.action).toHaveBeenCalledTimes(2))
    expect(api.decideProposal).toHaveBeenCalledExactlyOnceWith(
      'development',
      'proposal-one',
      true,
      'reviewer',
    )
    expect(vi.mocked(workflowApi.action).mock.calls.map(call => call[1])).toEqual([
      'apply',
      'plan-validation',
    ])
    expect(api.decideValidationPlan).not.toHaveBeenCalled()
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
  })

  it('does not approve validation as a side effect of a historical parallel decision', async () => {
    mount(
      task({
        state: 'waiting_validation_approval',
        parallel_plan: { ...plan, status: 'rejected' },
        validation_plan: {
          run_id: 'validation-one',
          operator: 'square_new',
          status: 'planned',
          command: 'pytest fixed_test.py',
          execution_target: 'remote',
          receipt_path: 'receipt.json',
          approval: { status: 'pending_approval' },
        },
      }),
    )
    expect(await screen.findByRole('button', { name: '批准并执行' })).toBeDisabled()
    expect(api.decideValidationPlan).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('审查人'), { target: { value: 'reviewer' } })
    fireEvent.click(screen.getByRole('button', { name: '批准并执行' }))
    await waitFor(() => expect(workflowApi.action).toHaveBeenCalledTimes(1))
    expect(api.decideValidationPlan).toHaveBeenCalledExactlyOnceWith('validation-one', true, 'reviewer')
    expect(workflowApi.action).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'workflow-one' }),
      'enqueue-validation',
    )
    expect(api.decideProposal).not.toHaveBeenCalled()
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
  })

  it.each(['approved', 'running', 'completed', 'partial', 'rejected', 'cancelled', 'unknown'] as const)(
    'shows %s and available results without sending any continuation',
    async status => {
      mount(
        task({
          state: 'preparing',
          parallel_plan: { ...plan, status, results: [{ task_id: 'tests', summary: 'Coverage inspected' }] },
        }),
      )
      const region = await screen.findByRole('region', { name: '工作流并行计划' })
      const labels = {
        approved: '已批准并行，等待执行',
        running: '并行运行中',
        completed: '并行子任务已完成',
        partial: '并行子任务部分完成',
        rejected: '已选择不并行，继续顺序处理',
        cancelled: '并行已取消',
        unknown: '并行状态未知',
      }
      expect(within(region).getByText(labels[status])).toBeVisible()
      fireEvent.click(within(region).getByText('已返回 1 条子任务结果'))
      expect(within(region).getByText(/Coverage inspected/)).toBeVisible()
      expect(screen.queryByRole('button', { name: '批准只读并行' })).not.toBeInTheDocument()
      expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
      expectNoOtherActions()
    },
  )

  it.each([null, undefined])(
    'keeps workflows with parallel_plan=%s on the original role path',
    async parallel_plan => {
      mount(
        task({
          state: 'preparing',
          parallel_plan,
          steps: [
            {
              id: 'semantic',
              role: 'semantic',
              state: 'preparing',
              status: 'completed',
              job_id: 'job-one',
              result: { summary: 'Requirements checked' },
            },
          ],
        }),
      )
      expect(await screen.findByText('语义 Agent · 确认需求')).toBeVisible()
      expect(screen.queryByRole('region', { name: '工作流并行计划' })).not.toBeInTheDocument()
      expectNoOtherActions()
    },
  )

  it('requires a loaded plan before offering decisions', async () => {
    mount(task({ parallel_plan: null }))
    expect(await screen.findByRole('alert')).toHaveTextContent('并行计划尚未加载')
    expect(screen.queryByRole('button', { name: '批准只读并行' })).not.toBeInTheDocument()
    vi.mocked(workflowApi.get).mockResolvedValue(task())
    fireEvent.click(screen.getByRole('button', { name: '刷新工作流并行计划' }))
    expect(await screen.findByRole('button', { name: '批准只读并行' })).toBeEnabled()
  })

  it.each([
    { ...plan, digest: '' },
    { ...plan, plan_id: '' },
    { ...plan, session_id: 'other-session' },
    { ...plan, workflow_id: 'other-workflow' },
  ])('blocks a plan with invalid approval identity: %j', async parallel_plan => {
    mount(task({ parallel_plan }))
    expect(await screen.findByRole('alert')).toHaveTextContent('并行计划标识或摘要不匹配')
    expect(screen.getByRole('button', { name: '批准只读并行' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '不并行，继续处理' })).toBeDisabled()
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
  })

  it.each([false, true])(
    'honors disabled installations and existing artifact blocks (enabled=%s)',
    async enabled => {
      mount(task(enabled ? { artifact_error: 'Missing evidence' } : {}), enabled)
      expect(await screen.findByRole('button', { name: '批准只读并行' })).toBeDisabled()
      expect(screen.getByRole('button', { name: '不并行，继续处理' })).toBeDisabled()
      expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
    },
  )

  it('does not let an old poll restore an approved plan after repeated refresh returns fallback', async () => {
    vi.useFakeTimers()
    const approved = task({ state: 'preparing', parallel_plan: { ...plan, status: 'approved' } })
    mount(approved)
    await act(async () => {})
    const oldPoll = deferred<Workflow>()
    const refresh = deferred<Workflow>()
    vi.mocked(workflowApi.get).mockReturnValueOnce(oldPoll.promise).mockReturnValueOnce(refresh.promise)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    const button = screen.getByRole('button', { name: '刷新工作流并行计划' })
    fireEvent.click(button)
    fireEvent.click(button)
    expect(workflowApi.get).toHaveBeenCalledTimes(3)
    await act(async () =>
      refresh.resolve(
        task({
          revision: 8,
          state: 'preparing',
          parallel_plan: {
            ...plan,
            status: 'approved',
            workflow_status: 'fallback',
            fallback_reason: 'plan expired',
          },
        }),
      ),
    )
    await act(async () => oldPoll.resolve(approved))
    expect(screen.getByText('已回退为顺序处理')).toBeVisible()
    expect(screen.getByRole('status')).toHaveTextContent('plan expired')
    expect(screen.queryByText('已批准并行，等待执行')).not.toBeInTheDocument()
    expect(button).toBeEnabled()
    expect(workflowApi.decideParallelPlan).not.toHaveBeenCalled()
    expectNoOtherActions()
  })

  it('discards an old in-flight poll after a decision and polls the later completed outcome', async () => {
    vi.useFakeTimers()
    mount()
    await act(async () => {})
    const oldPoll = deferred<Workflow>()
    vi.mocked(workflowApi.get).mockReturnValueOnce(oldPoll.promise)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(workflowApi.get).toHaveBeenCalledTimes(2)
    vi.mocked(workflowApi.decideParallelPlan).mockResolvedValue(
      task({ revision: 8, state: 'preparing', parallel_plan: { ...plan, status: 'running' } }),
    )
    await act(async () => fireEvent.click(screen.getByRole('button', { name: '批准只读并行' })))
    await act(async () => oldPoll.resolve(task()))
    expect(screen.getByText('并行运行中')).toBeVisible()
    expect(screen.queryByRole('button', { name: '批准只读并行' })).not.toBeInTheDocument()
    vi.mocked(workflowApi.get).mockResolvedValue(
      task({ revision: 9, state: 'preparing', parallel_plan: { ...plan, status: 'completed' } }),
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(screen.getByText('并行子任务已完成')).toBeVisible()
    expect(workflowApi.decideParallelPlan).toHaveBeenCalledTimes(1)
    expectNoOtherActions()
  })
})
