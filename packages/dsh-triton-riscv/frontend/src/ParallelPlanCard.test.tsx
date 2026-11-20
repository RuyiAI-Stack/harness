import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'
import { ParallelPlanCard } from './ParallelPlanCard'
import type { ModelRoute, ParallelPlan } from './types'

vi.mock('./api', () => ({
  api: { getParallelPlan: vi.fn(), decideParallelPlan: vi.fn(), cancelParallelPlan: vi.fn() },
}))

const plan: ParallelPlan = {
  plan_id: 'parallel-one',
  session_id: 'session-one',
  reason: 'Review implementation and tests independently',
  context: 'The operator has a boundary-case regression.\nDo not edit source or tests.',
  user_request: 'Inspect implementation and test coverage before proposing a repair.',
  tasks: [
    { task_id: 'implementation', goal: 'Read the operator implementation' },
    { task_id: 'tests', goal: 'Review existing test coverage' },
  ],
  status: 'pending',
  model: 'review-model',
  provider: 'test-provider',
  base_url: 'https://provider.example.test/v1',
  max_tool_calls: 8,
  max_parallel: 2,
  timeout_seconds: 120,
  tools: ['read_file', 'search_code'],
  results: [],
  digest: 'digest-reviewed',
}

const fastRoute: ModelRoute = {
  tier: 'fast',
  model: 'fast-child-model',
  provider: 'child-provider',
  reason: 'Small read-only review',
  timeout_seconds: 45,
  input_bytes: 512,
  policy_fingerprint: 'route-policy-one',
}

const mount = (onContinue = vi.fn().mockResolvedValue(undefined), disabled = false) => ({
  onContinue,
  ...render(
    <ParallelPlanCard sessionId={plan.session_id} planId={plan.plan_id} onContinue={onContinue} disabled={disabled} />,
  ),
})

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.getParallelPlan).mockResolvedValue(plan)
})
afterEach(cleanup)

describe('ParallelPlanCard', () => {
  it('shows the actual plan and read-only scope without approving or executing on mount', async () => {
    const { onContinue } = mount()
    expect(await screen.findByText(plan.reason)).toBeInTheDocument()
    for (const task of plan.tasks) expect(screen.getByText(task.goal)).toBeInTheDocument()
    expect(screen.getByText(plan.model)).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('120 秒')).toBeInTheDocument()
    expect(screen.getByText('只读工具范围')).toBeInTheDocument()
    expect(screen.getByText('read_file, search_code')).toBeInTheDocument()
    expect(api.decideParallelPlan).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('falls back to plan model and timeout for old plans without inventing route metadata', async () => {
    const { onContinue } = mount()
    await screen.findByRole('button', { name: '批准并继续' })
    for (const task of screen.getAllByRole('listitem')) {
      expect(task).toHaveTextContent(`模型：${plan.model}（未提供档位）`)
      expect(task).toHaveTextContent(`超时：${plan.timeout_seconds} 秒`)
      expect(task).toHaveTextContent('选择原因：旧计划未提供路由信息，沿用计划配置')
    }
    expect(screen.getByText('默认模型').parentElement).toHaveTextContent(plan.model)
    expect(api.decideParallelPlan).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('shows authoritative per-child routes and Chinese tiers before approving the original digest', async () => {
    const routes: ModelRoute[] = [
      fastRoute,
      {
        ...fastRoute,
        tier: 'strong',
        model: 'strong-child-model',
        reason: 'Complex implementation review',
        timeout_seconds: 240,
      },
      {
        ...fastRoute,
        tier: 'default',
        model: 'default-child-model',
        reason: 'Routing disabled',
        timeout_seconds: 90,
      },
    ]
    const routedPlan: ParallelPlan = {
      ...plan,
      tasks: routes.map((route, index) => ({ task_id: `child-${index}`, goal: `Review ${index}`, route })),
      digest: 'digest-with-routes',
    }
    vi.mocked(api.getParallelPlan).mockResolvedValue(routedPlan)
    vi.mocked(api.decideParallelPlan).mockResolvedValue({ ...routedPlan, status: 'approved' })
    const { onContinue } = mount()
    const approve = await screen.findByRole('button', { name: '批准并继续' })
    const tasks = screen.getAllByRole('listitem')
    const tiers = ['快速', '强力', '默认']
    routes.forEach((route, index) => {
      expect(tasks[index]).toHaveTextContent(`模型：${route.model}（${tiers[index]}）`)
      expect(tasks[index]).toHaveTextContent(`提供方：${route.provider}`)
      expect(tasks[index]).toHaveTextContent(`超时：${route.timeout_seconds} 秒`)
      expect(tasks[index]).toHaveTextContent(`选择原因：${route.reason}`)
      expect(tasks[index]).not.toHaveTextContent(plan.model)
      expect(tasks[index]).not.toHaveTextContent(plan.provider!)
      expect(tasks[index]).not.toHaveTextContent(`超时：${plan.timeout_seconds} 秒`)
      expect(within(tasks[index]).getByText(/选择原因/)).toBeVisible()
    })
    expect(api.decideParallelPlan).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
    fireEvent.click(approve)
    await waitFor(() => expect(onContinue).toHaveBeenCalledExactlyOnceWith(plan.plan_id))
    expect(api.decideParallelPlan).toHaveBeenCalledExactlyOnceWith(
      plan.session_id,
      plan.plan_id,
      true,
      routedPlan.digest,
    )
  })

  it('falls back only for the legacy child in a mixed plan', async () => {
    vi.mocked(api.getParallelPlan).mockResolvedValue({
      ...plan,
      tasks: [{ ...plan.tasks[0], route: fastRoute }, plan.tasks[1]],
    })
    mount()
    await screen.findByRole('button', { name: '批准并继续' })
    const [routed, legacy] = screen.getAllByRole('listitem')
    expect(routed).toHaveTextContent(fastRoute.model)
    expect(routed).not.toHaveTextContent('旧计划未提供路由信息')
    expect(legacy).toHaveTextContent(plan.model)
    expect(legacy).toHaveTextContent('旧计划未提供路由信息')
    expect(legacy).not.toHaveTextContent(fastRoute.model)
  })

  it('approves the reviewed digest once, then requests continuation of the exact plan', async () => {
    vi.mocked(api.decideParallelPlan).mockResolvedValue({ ...plan, status: 'approved' })
    const { onContinue } = mount()
    const approve = await screen.findByRole('button', { name: '批准并继续' })
    fireEvent.click(approve)
    fireEvent.click(approve)
    await waitFor(() => expect(onContinue).toHaveBeenCalledExactlyOnceWith(plan.plan_id))
    expect(api.decideParallelPlan).toHaveBeenCalledExactlyOnceWith(plan.session_id, plan.plan_id, true, plan.digest)
    expect(vi.mocked(api.decideParallelPlan).mock.invocationCallOrder[0]).toBeLessThan(
      onContinue.mock.invocationCallOrder[0],
    )
    expect(await screen.findByText('继续执行消息已发送，等待执行状态。')).toBeInTheDocument()
    expect(screen.queryByText('已完成')).not.toBeInTheDocument()
  })

  it('expands the exact delegated background and original request without approving', async () => {
    const { onContinue, container } = mount()
    await screen.findByText(plan.reason)
    const details = container.querySelector('.parallel-context')
    expect(details).not.toHaveAttribute('open')
    fireEvent.click(screen.getByText('委派内容与背景'))
    expect(details).toHaveAttribute('open')
    expect(screen.getByText(plan.user_request)).toBeVisible()
    expect(container.querySelectorAll('.parallel-context pre')[1].textContent).toBe(plan.context)
    expect(screen.getByText(plan.provider!)).toBeVisible()
    expect(screen.getByText(plan.base_url!)).toBeVisible()
    expect(screen.getByText('8')).toBeVisible()
    expect(api.decideParallelPlan).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('rejects without continuing or cancelling', async () => {
    vi.mocked(api.decideParallelPlan).mockResolvedValue({ ...plan, status: 'rejected' })
    const { onContinue } = mount()
    fireEvent.click(await screen.findByRole('button', { name: '拒绝' }))
    expect(await screen.findByText('已拒绝')).toBeInTheDocument()
    expect(api.decideParallelPlan).toHaveBeenCalledWith(plan.session_id, plan.plan_id, false, plan.digest)
    expect(onContinue).not.toHaveBeenCalled()
    expect(api.cancelParallelPlan).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: '批准并继续' })).not.toBeInTheDocument()
  })

  it('requires a refresh and explicit new approval after a wrong-digest error', async () => {
    vi.mocked(api.decideParallelPlan).mockRejectedValueOnce(new Error('digest mismatch'))
    const { onContinue } = mount()
    fireEvent.click(await screen.findByRole('button', { name: '批准并继续' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('digest mismatch')
    expect(screen.queryByRole('button', { name: '批准并继续' })).not.toBeInTheDocument()
    expect(onContinue).not.toHaveBeenCalled()
    const updated = {
      ...plan,
      reason: 'Updated scope',
      context: 'New delegated background',
      user_request: 'Updated original request',
      digest: 'digest-new',
    }
    vi.mocked(api.getParallelPlan).mockResolvedValue(updated)
    fireEvent.click(screen.getByRole('button', { name: '刷新并行计划' }))
    expect(await screen.findByText(updated.reason)).toBeInTheDocument()
    fireEvent.click(screen.getByText('委派内容与背景'))
    expect(screen.getByText(updated.context)).toBeVisible()
    expect(screen.getByText(updated.user_request)).toBeVisible()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(api.decideParallelPlan).toHaveBeenCalledTimes(1)
    vi.mocked(api.decideParallelPlan).mockResolvedValue({ ...updated, status: 'approved' })
    fireEvent.click(screen.getByRole('button', { name: '批准并继续' }))
    await waitFor(() => expect(onContinue).toHaveBeenCalledTimes(1))
    expect(api.decideParallelPlan).toHaveBeenLastCalledWith(plan.session_id, plan.plan_id, true, updated.digest)
  })

  it.each([
    { ...plan, status: 'approved' as const, digest: 'unexpected-digest' },
    { ...plan, status: 'approved' as const, session_id: 'other-session' },
    { ...plan, status: 'approved' as const, plan_id: 'other-plan' },
    { ...plan, status: 'rejected' as const },
    { ...plan, status: 'running' as const },
  ])('does not continue an unexpected decision response: %j', async response => {
    vi.mocked(api.decideParallelPlan).mockResolvedValue(response)
    const { onContinue } = mount()
    fireEvent.click(await screen.findByRole('button', { name: '批准并继续' }))
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('shows read failures and allows a refresh without any automatic decision', async () => {
    vi.mocked(api.getParallelPlan).mockRejectedValueOnce(new Error('Plan unavailable'))
    const { onContinue } = mount()
    expect(await screen.findByRole('alert')).toHaveTextContent('Plan unavailable')
    expect(screen.queryByRole('button', { name: '批准并继续' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '刷新并行计划' }))
    await screen.findByRole('button', { name: '批准并继续' })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(api.decideParallelPlan).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('keeps approval after a continuation error and retries only the parent message', async () => {
    vi.mocked(api.decideParallelPlan).mockResolvedValue({ ...plan, status: 'approved' })
    const onContinue = vi.fn().mockRejectedValueOnce(new Error('Message not delivered')).mockResolvedValue(undefined)
    mount(onContinue)
    fireEvent.click(await screen.findByRole('button', { name: '批准并继续' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Message not delivered')
    expect(screen.queryByText('继续执行消息已发送，等待执行状态。')).not.toBeInTheDocument()
    vi.mocked(api.getParallelPlan).mockResolvedValue({ ...plan, status: 'approved' })
    fireEvent.click(screen.getByRole('button', { name: '继续已批准计划' }))
    expect(await screen.findByText('继续执行消息已发送，等待执行状态。')).toBeInTheDocument()
    expect(onContinue).toHaveBeenCalledTimes(2)
    expect(api.decideParallelPlan).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('never auto-continues an already approved plan and rechecks its state before continuing', async () => {
    vi.mocked(api.getParallelPlan)
      .mockResolvedValueOnce({ ...plan, status: 'approved' })
      .mockResolvedValue({ ...plan, status: 'cancelled' })
    const { onContinue } = mount()
    const button = await screen.findByRole('button', { name: '继续已批准计划' })
    expect(onContinue).not.toHaveBeenCalled()
    fireEvent.click(button)
    expect(await screen.findByText('已取消')).toBeInTheDocument()
    expect(onContinue).not.toHaveBeenCalled()
    expect(api.decideParallelPlan).not.toHaveBeenCalled()
  })

  it('cancels a running plan and displays only the returned state', async () => {
    vi.mocked(api.getParallelPlan).mockResolvedValue({ ...plan, status: 'running' })
    vi.mocked(api.cancelParallelPlan).mockResolvedValue({ ...plan, status: 'cancelled' })
    const { onContinue } = mount()
    fireEvent.click(await screen.findByRole('button', { name: '取消计划' }))
    expect(await screen.findByText('已取消')).toBeInTheDocument()
    expect(api.cancelParallelPlan).toHaveBeenCalledExactlyOnceWith(plan.session_id, plan.plan_id)
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('does not continue a decision that resolves after leaving the session', async () => {
    let resolve!: (value: ParallelPlan) => void
    vi.mocked(api.decideParallelPlan).mockReturnValue(
      new Promise(done => {
        resolve = done
      }),
    )
    const { onContinue, unmount } = mount()
    fireEvent.click(await screen.findByRole('button', { name: '批准并继续' }))
    unmount()
    await act(async () => resolve({ ...plan, status: 'approved' }))
    expect(onContinue).not.toHaveBeenCalled()
  })

  it('disables approval during parent activity but still allows rejection', async () => {
    mount(undefined, true)
    expect(await screen.findByRole('button', { name: '批准并继续' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '拒绝' })).toBeEnabled()
  })

  it.each(['partial', 'unknown', 'completed'] as const)(
    'renders %s without inventing successful results',
    async status => {
      vi.mocked(api.getParallelPlan).mockResolvedValue({ ...plan, status, results: [{ task_id: 'implementation' }] })
      const { onContinue } = mount()
      expect(await screen.findByText('已返回 1 条子任务结果')).toBeInTheDocument()
      expect(screen.queryByRole('button', { name: '批准并继续' })).not.toBeInTheDocument()
      expect(screen.queryByText('证据确认通过')).not.toBeInTheDocument()
      expect(onContinue).not.toHaveBeenCalled()
    },
  )
})
