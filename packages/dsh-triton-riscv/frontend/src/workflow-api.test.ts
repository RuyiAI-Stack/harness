import { afterEach, describe, expect, it, vi } from 'vitest'
import { setAccount } from './auth'
import { workflowApi, type Workflow } from './workflow-api'

afterEach(() => {
  setAccount(null)
  vi.unstubAllGlobals()
})

describe('workflow parallel decisions', () => {
  it.each([null, undefined])('does not post a decision when parallel_plan=%s', async parallel_plan => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    await expect(
      workflowApi.decideParallelPlan({ id: 'workflow-one', parallel_plan } as Workflow, true),
    ).rejects.toThrow('并行计划尚未加载')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([true, false])('sends the reviewed workflow revision, plan and digest with approved=%s', async approved => {
    const task = {
      id: 'workflow/one',
      revision: 7,
      parallel_plan: { plan_id: 'plan-one', digest: 'reviewed-digest' },
    } as Workflow
    const updated = { ...task, revision: 8, state: 'preparing' }
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => updated })
    vi.stubGlobal('fetch', fetchMock)
    setAccount({ id: 'account-one', username: 'reviewer', csrf_token: 'csrf-one' })

    expect(await workflowApi.decideParallelPlan(task, approved)).toEqual(updated)
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith(
      '/api/operator-workflows/workflow%2Fone/parallel-decision',
      expect.objectContaining({
        method: 'POST',
        credentials: 'same-origin',
        body: JSON.stringify({
          expected_revision: 7,
          plan_id: 'plan-one',
          digest: 'reviewed-digest',
          approved,
        }),
      }),
    )
    const headers = fetchMock.mock.calls[0][1].headers as Headers
    expect(headers.get('Content-Type')).toBe('application/json')
    expect(headers.get('X-Triton-Account')).toBe('account-one')
    expect(headers.get('X-CSRF-Token')).toBe('csrf-one')
  })

  it.each(['stale workflow revision', 'digest mismatch'])('surfaces %s without retrying', async detail => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 409, json: async () => ({ detail }) })
    vi.stubGlobal('fetch', fetchMock)
    const task = {
      id: 'workflow-one',
      revision: 7,
      parallel_plan: { plan_id: 'plan-one', digest: 'old' },
    } as Workflow

    await expect(workflowApi.decideParallelPlan(task, true)).rejects.toThrow(detail)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
