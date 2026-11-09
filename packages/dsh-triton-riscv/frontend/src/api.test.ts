import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

function jsonResponse(value: unknown, ok = true): Response {
  return {
    ok,
    statusText: ok ? 'OK' : 'Bad Request',
    json: async () => value,
  } as Response
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('FastAPI client', () => {
  it('sends natural-language tasks to the active session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ run: null }))
    vi.stubGlobal('fetch', fetchMock)

    await api.sendMessage('session-1', '验证 relu_and_mul', 'request-1')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/sessions/session-1/messages',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ content: '验证 relu_and_mul', request_id: 'request-1' }),
      }),
    )
  })

  it('records exact validation-command approval before execution', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ run_id: 'run-1', approval: { status: 'approved' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.decideValidationPlan('run-1', true, 'ada-cl25')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/validation-plans/run-1/decision',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ approve: true, reviewer: 'ada-cl25' }),
      }),
    )
  })

  it('records proposal approval through the host-only review endpoint', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ proposal_id: 'development-proposal-1', status: 'approved' }))
    vi.stubGlobal('fetch', fetchMock)

    await api.decideProposal('development', 'development-proposal-1', true, 'ada-cl25')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/development-proposals/development-proposal-1/decision',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ approve: true, reviewer: 'ada-cl25' }),
      }),
    )
  })

  it('persists session pinning and reads approvals across runs', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]))
    vi.stubGlobal('fetch', fetchMock)

    await api.setSessionPinned('session-1', true)
    await api.getSessionApprovals('session-1')

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/sessions/session-1',
      expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify({ pinned: true }),
      }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/sessions/session-1/approvals', expect.any(Object))
  })

  it('deletes a session through the HTTP API', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(null))
    vi.stubGlobal('fetch', fetchMock)

    await api.deleteSession('session-1')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/sessions/session-1',
      expect.objectContaining({ method: 'DELETE', credentials: 'same-origin' }),
    )
    expect(fetchMock.mock.calls[0][1].headers.get('Content-Type')).toBe('application/json')
  })

  it('surfaces FastAPI validation errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'message cannot be empty' }, false)))

    await expect(api.sendMessage('session-1', '')).rejects.toThrow('message cannot be empty')
  })

  it('reads and cancels session-scoped parallel plans with encoded identifiers', async () => {
    const plan = { plan_id: 'plan/1', session_id: 'session/1', status: 'cancelled' }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(plan))
    vi.stubGlobal('fetch', fetchMock)
    expect(await api.getParallelPlan('session/1', 'plan/1')).toEqual(plan)
    expect(await api.cancelParallelPlan('session/1', 'plan/1')).toEqual(plan)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/sessions/session%2F1/parallel-plans/plan%2F1',
      expect.any(Object),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/sessions/session%2F1/parallel-plans/plan%2F1/cancel',
      expect.objectContaining({ method: 'POST', body: '{}' }),
    )
  })

  it.each([true, false])('sends approved=%s and the exact parallel-plan digest', async approved => {
    const plan = { plan_id: 'plan-1', status: approved ? 'approved' : 'rejected' }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(plan))
    vi.stubGlobal('fetch', fetchMock)
    expect(await api.decideParallelPlan('session-1', 'plan-1', approved, 'reviewed-digest')).toEqual(plan)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/sessions/session-1/parallel-plans/plan-1/decision',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ approved, digest: 'reviewed-digest' }),
      }),
    )
  })

  it('surfaces wrong-digest decisions without retrying', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ detail: 'digest mismatch' }, false))
    vi.stubGlobal('fetch', fetchMock)
    await expect(api.decideParallelPlan('session-1', 'plan-1', true, 'wrong-digest')).rejects.toThrow('digest mismatch')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
