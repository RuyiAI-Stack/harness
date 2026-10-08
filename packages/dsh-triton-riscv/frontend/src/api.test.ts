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

    await api.sendMessage('session-1', '验证 relu_and_mul')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/sessions/session-1/messages',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ content: '验证 relu_and_mul' }),
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

    expect(fetchMock).toHaveBeenCalledWith('/api/sessions/session-1', { method: 'DELETE' })
  })

  it('surfaces FastAPI validation errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'message cannot be empty' }, false)))

    await expect(api.sendMessage('session-1', '')).rejects.toThrow('message cannot be empty')
  })
})
