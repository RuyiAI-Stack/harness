import type {
  Bootstrap,
  MessageResult,
  Run,
  RunEvent,
  ProposalReview,
  ValidationPlanReview,
  Session,
  SessionBundle,
} from './types'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as { detail?: string } | null
    throw new Error(error?.detail || response.statusText || '请求失败')
  }
  return response.json() as Promise<T>
}

export const api = {
  bootstrap: () => request<Bootstrap>('/api/bootstrap'),
  listSessions: () => request<Session[]>('/api/sessions'),
  createSession: (title = '新对话') =>
    request<Session>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),
  getSession: (sessionId: string) => request<SessionBundle>(`/api/sessions/${sessionId}`),
  setSessionPinned: (sessionId: string, pinned: boolean) =>
    request<Session>(`/api/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify({ pinned }),
    }),
  deleteSession: async (sessionId: string) => {
    const response = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' })
    if (!response.ok) {
      const error = (await response.json().catch(() => null)) as { detail?: string } | null
      throw new Error(error?.detail || response.statusText || '删除会话失败')
    }
  },
  getSessionApprovals: (sessionId: string) => request<RunEvent[]>(`/api/sessions/${sessionId}/approvals`),
  sendMessage: (sessionId: string, content: string) =>
    request<MessageResult>(`/api/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),
  getRun: (runId: string) => request<Run>(`/api/runs/${runId}`),
  getEvents: (runId: string) => request<RunEvent[]>(`/api/runs/${runId}/events`),
  getProposal: (type: 'development' | 'repair', proposalId: string) =>
    request<ProposalReview>(
      `/api/${type === 'development' ? 'development-proposals' : 'repair-proposals'}/${proposalId}`,
    ),
  decideProposal: (type: 'development' | 'repair', proposalId: string, approve: boolean, reviewer: string) =>
    request<ProposalReview>(
      `/api/${type === 'development' ? 'development-proposals' : 'repair-proposals'}/${proposalId}/decision`,
      {
        method: 'POST',
        body: JSON.stringify({ approve, reviewer }),
      },
    ),
  getValidationPlan: (runId: string) => request<ValidationPlanReview>(`/api/validation-plans/${runId}`),
  decideValidationPlan: (runId: string, approve: boolean, reviewer: string) =>
    request<ValidationPlanReview>(`/api/validation-plans/${runId}/decision`, {
      method: 'POST',
      body: JSON.stringify({ approve, reviewer }),
    }),
}
