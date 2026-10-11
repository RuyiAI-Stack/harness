import type { ProposalReview, ValidationPlanReview } from './types'

export interface Workflow {
  id: string
  session_id: string
  operator: string
  kind: 'create' | 'repair'
  state: string
  revision: number
  repair_count: number
  max_repairs: number
  recovery_action: string
  request: { requirement: string }
  result: {
    reason?: string
    rework_count?: number
    parent_workflow_id?: string
    lifecycle?: {
      child_workflow_id?: string
      validation_job_id?: string
      validation_run_id?: string
      reason?: string
      operation?: { action: string; status: string }
    }
  }
  steps: {
    id: string
    role: string
    state: string
    status: string
    job_id: string
    result: { summary?: string; artifact_id?: string }
  }[]
  proposal?: ProposalReview | null
  acceptance?: Record<string, unknown> | null
  artifact_error?: string | null
  validation_plan?: ValidationPlanReview | null
  job?: { id: string; status: string; result: { reason?: string } } | null
}

export interface WorkflowSubmission {
  request: {
    session_id: string
    kind: 'create' | 'repair'
    operator: string
    requirement: string
    auto_publish: boolean
    auto_repair: boolean
    max_repairs: number
  }
  request_id: string
  failure_run_id?: string
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch('/api/operator-workflows' + path, {
    ...options,
    headers: { 'Content-Type': 'application/json' },
  })
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求未通过校验，请检查输入。')
  return data as T
}

export const workflowApi = {
  config: () => request<{ enabled: boolean }>('/config'),
  list: (session: string) => request<Workflow[]>('?session_id=' + encodeURIComponent(session)),
  get: (id: string) => request<Workflow>('/' + encodeURIComponent(id)),
  submit: (body: WorkflowSubmission) =>
    request<{ payload: { workflow_id: string } }>('', { method: 'POST', body: JSON.stringify(body) }),
  action: (task: Workflow, action: string) =>
    request<Workflow>(`/${encodeURIComponent(task.id)}/${action}`, {
      method: 'POST',
      body: JSON.stringify({ expected_revision: task.revision }),
    }),
}
