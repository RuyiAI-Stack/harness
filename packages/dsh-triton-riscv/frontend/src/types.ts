export type RunStatus = 'awaiting-confirmation' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface Session {
  id: string
  title: string
  pinned: number | boolean
  created_at: string
  updated_at: string
}

export interface MessageMetadata {
  run_id?: string
  task?: {
    intent: string
    runtime?: string
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  metadata: MessageMetadata
  created_at: string
}

export interface Run {
  id: string
  session_id: string
  status: RunStatus
  command?: string
  created_at: string
  updated_at: string
}

export interface ModelRoute {
  tier: 'fast' | 'strong' | 'default'
  model: string
  provider: string
  reason: string
  timeout_seconds: number
  input_bytes: number
  policy_fingerprint: string
}

export interface RunEvent {
  id: string
  run_id: string
  sequence: number
  event_type: string
  payload: Partial<ModelRoute> & {
    scope?: 'chat' | 'role' | 'replica'
    line?: string
    message?: string
    text?: string
    command?: string
    detail?: string
    kind?: string
    tool?: string
    status?: string
    step?: number
    turn?: number
    proposal_id?: string
    proposal_type?: 'development' | 'repair'
    approval_type?: 'proposal' | 'command' | 'parallel'
    plan_id?: string
    summary?: string
    validation_run_id?: string
    operator?: string
    [key: string]: unknown
  }
  created_at: string
}

export interface ProposalReview {
  proposal_id: string
  status: string
  operator: string
  rationale: string
  diff?: string
  implementation_file?: string
  test_file?: string
}

export interface ValidationPlanReview {
  run_id: string
  operator: string
  status: string
  command?: string
  execution_target: 'local' | 'remote'
  receipt_path: string
  approval: {
    status: 'pending_approval' | 'approved' | 'rejected'
    execution_run_id?: string
    reviewer?: string
  }
}

export type ParallelPlanStatus =
  'pending' | 'approved' | 'rejected' | 'running' | 'completed' | 'partial' | 'cancelled' | 'unknown'

export interface ParallelPlan {
  plan_id: string
  session_id: string
  reason: string
  context: string
  user_request: string
  tasks: { task_id: string; goal: string; route?: ModelRoute }[]
  status: ParallelPlanStatus
  model: string
  provider?: string
  base_url?: string
  max_tool_calls?: number
  max_parallel: number
  timeout_seconds: number
  tools: string[]
  results: unknown[]
  digest: string
}

export interface SessionBundle {
  session: Session
  messages: Message[]
  runs: Run[]
}

export interface Bootstrap {
  operators: { operators?: number; [key: string]: unknown }
  project: { total_targets?: number; [key: string]: unknown }
  harness: {
    runtime: string
    plugin: {
      name: string
      version: string
      loaded: boolean
    }
    provider: string
    model: string
    api_configured: boolean
    remote_configured: boolean
    routing?: {
      enabled: boolean
      fastModel: string
      strongModel: string
      fastTimeoutSeconds: number
      strongTimeoutSeconds: number
    }
  }
  suggestions: string[]
}

export interface MessageResult {
  user_message: Message
  assistant_message: Message | null
  run: Run | null
}
