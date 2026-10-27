export interface TaskConfig {
  systemPrompt: string
  resultSchema: Record<string, unknown>
}
export interface Config {
  enabled?: boolean
  python: string
  model: {
    provider: string
    model: string
    apiKeyEnv: string
    baseURL?: string | null
    requestTimeoutSeconds?: number
  }
  tasks: Record<string, TaskConfig>
}
export type Outcome = 'validated' | 'invalid_output' | 'incomplete' | 'error' | 'cancelled'
export interface Usage {
  schema_version: 1
  session_id: string
  outcome: string
  attempt_count: number
  attempts: Record<string, unknown>[]
  totals: Record<string, number | string | null>
  issues: string[]
  cost: null
  cost_status: 'pricing_not_configured'
  [key: string]: unknown
}
export interface RunResult<T = unknown> {
  session_id: string | null
  outcome: Outcome
  result: T | null
  error: string | null
  usage: Usage | null
  harness?: { session_id: string; final_response: string; finish_reason: string | null; events: object[] }
}
export interface MultiAgent {
  run(request: { task: string; input: unknown }): Readonly<{ id: string }>
  status(id: string): { id: string; sessionId: string | null; status: 'running' | 'cancelling' | Outcome }
  result<T = unknown>(id: string): Promise<RunResult<T>>
  usage(id: string): Usage | null
  cancel(id: string): boolean
  release(id: string): void
  close(): Promise<void>
}
export function createMultiAgent(config: Config): MultiAgent
export const name: 'multi-agent'
export const inject: []
export function apply(
  ctx: {
    provide(name: string, service: MultiAgent): unknown
    effect(install: () => () => Promise<void>, name: string): unknown
  },
  config: Config,
): void

declare module '@deepseek-ai/cordis' {
  interface Context {
    multiAgent: MultiAgent
  }
}
