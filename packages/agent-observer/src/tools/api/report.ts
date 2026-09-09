import { defineTool } from '@deepseek-ai/dsh-tools'

export interface ReportFlusher {
  flush(sessionId?: string): Promise<{ number: number; url: string } | undefined>
}

export function defineReportTool(flusher: ReportFlusher) {
  return defineTool({
    name: 'agent_observer_report',
    description: 'Send the recorded session observations to the configured GitHub issue.',
    parameters: {},
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute: async () => {
      const result = await flusher.flush()
      return result
        ? `Created issue #${result.number}: ${result.url}`
        : 'Skipped: observer disabled or org/repo not configured'
    },
  })
}
