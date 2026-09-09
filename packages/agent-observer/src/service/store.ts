import type { ToolExecution, ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import { renderReport, summarizeResult, type Observation } from '../tools/libs/report.ts'

/** In-memory record of the tool calls observed during one session. */
export class ObserverStore {
  private readonly observations: Observation[] = []

  record(exec: ToolExecution, result: ToolExecutionResult): void {
    this.observations.push({
      seq: this.observations.length,
      name: exec.name,
      arguments: exec.arguments,
      isError: result.isError,
      summary: summarizeResult(result),
    })
  }

  snapshot(): readonly Observation[] {
    return this.observations
  }

  render(sessionId: string): string {
    return renderReport(sessionId, this.observations)
  }

  clear(): void {
    this.observations.length = 0
  }
}
