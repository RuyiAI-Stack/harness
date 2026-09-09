import type { ContentBlock } from '@deepseek-ai/dsh-llm'

/** One recorded tool execution, captured from `tools/result`. */
export interface Observation {
  /** Zero-based call order within the session. */
  seq: number
  name: string
  arguments: unknown
  isError: boolean
  /** Truncated human-readable result text. */
  summary: string
}

const ARGS_LIMIT = 240
const RESULT_LIMIT = 400

export function summarizeArgs(args: unknown): string {
  if (args === undefined || args === null) return ''
  let text: string
  try {
    text = JSON.stringify(args)
  } catch {
    text = String(args)
  }
  return truncate(text, ARGS_LIMIT)
}

export function summarizeResult(result: { content: ContentBlock[] }): string {
  const parts: string[] = []
  for (const block of result.content) {
    if (block.type === 'text') parts.push(block.text)
  }
  return truncate(parts.join('\n').trim(), RESULT_LIMIT)
}

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}…`
}

function escapeCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ')
}

export function renderReport(sessionId: string, observations: readonly Observation[]): string {
  const lines = ['# Agent execution report', '', `- Session: \`${sessionId}\``, `- Tool calls: ${observations.length}`]
  if (observations.length > 0) {
    lines.push('', '| # | tool | error | arguments | result |', '|---|---|---|---|---|')
    for (const obs of observations) {
      lines.push(
        `| ${obs.seq} | \`${obs.name}\` | ${obs.isError ? 'yes' : 'no'} | ${escapeCell(summarizeArgs(obs.arguments))} | ${escapeCell(obs.summary)} |`,
      )
    }
  }
  return `${lines.join('\n')}\n`
}
