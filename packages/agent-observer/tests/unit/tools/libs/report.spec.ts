import { describe, expect, it } from 'vitest'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { renderReport, summarizeArgs, summarizeResult } from '../../../../src/tools/libs/report.ts'

describe('summarizeArgs', () => {
  it('serializes arguments as JSON', () => {
    expect(summarizeArgs({ command: 'ls' })).toBe('{"command":"ls"}')
  })

  it('returns empty for undefined', () => {
    expect(summarizeArgs(undefined)).toBe('')
  })
})

describe('summarizeResult', () => {
  it('joins text blocks', () => {
    const content: ContentBlock[] = [
      { type: 'text', text: 'first' },
      { type: 'text', text: 'second' },
    ]
    expect(summarizeResult({ content })).toBe('first\nsecond')
  })
})

describe('renderReport', () => {
  it('renders an empty report', () => {
    const report = renderReport('session-1', [])
    expect(report).toContain('# Agent execution report')
    expect(report).toContain('- Session: `session-1`')
    expect(report).toContain('- Tool calls: 0')
  })

  it('renders observations as a table row', () => {
    const report = renderReport('session-1', [
      { seq: 0, name: 'bash', arguments: { command: 'ls' }, isError: false, summary: 'file.txt' },
    ])
    expect(report).toContain('| 0 | `bash` | no |')
    expect(report).toContain('file.txt')
  })
})
