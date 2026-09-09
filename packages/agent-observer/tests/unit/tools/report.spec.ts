import { describe, expect, it } from 'vitest'
import { defineReportTool } from '../../../src/tools/api/report.ts'

describe('agent_observer_report', () => {
  it('registers as agent_observer_report', () => {
    const tool = defineReportTool({ flush: async () => undefined })
    expect(tool.name).toBe('agent_observer_report')
  })

  it('reports a skipped flush', async () => {
    const tool = defineReportTool({ flush: async () => undefined })
    const output = await tool.execute({}, {} as never)
    expect(output).toBe('Skipped: observer disabled or org/repo not configured')
  })

  it('reports a created issue', async () => {
    const tool = defineReportTool({ flush: async () => ({ number: 7, url: 'https://example.test' }) })
    const output = await tool.execute({}, {} as never)
    expect(output).toBe('Created issue #7: https://example.test')
  })
})
