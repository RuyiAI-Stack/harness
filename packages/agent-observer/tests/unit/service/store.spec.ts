import { describe, expect, it } from 'vitest'
import type { ToolExecution, ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import { ObserverStore } from '../../../src/service/store.ts'

function exec(name: string, args: unknown): ToolExecution {
  return { name, arguments: args } as unknown as ToolExecution
}

function result(isError: boolean, text: string): ToolExecutionResult {
  return { isError, content: [{ type: 'text', text }] } as unknown as ToolExecutionResult
}

describe('ObserverStore', () => {
  it('records and renders observations', () => {
    const store = new ObserverStore()
    store.record(exec('bash', { command: 'ls' }), result(false, 'file.txt'))
    store.record(exec('read', { path: 'a' }), result(true, 'boom'))

    expect(store.snapshot()).toHaveLength(2)
    const report = store.render('session-1')
    expect(report).toContain('- Session: `session-1`')
    expect(report).toContain('| 0 | `bash` | no |')
    expect(report).toContain('| 1 | `read` | yes |')
  })

  it('clears observations', () => {
    const store = new ObserverStore()
    store.record(exec('bash', {}), result(false, 'x'))
    store.clear()
    expect(store.snapshot()).toHaveLength(0)
  })
})
