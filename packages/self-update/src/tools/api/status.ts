import { defineTool } from '@deepseek-ai/dsh-tools'
import type { InstallRunner } from '../../service/runner.ts'
import { readTailFile } from '../libs/tail.ts'
import { text } from '../libs/text.ts'

export function defineStatusTool(runner: InstallRunner) {
  return defineTool({
    name: 'self_update_status',
    description: 'Report whether the background self-update is running or finished, and print the tail of its log.',
    parameters: {},
    output: { schema: { type: 'string' }, render: (_args, value) => text(value) },
    execute: async () => {
      const state = runner.snapshot()
      const lines = [`status: ${state.phase}`]
      if (state.pid !== undefined) lines.push(`pid: ${state.pid}`)
      if (state.startedAt !== undefined) lines.push(`startedAt: ${new Date(state.startedAt).toISOString()}`)
      if (state.finishedAt !== undefined) lines.push(`finishedAt: ${new Date(state.finishedAt).toISOString()}`)
      lines.push(`exitCode: ${state.exitCode ?? '-'}`)
      if (state.error) lines.push(`error: ${state.error}`)
      lines.push(`log: ${runner.logPath}`)

      if (state.phase === 'ok') {
        lines.push(
          '',
          'self-update finished. The host restarts automatically to load the rebuilt plugins; this session resumes after the restart.',
        )
      } else if (state.phase === 'failed') {
        lines.push('', 'self-update failed. Inspect the full log before retrying.')
      }

      if (runner.logExists()) {
        const tail = readTailFile(runner.logPath)
        lines.push('', '--- tail ---', tail || '(empty)')
      }
      return lines.join('\n')
    },
  })
}
