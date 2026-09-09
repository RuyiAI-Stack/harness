import { defineTool } from '@deepseek-ai/dsh-tools'
import type { InstallRunner } from '../../service/runner.ts'
import { text } from '../libs/text.ts'

export function defineStartTool(runner: InstallRunner, enabled: boolean) {
  return defineTool({
    name: 'self_update',
    description:
      'Start a background self-update: run the repository install script (tools/scripts/install-all.sh) to rebuild and re-register plugins, then automatically restart the host to apply the update (the session resumes after the restart).',
    parameters: {},
    output: { schema: { type: 'string' }, render: (_args, value) => text(value) },
    execute: async () => {
      if (!enabled) return 'self-update is disabled by config'
      const state = runner.start()
      if (!state) {
        const running = runner.snapshot()
        return `self-update already running (pid ${running.pid}); poll self_update_status`
      }
      return `self-update started (pid ${state.pid}, log ${runner.logPath}); poll self_update_status for progress`
    },
  })
}
