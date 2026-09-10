import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineMarkCleanTool(ctx: Context, store: EnvStore) {
  return defineTool({
    name: 'env_mark_clean',
    description:
      'Mark an environment clean and rebindable after a clean task. Call only when sessions are detached and the env path is ready.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => {
      const env = store.markClean(args.env)
      ctx.emit('envBuilder/cleaned', env.id)
      return `cleaned ${env.id}`
    },
  })
}
