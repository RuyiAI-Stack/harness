import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineResetTool(store: EnvStore) {
  return defineTool({
    name: 'env_reset',
    description: 'git reset --hard && git clean -fd for every component in the environment.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => {
      store.reset(args.env)
      return `reset ${args.env}`
    },
  })
}
