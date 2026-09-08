import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineDeleteTool(store: EnvStore) {
  return defineTool({
    name: 'env_delete',
    description: 'Delete an environment directory and remove it from the manifest.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => {
      store.delete(args.env)
      return `deleted ${args.env}`
    },
  })
}
