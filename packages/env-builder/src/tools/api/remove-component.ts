import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineRemoveComponentTool(store: EnvStore) {
  return defineTool({
    name: 'env_remove_component',
    description: 'Remove a GitHub component (owner/name) from an environment.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
      repo: { type: 'string', required: true, description: 'owner/name' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => {
      store.removeComponent(args.env, args.repo)
      return `removed ${args.repo} from ${args.env}`
    },
  })
}
