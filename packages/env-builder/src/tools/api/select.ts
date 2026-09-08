import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineSelectTool(store: EnvStore) {
  return defineTool({
    name: 'env_select',
    description: 'Select the active environment for repository and workload tools.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.select(args.env)),
  })
}
