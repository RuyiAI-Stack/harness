import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineSetComponentStatusTool(store: EnvStore) {
  return defineTool({
    name: 'env_set_component_status',
    description: 'Set component status: ready | modified | installing.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
      repo: { type: 'string', required: true, description: 'owner/name' },
      status: { type: 'string', required: true, description: 'ready | modified | installing' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.setComponentStatus(args.env, args.repo, args.status)),
  })
}
