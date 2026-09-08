import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineSetRunningTool(store: EnvStore) {
  return defineTool({
    name: 'env_set_running',
    description: 'Set whether an environment is running (Live indicator).',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
      running: { type: 'boolean', required: true, description: 'true if running' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.setRunning(args.env, args.running)),
  })
}
