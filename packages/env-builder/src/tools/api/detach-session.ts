import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineDetachSessionTool(store: EnvStore) {
  return defineTool({
    name: 'env_detach_session',
    description: 'Detach a session id from an environment.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
      sessionId: { type: 'string', required: true, description: 'Session id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.detachSession(args.env, args.sessionId)),
  })
}
