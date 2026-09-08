import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineAttachSessionTool(store: EnvStore) {
  return defineTool({
    name: 'env_attach_session',
    description: 'Attach a session id to an environment session tree.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
      sessionId: { type: 'string', required: true, description: 'Session id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.attachSession(args.env, args.sessionId)),
  })
}
