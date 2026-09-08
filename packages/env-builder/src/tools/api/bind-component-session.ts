import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineBindComponentSessionTool(store: EnvStore) {
  return defineTool({
    name: 'env_bind_component_session',
    description: 'Bind a session id to a component; throws if missing or already bound.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id' },
      repo: { type: 'string', required: true, description: 'owner/name' },
      sessionId: { type: 'string', required: true, description: 'Session id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.bindComponentSession(args.env, args.repo, args.sessionId)),
  })
}
