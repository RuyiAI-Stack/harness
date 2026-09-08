import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineRenameTool(store: EnvStore) {
  return defineTool({
    name: 'env_rename',
    description: 'Rename an environment id; throws if missing or conflict.',
    parameters: {
      env: { type: 'string', required: true, description: 'Current environment id' },
      id: { type: 'string', required: true, description: 'New environment id' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => JSON.stringify(store.rename(args.env, args.id)),
  })
}
