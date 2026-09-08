import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineRegisterComponentTool(store: EnvStore) {
  return defineTool({
    name: 'env_register_component',
    description:
      'Register an already cloned and built repository in the environment manifest. Call after bash install completes.',
    parameters: {
      env: { type: 'string', required: true, description: 'Environment id, e.g. project1' },
      repo: { type: 'string', required: true, description: 'owner/repo or https://github.com/owner/repo' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => store.registerComponent(args.env, args.repo),
  })
}
