import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineEnsureComponentTool(store: EnvStore) {
  return defineTool({
    name: 'env_ensure_component',
    description: 'Verify the selected repository component is cloned in the selected environment.',
    parameters: {
      name: { type: 'string', required: true, description: 'Repository name, e.g. buckyball' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => store.ensureComponent(args.name),
  })
}
