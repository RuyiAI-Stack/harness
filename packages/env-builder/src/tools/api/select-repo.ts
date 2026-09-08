import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineSelectRepoTool(store: EnvStore) {
  return defineTool({
    name: 'env_select_repo',
    description: 'Select the active owner/name repository within the selected environment.',
    parameters: {
      repo: { type: 'string', required: true, description: 'owner/name, e.g. DangoSys/buckyball' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async args => store.selectRepo(args.repo),
  })
}
