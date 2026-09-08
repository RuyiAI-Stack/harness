import { defineTool } from '@deepseek-ai/dsh-tools'
import type { EnvStore } from '../../service/store.ts'
import { text } from '../libs/text.ts'

export function defineListTool(store: EnvStore) {
  return defineTool({
    name: 'env_list',
    description: 'List environments and their GitHub components under environment/projectN.',
    parameters: {},
    output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
    execute: async () => JSON.stringify(store.list(), null, 2),
  })
}
