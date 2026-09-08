/**
 * Buckyball workload-construction capability for the DeepSeek Harness.
 * @module dsh-workload-integration
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Config } from './config.ts'
import { playbookText } from './prompt.ts'
import { defineExistingModelsIndexTool } from './tools/api/existing-models-index.ts'
import { defineModelInfoTool } from './tools/api/model-info.ts'
import { defineWorkloadAuditTool } from './tools/api/workload-audit.ts'
import { selectedRepoPath } from './tools/libs/repo.ts'

export type { Config } from './config.ts'
export { Config } from './config.ts'
export { playbookText } from './prompt.ts'
export { selectedRepoPath } from './tools/libs/repo.ts'

export const name = 'tool-workload-integration'
export const inject = ['tools', 'systemPrompt', 'role', 'envBuilder']

declare module '@deepseek-ai/cordis' {
  interface Context {
    role: {
      readonly repoList: readonly string[]
    }
    envBuilder: {
      store: {
        selectedRepo(): string | undefined
        ensureComponent(name: string): string
      }
    }
  }
}

export function apply(ctx: Context, config: Config): void {
  const buckyball = () => selectedRepoPath(ctx)
  ctx.systemPrompt.section({
    name: 'buckyball:workload',
    order: 60,
    text: () => (ctx.envBuilder.store.selectedRepo() === undefined ? '' : playbookText(buckyball())),
  })
  ctx.tools.register(defineModelInfoTool(config))
  ctx.tools.register(defineExistingModelsIndexTool(buckyball))
  ctx.tools.register(defineWorkloadAuditTool(buckyball))
}
