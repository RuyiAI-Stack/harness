/**
 * Buckyball workload-construction capability for the DeepSeek Harness.
 * @module dsh-workload-integration
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { playbookText } from './prompt.ts'
import { defineModelInfoTool } from './tools/model-info.ts'
import { defineExistingModelsIndexTool } from './tools/existing-models-index.ts'
import { defineWorkloadAuditTool } from './tools/workload-audit.ts'

export const name = 'tool-workload-integration'
export const inject = ['tools', 'systemPrompt', 'role', 'envBuilder']

export interface Config {
  fetchTimeoutMs: number
  hfToken: string
  hfEndpoint: string
}

export const Config: z<Config> = z.object({
  fetchTimeoutMs: z.number().default(30000),
  hfToken: z.string().required(),
  hfEndpoint: z.string().default('https://huggingface.co'),
})

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

function selectedRepoPath(ctx: Context): string {
  const repo = ctx.envBuilder.store.selectedRepo()
  if (repo === undefined) throw new Error('workload integration requires a selected repository')
  if (!ctx.role.repoList.includes(repo)) throw new Error('selected repository is not in the repoList: ' + repo)
  const separator = repo.indexOf('/')
  if (separator <= 0 || separator === repo.length - 1)
    throw new Error('selected repository must use owner/name format: ' + repo)
  return ctx.envBuilder.store.ensureComponent(repo.slice(separator + 1))
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
