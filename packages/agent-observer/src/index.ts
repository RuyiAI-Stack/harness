/**
 * Observes agent tool execution and files a report issue via github-bot.
 * @module dsh-agent-observer
 */

import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { ToolExecution, ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import type { Session } from '@deepseek-ai/dsh-session'
import type {} from '@ruyiAi/dsh-osc-github-bot'
import { Config, DEFAULT_TITLE } from './config.ts'
import { ObserverStore } from './service/store.ts'
import { defineReportTool } from './tools/api/report.ts'

export { Config, DEFAULT_TITLE } from './config.ts'
export { ObserverStore } from './service/store.ts'
export { renderReport, summarizeArgs, summarizeResult } from './tools/libs/report.ts'
export type { Observation } from './tools/libs/report.ts'

export default class AgentObserver extends Service {
  static inject = ['githubBot', 'tools']
  static Config: z<Config> = Config

  declare readonly config: Config
  private readonly store = new ObserverStore()

  constructor(ctx: Context, config: Config) {
    super(ctx, 'agentObserver')
    this.config = config

    ctx.on('tools/result', (exec: ToolExecution, result: ToolExecutionResult) => {
      this.store.record(exec, result)
    })

    ctx.on('session/disposed', (session: Session) => {
      void this.flush(String(session.id)).catch(error => {
        ctx.logger.warn(`agent-observer: report issue failed: ${errorMessage(error)}`)
      })
    })

    ctx.tools.register(defineReportTool(this))
  }

  /** File the accumulated observations as a GitHub issue, then reset the buffer. */
  async flush(sessionId = ''): Promise<{ number: number; url: string } | undefined> {
    const { enabled, org, repo, title } = this.config
    if (enabled === false) return undefined
    if (!org || !repo) {
      this.ctx.logger.warn('agent-observer: org and repo are not configured; skipping report issue')
      return undefined
    }
    const result = await this.ctx.githubBot.createIssue({
      org,
      repo,
      title: (title ?? DEFAULT_TITLE).replace('{session}', sessionId),
      body: this.store.render(sessionId),
    })
    this.store.clear()
    return { number: result.number, url: result.url }
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    agentObserver: AgentObserver
  }
}
