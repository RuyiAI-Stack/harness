import { Context, Service } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

export type Path = 'pr' | 'bot'

export interface PrTarget {
  readonly repo: string
  readonly number: number
}

export interface BotTarget {
  readonly sessionId: string
}

export interface SendToPrRequest extends PrTarget {
  readonly body: string
}

export interface SendToBotRequest extends BotTarget {
  readonly body: string
}

export type SendRequest =
  | { readonly path: 'pr'; readonly target: PrTarget; readonly body: string }
  | { readonly path: 'bot'; readonly target: BotTarget; readonly body: string }

export interface PrSendResult {
  readonly path: 'pr'
  readonly target: PrTarget
  readonly commentId: number
  readonly url: string
}

export interface BotSendResult {
  readonly path: 'bot'
  readonly target: BotTarget
}

export type SendResult = PrSendResult | BotSendResult

export interface PathEvent {
  readonly path: Path
  readonly target: PrTarget | BotTarget
}

export interface SentEvent extends PathEvent {
  readonly result: SendResult
}

interface Role {
  githubJson(path: string, init?: RequestInit): Promise<unknown>
}

interface Agents {
  get(id: string): { readonly id: string } | undefined
}

interface AgentRuntime {
  prompt(
    agent: { readonly id: string },
    prompt: readonly [{ readonly type: 'text'; readonly text: string }],
  ): Promise<void>
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    role: Role
    agents: Agents
    agentRuntime: AgentRuntime
    prBotPath: PrBotPath
  }

  interface Events {
    'pr-bot/path': (event: PathEvent) => void
    'pr-bot/sent': (event: SentEvent) => void
  }
}

function requireBody(body: string): void {
  if (typeof body !== 'string' || body.length === 0) throw new Error('pr-bot-path: body is required')
}

function requireRepo(repo: string): [string, string] {
  if (typeof repo !== 'string') throw new Error('pr-bot-path: repo is required')
  const parts = repo.split('/')
  if (parts.length !== 2 || parts.some(part => part.length === 0 || part === '.' || part === '..')) {
    throw new Error('pr-bot-path: repo must be in owner/name form')
  }
  return parts as [string, string]
}

function requireNumber(number: number): void {
  if (!Number.isSafeInteger(number) || number <= 0) throw new Error('pr-bot-path: pull request number must be positive')
}

function requireSessionId(sessionId: string): void {
  if (typeof sessionId !== 'string' || sessionId.length === 0) throw new Error('pr-bot-path: sessionId is required')
}

function requirePath(path: string): asserts path is Path {
  if (path !== 'pr' && path !== 'bot') throw new Error(`pr-bot-path: invalid path ${String(path)}`)
}

export default class PrBotPath extends Service {
  static inject = ['role', 'agents', 'agentRuntime', 'tools']

  constructor(ctx: Context) {
    super(ctx, 'prBotPath')
    ctx.tools.register(
      defineTool({
        name: 'pr_bot_send_to_pr',
        description: 'Send a human-authenticated comment to a GitHub pull request.',
        parameters: {
          repo: { type: 'string', required: true, description: 'Repository in owner/name form.' },
          number: { type: 'number', required: true, description: 'Pull request number.' },
          body: { type: 'string', required: true },
        },
        output: {
          schema: {
            type: 'object',
            properties: {
              path: { type: 'string', required: true },
              target: { type: 'json', required: true },
              commentId: { type: 'number', required: true },
              url: { type: 'string', required: true },
            },
            additionalProperties: false,
          },
          render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
        },
        execute: async args => this.sendToPr(args as SendToPrRequest),
      }),
    )
    ctx.tools.register(
      defineTool({
        name: 'pr_bot_send_to_bot',
        description: 'Send a message to a local bot session without contacting GitHub.',
        parameters: {
          sessionId: { type: 'string', required: true },
          body: { type: 'string', required: true },
        },
        output: {
          schema: {
            type: 'object',
            properties: {
              path: { type: 'string', required: true },
              target: { type: 'json', required: true },
            },
            additionalProperties: false,
          },
          render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
        },
        execute: async args => this.sendToBot(args as SendToBotRequest),
      }),
    )
  }

  async send(request: SendRequest): Promise<SendResult> {
    requirePath(request.path)
    requireBody(request.body)
    ctxEmit(this.ctx, 'pr-bot/path', { path: request.path, target: request.target })
    const result =
      request.path === 'pr'
        ? await this.sendPr(request.target, request.body)
        : await this.sendBot(request.target, request.body)
    ctxEmit(this.ctx, 'pr-bot/sent', { path: request.path, target: request.target, result })
    return result
  }

  async sendToPr(request: SendToPrRequest): Promise<PrSendResult> {
    requireBody(request.body)
    return this.send({
      path: 'pr',
      target: { repo: request.repo, number: request.number },
      body: request.body,
    }) as Promise<PrSendResult>
  }

  async sendToBot(request: SendToBotRequest): Promise<BotSendResult> {
    requireBody(request.body)
    return this.send({
      path: 'bot',
      target: { sessionId: request.sessionId },
      body: request.body,
    }) as Promise<BotSendResult>
  }

  private async sendPr(target: PrTarget, body: string): Promise<PrSendResult> {
    const [owner, repo] = requireRepo(target.repo)
    requireNumber(target.number)
    const response = await this.ctx.role.githubJson(`/repos/${owner}/${repo}/issues/${target.number}/comments`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ body }),
    })
    if (response === null || typeof response !== 'object')
      throw new Error('pr-bot-path: GitHub comment response is invalid')
    const result = response as { id?: unknown; html_url?: unknown }
    if (!Number.isSafeInteger(result.id) || typeof result.html_url !== 'string' || result.html_url.length === 0) {
      throw new Error('pr-bot-path: GitHub comment response is missing id or html_url')
    }
    return { path: 'pr', target, commentId: result.id, url: result.html_url }
  }

  private async sendBot(target: BotTarget, body: string): Promise<BotSendResult> {
    requireSessionId(target.sessionId)
    const agent = this.ctx.agents.get(target.sessionId)
    if (agent === undefined) throw new Error(`pr-bot-path: bot session "${target.sessionId}" is not live`)
    await this.ctx.agentRuntime.prompt(agent, [{ type: 'text', text: body }])
    return { path: 'bot', target }
  }
}

function ctxEmit(ctx: Context, event: 'pr-bot/path', value: PathEvent): void
function ctxEmit(ctx: Context, event: 'pr-bot/sent', value: SentEvent): void
function ctxEmit(ctx: Context, event: 'pr-bot/path' | 'pr-bot/sent', value: PathEvent | SentEvent): void {
  ctx.emit(event, value as never)
}
