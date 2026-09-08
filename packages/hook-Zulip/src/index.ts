import { readFileSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { Context, Service } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-host-webserver'
import z from '@deepseek-ai/schemastery'

const DEFAULT_PATH = '/integrations/hook-zulip'

export interface Config {
  tokenFile: string
  path?: string
  botEmail?: string
}

export const Config: z<Config> = z.object({
  tokenFile: z.string().required(),
  path: z.string().default(DEFAULT_PATH),
  botEmail: z.string(),
})

declare module '@deepseek-ai/cordis' {
  interface Context {
    hookZulip: HookZulip
  }

  interface Events {
    'hook/zulip'(payload: Record<string, unknown>): void
  }
}

class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(Buffer.from(chunk))
  return Buffer.concat(chunks).toString('utf8')
}

export default class HookZulip extends Service {
  static inject = ['webServer']
  static Config: z<Config> = Config

  readonly config: Config

  constructor(ctx: Context, config: Config) {
    super(ctx, 'hookZulip')
    this.config = config

    ctx.effect(
      () =>
        ctx.webServer.register({
          kind: 'prefix',
          path: config.path ?? DEFAULT_PATH,
          handler: (req, res) => void this.handle(req, res),
        }),
      'hook-zulip: api',
    )
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    try {
      if (req.method !== 'POST') {
        res.writeHead(405, { allow: 'POST', 'content-type': 'application/json; charset=utf-8' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }

      let body: Record<string, unknown>
      try {
        const parsed: unknown = JSON.parse(await readBody(req))
        if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error('object required')
        body = parsed as Record<string, unknown>
      } catch {
        res.writeHead(400, { 'content-type': 'application/json; charset=utf-8' })
        res.end(JSON.stringify({ error: 'invalid json' }))
        return
      }

      const expected = readFileSync(this.config.tokenFile, 'utf8').trim()
      if (typeof body.token !== 'string' || body.token !== expected) throw new HttpError('unauthorized', 401)
      if (this.config.botEmail !== undefined && body.bot_email !== this.config.botEmail) {
        throw new HttpError('unauthorized', 401)
      }

      const { token: _token, ...payload } = body
      this.ctx.emit('hook/zulip', payload)
      res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' })
      res.end(JSON.stringify({ response_not_required: true }))
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500
      res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' })
      res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }))
    }
  }
}
