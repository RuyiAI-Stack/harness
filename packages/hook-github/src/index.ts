import { createHmac, timingSafeEqual } from 'node:crypto'
import { readFileSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { Context, Service } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-host-webserver'
import { Config } from './config.ts'

export type { Config } from './config.ts'

interface HookEvent {
  delivery: string
  event: string
  payload: unknown
}

export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

declare module '@deepseek-ai/cordis' {
  interface Events {
    'hook/github': (event: HookEvent) => void
  }
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(Buffer.from(chunk))
  return Buffer.concat(chunks)
}

export default class HookGithub extends Service {
  static inject = ['webServer']
  static Config = Config

  declare readonly config: Config

  constructor(ctx: Context, config: Config) {
    super(ctx, 'hookGithub')
    this.config = config

    ctx.effect(
      () =>
        ctx.webServer.register({
          kind: 'prefix',
          path: config.path,
          handler: async (req, res) => {
            try {
              await this.handle(req, res)
            } catch (err) {
              if (!(err instanceof HttpError)) throw err
              res.writeHead(err.status)
              res.end(err.message)
            }
          },
        }),
      'hook-github: api',
    )
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (req.method !== 'POST') {
      res.writeHead(405)
      res.end()
      return
    }

    const signature = req.headers['x-hub-signature-256']
    if (typeof signature !== 'string') throw new HttpError('missing GitHub signature', 401)

    const body = await readBody(req)
    let secret: Buffer
    try {
      secret = readFileSync(this.config.secretFile)
    } catch {
      throw new HttpError('unable to read GitHub webhook secret', 401)
    }

    const expected = 'sha256=' + createHmac('sha256', secret).update(body).digest('hex')
    const actual = Buffer.from(signature)
    const expectedBuffer = Buffer.from(expected)
    if (actual.length !== expectedBuffer.length || !timingSafeEqual(actual, expectedBuffer)) {
      throw new HttpError('invalid GitHub signature', 401)
    }

    const delivery = req.headers['x-github-delivery']
    const event = req.headers['x-github-event']
    if (typeof delivery !== 'string' || typeof event !== 'string') {
      throw new HttpError('missing GitHub event headers', 400)
    }

    let payload: unknown
    try {
      payload = JSON.parse(body.toString('utf8'))
    } catch {
      throw new HttpError('invalid JSON', 400)
    }

    this.ctx.emit('hook/github', { delivery, event, payload })
    res.writeHead(204)
    res.end()
  }
}
