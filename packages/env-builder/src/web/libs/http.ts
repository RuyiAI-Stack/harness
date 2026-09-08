import type { IncomingMessage, ServerResponse } from 'node:http'

export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

export function statusOf(err: unknown): number {
  if (err instanceof HttpError) return err.status
  return 500
}

export function write(res: ServerResponse, status: number, type: string, body: string) {
  res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' })
  res.end(body)
}

export async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(Buffer.from(chunk))
  return Buffer.concat(chunks).toString('utf8')
}

export async function readJson<T>(req: IncomingMessage): Promise<T> {
  try {
    return JSON.parse(await readBody(req)) as T
  } catch {
    throw new HttpError('env-builder: invalid json body', 400)
  }
}
