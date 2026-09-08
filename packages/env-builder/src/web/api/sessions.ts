import type { IncomingMessage, ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { HttpError, readJson, write } from '../libs/http.ts'

export async function attachSession(
  store: EnvStore,
  id: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson<{ sessionId?: string }>(req)
  if (!body.sessionId) throw new HttpError('env-builder: missing session id', 400)
  write(res, 200, 'application/json; charset=utf-8', JSON.stringify(store.attachSession(id, body.sessionId)))
}

export async function detachSession(
  store: EnvStore,
  id: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson<{ sessionId?: string }>(req)
  if (!body.sessionId) throw new HttpError('env-builder: missing session id', 400)
  write(res, 200, 'application/json; charset=utf-8', JSON.stringify(store.detachSession(id, body.sessionId)))
}
