import type { IncomingMessage, ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { HttpError, readJson, write } from '../libs/http.ts'

export async function setRunning(
  store: EnvStore,
  id: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson<{ running?: unknown }>(req)
  if (typeof body.running !== 'boolean') throw new HttpError('env-builder: running must be boolean', 400)
  write(res, 200, 'application/json; charset=utf-8', JSON.stringify(store.setRunning(id, body.running)))
}
