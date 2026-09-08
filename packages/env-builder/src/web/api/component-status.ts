import type { IncomingMessage, ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { HttpError, readJson, write } from '../libs/http.ts'

export async function setComponentStatus(
  store: EnvStore,
  id: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson<{ repo?: string; status?: unknown }>(req)
  if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
  write(
    res,
    200,
    'application/json; charset=utf-8',
    JSON.stringify(store.setComponentStatus(id, body.repo, body.status)),
  )
}
