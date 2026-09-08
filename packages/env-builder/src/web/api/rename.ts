import type { IncomingMessage, ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { HttpError, readJson, write } from '../libs/http.ts'

export async function renameEnvironment(
  store: EnvStore,
  id: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson<{ id?: string }>(req)
  if (!body.id) throw new HttpError('env-builder: missing new environment id', 400)
  write(res, 200, 'application/json; charset=utf-8', JSON.stringify(store.rename(id, body.id)))
}
