import type { IncomingMessage, ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { HttpError, readJson, write } from '../libs/http.ts'

export async function removeComponent(
  store: EnvStore,
  id: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson<{ repo?: string }>(req)
  if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
  store.removeComponent(id, body.repo)
  write(res, 200, 'text/plain; charset=utf-8', `removed ${body.repo}`)
}
