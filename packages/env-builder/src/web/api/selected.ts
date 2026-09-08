import type { ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { write } from '../libs/http.ts'

export function getSelected(store: EnvStore, res: ServerResponse): void {
  const env = store.selected()
  const repo = store.selectedRepo()
  write(
    res,
    200,
    'application/json; charset=utf-8',
    JSON.stringify(env ? { id: env.id, repo, running: env.running, sessionIds: env.sessionIds } : null),
  )
}
