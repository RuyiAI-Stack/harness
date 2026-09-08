import type { ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { write } from '../libs/http.ts'

export function listEnvironments(store: EnvStore, res: ServerResponse): void {
  write(res, 200, 'application/json; charset=utf-8', JSON.stringify(store.list()))
}

export function deleteEnvironment(store: EnvStore, id: string, res: ServerResponse): void {
  store.delete(id)
  write(res, 200, 'text/plain; charset=utf-8', `deleted ${id}`)
}
