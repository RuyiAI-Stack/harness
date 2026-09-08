import type { ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { write } from '../libs/http.ts'

export function selectEnvironment(store: EnvStore, id: string, res: ServerResponse): void {
  write(res, 200, 'application/json; charset=utf-8', JSON.stringify(store.select(id)))
}
