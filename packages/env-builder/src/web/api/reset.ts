import type { ServerResponse } from 'node:http'
import type { EnvStore } from '../../service/store.ts'
import { write } from '../libs/http.ts'

export function resetEnvironment(store: EnvStore, id: string, res: ServerResponse): void {
  store.reset(id)
  write(res, 200, 'text/plain; charset=utf-8', `reset ${id}`)
}
