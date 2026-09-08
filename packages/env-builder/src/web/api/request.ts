import type { IncomingMessage, ServerResponse } from 'node:http'
import { PATH } from '../../constants.ts'
import type { EnvStore } from '../../service/store.ts'
import { HttpError } from '../libs/http.ts'
import { bindComponentSession } from './component-session.ts'
import { setComponentStatus } from './component-status.ts'
import { removeComponent } from './components.ts'
import { deleteEnvironment, listEnvironments } from './environments.ts'
import { renameEnvironment } from './rename.ts'
import { resetEnvironment } from './reset.ts'
import { setRunning } from './running.ts'
import { selectEnvironment } from './select.ts'
import { getSelected } from './selected.ts'
import { selectRepo } from './selected-repo.ts'
import { attachSession, detachSession } from './sessions.ts'

export async function handleRequest(store: EnvStore, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url!, 'http://local')
  const path = url.pathname
  const method = req.method ?? 'GET'

  if (method === 'GET' && path === `${PATH}/environments`) {
    listEnvironments(store, res)
    return
  }
  if (method === 'GET' && path === `${PATH}/selected`) {
    getSelected(store, res)
    return
  }
  if (method === 'POST' && path === `${PATH}/selected-repo`) {
    await selectRepo(store, req, res)
    return
  }
  const m = path.match(
    new RegExp(
      `^${PATH}/environments/([^/]+)(?:/(components|component-status|component-session|sessions|reset|select|running|rename))?$`,
    ),
  )
  if (!m) throw new HttpError(`env-builder: bad path ${path}`, 404)
  const id = decodeURIComponent(m[1]!)
  const action = m[2]

  if (method === 'DELETE' && !action) {
    deleteEnvironment(store, id, res)
    return
  }
  if (method === 'POST' && action === 'select') {
    selectEnvironment(store, id, res)
    return
  }
  if (method === 'POST' && action === 'reset') {
    resetEnvironment(store, id, res)
    return
  }
  if (method === 'POST' && action === 'running') {
    await setRunning(store, id, req, res)
    return
  }
  if (method === 'DELETE' && action === 'components') {
    await removeComponent(store, id, req, res)
    return
  }
  if (method === 'POST' && action === 'component-status') {
    await setComponentStatus(store, id, req, res)
    return
  }
  if (method === 'POST' && action === 'rename') {
    await renameEnvironment(store, id, req, res)
    return
  }
  if (method === 'POST' && action === 'component-session') {
    await bindComponentSession(store, id, req, res)
    return
  }
  if (method === 'POST' && action === 'sessions') {
    await attachSession(store, id, req, res)
    return
  }
  if (method === 'DELETE' && action === 'sessions') {
    await detachSession(store, id, req, res)
    return
  }

  throw new HttpError(`env-builder: bad path ${path}`, 404)
}
