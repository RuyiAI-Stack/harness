import type { IncomingMessage, ServerResponse } from 'node:http'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Context, Service } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-host-webserver'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { HttpError, statusOf } from './http.ts'
import { EnvStore } from './store.ts'

const HARNESS_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const ENV_ROOT = join(HARNESS_ROOT, 'environment')

declare module '@deepseek-ai/cordis' {
  interface Context {
    envBuilder: EnvBuilder
  }
}

function write(res: ServerResponse, status: number, type: string, body: string) {
  res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' })
  res.end(body)
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(Buffer.from(chunk))
  return Buffer.concat(chunks).toString('utf8')
}

export default class EnvBuilder extends Service {
  static inject = ['tools', 'webServer']
  readonly store = new EnvStore(ENV_ROOT)

  constructor(ctx: Context) {
    super(ctx, 'envBuilder')

    const text = (value: string) => [{ type: 'text' as const, text: value }]

    ctx.tools.register(
      defineTool({
        name: 'env_list',
        description: 'List environments and their GitHub components under environment/projectN.',
        parameters: {},
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async () => JSON.stringify(this.store.list(), null, 2),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_create',
        description: 'Create an empty environment directory environment/projectN and register it.',
        parameters: {},
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async () => JSON.stringify(this.store.create()),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_add_component',
        description: 'Clone owner/repo or GitHub URL into an environment as a component.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id, e.g. project1' },
          repo: { type: 'string', required: true, description: 'owner/repo or https://github.com/owner/repo' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => await this.store.addComponent(args.env, args.repo),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_remove_component',
        description: 'Remove a GitHub component (owner/name) from an environment.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
          repo: { type: 'string', required: true, description: 'owner/name' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => {
          this.store.removeComponent(args.env, args.repo)
          return `removed ${args.repo} from ${args.env}`
        },
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_set_component_status',
        description: 'Set component status: ready | modified | installing.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
          repo: { type: 'string', required: true, description: 'owner/name' },
          status: { type: 'string', required: true, description: 'ready | modified | installing' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.setComponentStatus(args.env, args.repo, args.status)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_select',
        description: 'Select the active environment for repository and workload tools.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.select(args.env)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_select_repo',
        description: 'Select the active owner/name repository within the selected environment.',
        parameters: {
          repo: { type: 'string', required: true, description: 'owner/name, e.g. DangoSys/buckyball' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => this.store.selectRepo(args.repo),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_ensure_component',
        description: 'Verify the selected repository component is cloned in the selected environment.',
        parameters: {
          name: { type: 'string', required: true, description: 'Repository name, e.g. buckyball' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => this.store.ensureComponent(args.name),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_bind_component_session',
        description: 'Bind a session id to a component; throws if missing or already bound.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
          repo: { type: 'string', required: true, description: 'owner/name' },
          sessionId: { type: 'string', required: true, description: 'Session id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.bindComponentSession(args.env, args.repo, args.sessionId)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_attach_session',
        description: 'Attach a session id to an environment session tree.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
          sessionId: { type: 'string', required: true, description: 'Session id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.attachSession(args.env, args.sessionId)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_detach_session',
        description: 'Detach a session id from an environment.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
          sessionId: { type: 'string', required: true, description: 'Session id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.detachSession(args.env, args.sessionId)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_rename',
        description: 'Rename an environment id; throws if missing or conflict.',
        parameters: {
          env: { type: 'string', required: true, description: 'Current environment id' },
          id: { type: 'string', required: true, description: 'New environment id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.rename(args.env, args.id)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_set_running',
        description: 'Set whether an environment is running (Live indicator).',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
          running: { type: 'boolean', required: true, description: 'true if running' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => JSON.stringify(this.store.setRunning(args.env, args.running)),
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_reset',
        description: 'git reset --hard && git clean -fd for every component in the environment.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => {
          this.store.reset(args.env)
          return `reset ${args.env}`
        },
      }),
    )

    ctx.tools.register(
      defineTool({
        name: 'env_delete',
        description: 'Delete an environment directory and remove it from the manifest.',
        parameters: {
          env: { type: 'string', required: true, description: 'Environment id' },
        },
        output: { schema: { type: 'string' }, render: (_a, v) => text(v) },
        execute: async args => {
          this.store.delete(args.env)
          return `deleted ${args.env}`
        },
      }),
    )

    ctx.effect(
      () =>
        ctx.webServer.register({
          kind: 'prefix',
          path: '/integrations/env-builder',
          handler: async (req, res) => {
            try {
              await this.handle(req, res)
            } catch (err) {
              const message = err instanceof Error ? err.message : String(err)
              write(res, statusOf(err), 'text/plain; charset=utf-8', message)
            }
          },
        }),
      'env-builder: api',
    )
  }

  private async handle(req: IncomingMessage, res: ServerResponse) {
    const url = new URL(req.url!, 'http://local')
    const base = '/integrations/env-builder'
    const path = url.pathname
    const method = req.method ?? 'GET'

    if (method === 'GET' && path === `${base}/environments`) {
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.list()))
      return
    }

    if (method === 'GET' && path === `${base}/selected`) {
      const env = this.store.selected()
      const repo = this.store.selectedRepo()
      write(
        res,
        200,
        'application/json; charset=utf-8',
        JSON.stringify(env ? { id: env.id, repo, running: env.running, sessionIds: env.sessionIds } : null),
      )
      return
    }

    if (method === 'POST' && path === `${base}/selected-repo`) {
      let body: { repo?: string }
      try {
        body = JSON.parse(await readBody(req)) as { repo?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify({ repo: this.store.selectRepo(body.repo) }))
      return
    }

    if (method === 'POST' && path === `${base}/environments`) {
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.create()))
      return
    }

    const m = path.match(
      new RegExp(
        `^${base}/environments/([^/]+)(?:/(components|component-status|component-session|sessions|reset|select|running|rename))?$`,
      ),
    )
    if (!m) throw new HttpError(`env-builder: bad path ${path}`, 404)
    const id = decodeURIComponent(m[1])
    const action = m[2]

    if (method === 'DELETE' && !action) {
      this.store.delete(id)
      write(res, 200, 'text/plain; charset=utf-8', `deleted ${id}`)
      return
    }

    if (method === 'POST' && action === 'select') {
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.select(id)))
      return
    }

    if (method === 'POST' && action === 'reset') {
      this.store.reset(id)
      write(res, 200, 'text/plain; charset=utf-8', `reset ${id}`)
      return
    }

    if (method === 'POST' && action === 'running') {
      let body: { running?: unknown }
      try {
        body = JSON.parse(await readBody(req)) as { running?: unknown }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (typeof body.running !== 'boolean') throw new HttpError('env-builder: running must be boolean', 400)
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.setRunning(id, body.running)))
      return
    }

    if (method === 'POST' && action === 'components') {
      let body: { repo?: string }
      try {
        body = JSON.parse(await readBody(req)) as { repo?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
      const dir = await this.store.addComponent(id, body.repo)
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify({ path: dir }))
      return
    }

    if (method === 'DELETE' && action === 'components') {
      let body: { repo?: string }
      try {
        body = JSON.parse(await readBody(req)) as { repo?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
      this.store.removeComponent(id, body.repo)
      write(res, 200, 'text/plain; charset=utf-8', `removed ${body.repo}`)
      return
    }

    if (method === 'POST' && action === 'component-status') {
      let body: { repo?: string; status?: unknown }
      try {
        body = JSON.parse(await readBody(req)) as { repo?: string; status?: unknown }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
      write(
        res,
        200,
        'application/json; charset=utf-8',
        JSON.stringify(this.store.setComponentStatus(id, body.repo, body.status)),
      )
      return
    }

    if (method === 'POST' && action === 'rename') {
      let body: { id?: string }
      try {
        body = JSON.parse(await readBody(req)) as { id?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.id) throw new HttpError('env-builder: missing new environment id', 400)
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.rename(id, body.id)))
      return
    }

    if (method === 'POST' && action === 'component-session') {
      let body: { repo?: string; sessionId?: string }
      try {
        body = JSON.parse(await readBody(req)) as { repo?: string; sessionId?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.repo) throw new HttpError('env-builder: missing repo', 400)
      if (!body.sessionId) throw new HttpError('env-builder: missing session id', 400)
      write(
        res,
        200,
        'application/json; charset=utf-8',
        JSON.stringify(this.store.bindComponentSession(id, body.repo, body.sessionId)),
      )
      return
    }

    if (method === 'POST' && action === 'sessions') {
      let body: { sessionId?: string }
      try {
        body = JSON.parse(await readBody(req)) as { sessionId?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.sessionId) throw new HttpError('env-builder: missing session id', 400)
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.attachSession(id, body.sessionId)))
      return
    }

    if (method === 'DELETE' && action === 'sessions') {
      let body: { sessionId?: string }
      try {
        body = JSON.parse(await readBody(req)) as { sessionId?: string }
      } catch {
        throw new HttpError('env-builder: invalid json body', 400)
      }
      if (!body.sessionId) throw new HttpError('env-builder: missing session id', 400)
      write(res, 200, 'application/json; charset=utf-8', JSON.stringify(this.store.detachSession(id, body.sessionId)))
      return
    }

    throw new HttpError(`env-builder: bad path ${path}`, 404)
  }
}
