/**
 * Ephemeral environments and GitHub component install.
 * @module dsh-env-builder
 */

import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Context, Service } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-host-webserver'
import { PATH } from './constants.ts'
import { EnvStore } from './service/store.ts'
import { defineAttachSessionTool } from './tools/api/attach-session.ts'
import { defineBindComponentSessionTool } from './tools/api/bind-component-session.ts'
import { defineDeleteTool } from './tools/api/delete.ts'
import { defineDetachSessionTool } from './tools/api/detach-session.ts'
import { defineEnsureComponentTool } from './tools/api/ensure-component.ts'
import { defineListTool } from './tools/api/list.ts'
import { defineRegisterComponentTool } from './tools/api/register-component.ts'
import { defineRemoveComponentTool } from './tools/api/remove-component.ts'
import { defineRenameTool } from './tools/api/rename.ts'
import { defineResetTool } from './tools/api/reset.ts'
import { defineSelectTool } from './tools/api/select.ts'
import { defineSelectRepoTool } from './tools/api/select-repo.ts'
import { defineSetComponentStatusTool } from './tools/api/set-component-status.ts'
import { defineSetRunningTool } from './tools/api/set-running.ts'
import { handleRequest } from './web/api/request.ts'
import { statusOf, write } from './web/libs/http.ts'

export { EnvStore } from './service/store.ts'
export type { ComponentStatus, EnvComponent, EnvManifest, EnvRecord } from './service/types.ts'
export { HttpError, statusOf } from './web/libs/http.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    envBuilder: EnvBuilder
  }
}

const ENV_ROOT = join(resolve(dirname(fileURLToPath(import.meta.url)), '../../..'), 'environment')

export default class EnvBuilder extends Service {
  static inject = ['tools', 'webServer']
  readonly store = new EnvStore(ENV_ROOT)

  constructor(ctx: Context) {
    super(ctx, 'envBuilder')

    ctx.tools.register(defineListTool(this.store))
    ctx.tools.register(defineRegisterComponentTool(this.store))
    ctx.tools.register(defineRemoveComponentTool(this.store))
    ctx.tools.register(defineSetComponentStatusTool(this.store))
    ctx.tools.register(defineSelectTool(this.store))
    ctx.tools.register(defineSelectRepoTool(this.store))
    ctx.tools.register(defineEnsureComponentTool(this.store))
    ctx.tools.register(defineBindComponentSessionTool(this.store))
    ctx.tools.register(defineAttachSessionTool(this.store))
    ctx.tools.register(defineDetachSessionTool(this.store))
    ctx.tools.register(defineRenameTool(this.store))
    ctx.tools.register(defineSetRunningTool(this.store))
    ctx.tools.register(defineResetTool(this.store))
    ctx.tools.register(defineDeleteTool(this.store))

    ctx.effect(
      () =>
        ctx.webServer.register({
          kind: 'prefix',
          path: PATH,
          handler: async (req, res) => {
            try {
              await handleRequest(this.store, req, res)
            } catch (err) {
              const message = err instanceof Error ? err.message : String(err)
              write(res, statusOf(err), 'text/plain; charset=utf-8', message)
            }
          },
        }),
      'env-builder: api',
    )
  }
}
