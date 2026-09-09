/**
 * Self-update the harness: run the repo install script in the background so the
 * agent can rebuild and re-register plugins (online evolution).
 * @module dsh-self-update
 */

import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { Config } from './config.ts'
import { InstallRunner } from './service/runner.ts'
import { defineStartTool } from './tools/api/start.ts'
import { defineStatusTool } from './tools/api/status.ts'

export { Config } from './config.ts'
export { InstallRunner } from './service/runner.ts'
export type { RunPhase, RunState, RunnerOptions } from './service/runner.ts'
export { readTailFile, tailText } from './tools/libs/tail.ts'

// lib/index.js lives at packages/self-update/lib/, so ../../.. is the repo root.
const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')

declare module '@deepseek-ai/cordis' {
  interface Context {
    selfUpdate: SelfUpdate
  }
}

export default class SelfUpdate extends Service {
  static inject = ['tools']
  static Config: z<Config> = Config

  declare readonly config: Config
  private readonly runner: InstallRunner

  constructor(ctx: Context, config: Config) {
    super(ctx, 'selfUpdate')
    this.config = config

    const root = config.root || REPO_ROOT
    this.runner = new InstallRunner({
      script: resolve(root, config.script),
      cwd: root,
      logPath: resolve(root, config.logDir, 'self-update.log'),
    })

    ctx.tools.register(defineStartTool(this.runner, config.enabled !== false))
    ctx.tools.register(defineStatusTool(this.runner))

    ctx.on('dispose', () => this.runner.kill())
  }
}
