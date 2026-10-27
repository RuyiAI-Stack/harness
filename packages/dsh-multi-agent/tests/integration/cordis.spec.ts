import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { expect, it } from 'vitest'
import * as plugin from '../../index.js'
import { root, python } from '../probe'

it('mounts a real Cordis multiAgent service, resolves injection, and disposes it', async () => {
  const host =
    process.env.DSH_PLUGIN_TEST_HOST ||
    process.env.TRITON_PLUGIN_TEST_HOST ||
    resolve(root, '../../thirdparty/deepseek-harness')
  const require = createRequire(resolve(host, 'package.json'))
  const url = pathToFileURL(require.resolve('./vendor/cordis')).href
  const { Context } = await import(/* @vite-ignore */ url)
  const ctx = new Context()
  const mounted = ctx.plugin(plugin, {
    python,
    model: { provider: 'fixture', model: 'classification-model', apiKeyEnv: 'FIXTURE_API_KEY' },
    tasks: { classify: { systemPrompt: 'Classify mail.', resultSchema: { type: 'object' } } },
  })
  let injected: any
  const consumer = ctx.inject(['multiAgent'], (scope: any) => {
    injected = scope.multiAgent
  })
  try {
    await expect.poll(() => ctx.get('multiAgent')).toBeDefined()
    await expect.poll(() => injected).toBeDefined()
    expect(injected.run).toBeTypeOf('function')
    expect(injected.cancel).toBeTypeOf('function')
    expect(ctx.get('subagents')).toBeUndefined()
    expect(ctx.get('tools')).toBeUndefined()
    await mounted.dispose()
    expect(ctx.get('multiAgent')).toBeUndefined()
    expect(() => injected.run({ task: 'classify', input: {} })).toThrow(/closed/)
  } finally {
    await consumer.dispose()
    await ctx.fiber.dispose()
  }
})
