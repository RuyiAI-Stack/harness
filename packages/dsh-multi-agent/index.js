import { createMultiAgent } from './lib/service.js'

export { createMultiAgent } from './lib/service.js'
export const name = 'multi-agent'
export const inject = []

export function apply(ctx, config = {}) {
  if (config.enabled === false) return
  if (typeof ctx.provide !== 'function' || typeof ctx.effect !== 'function')
    throw new Error('multiAgent requires Cordis service registration and lifecycle effects')
  const service = createMultiAgent(config)
  ctx.provide('multiAgent', service)
  ctx.effect(() => () => service.close(), 'multi-agent.runs')
}
