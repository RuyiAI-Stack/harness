import { Context } from '@deepseek-ai/cordis'
import { expect, it } from 'vitest'
import VectorRetrieval from '../../index.js'

it('registers a callable Cordis service and cleans up with its owning plugin', async () => {
  const ctx = new Context()
  const config = VectorRetrieval.Config({
    python: '/not-present/python',
    workspaceId: 'support-a',
    generation: 'faq-v1',
    fingerprint: 'fp',
    collection: 'faq',
    dimensions: 4,
  })
  const fiber = await ctx.plugin(VectorRetrieval, config)
  try {
    const service = ctx.get('vectorRetrieval')
    expect(service).toBeTruthy()
    for (const method of ['search', 'index', 'deleteVersion', 'embed']) expect(typeof service[method]).toBe('function')
    await expect(service.search({ vector: [1, 0, 0, 0] })).rejects.toThrow('Vector retrieval operation failed')
    await fiber.dispose()
    expect(ctx.get('vectorRetrieval', false)).toBeUndefined()
  } finally {
    await ctx.fiber.dispose()
  }
})
