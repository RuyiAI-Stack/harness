import { Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { createVectorRetrieval } from './lib/service.js'

export { createVectorRetrieval } from './lib/service.js'
export { resolveConfig } from './lib/config.js'

export default class VectorRetrieval extends Service {
  static Config = z.object({
    python: z.string(),
    workspaceId: z.string(),
    generation: z.string(),
    fingerprint: z.string(),
    collection: z.string(),
    qdrantUrl: z.string().default('http://127.0.0.1:6333'),
    dimensions: z.number(),
    apiKeyEnv: z.string().default('DSH_VECTOR_API_KEY'),
    timeoutSeconds: z.number().default(30),
    timeoutMs: z.number().default(120000),
    embedding: z.object({
      provider: z.string().default('none'),
      model: z.string(),
      baseUrl: z.string(),
      apiKeyEnv: z.string().default('DSH_EMBEDDING_API_KEY'),
      tokenizerJson: z.string(),
      tokenBudget: z.number(),
      revision: z.string(),
    }),
  })

  constructor(ctx, config) {
    const client = createVectorRetrieval(config)
    super(ctx, 'vectorRetrieval')
    // Cordis traces service calls through proxies; use scope-bound closures.
    for (const method of ['search', 'index', 'deleteVersion', 'embed']) {
      Object.defineProperty(this, method, { value: client[method], configurable: true })
    }
    ctx.effect(() => () => client.dispose(), 'vector-retrieval.children')
  }
}
