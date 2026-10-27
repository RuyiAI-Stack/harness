import { Context, Service } from '@deepseek-ai/cordis'

export interface EmbeddingConfig {
  provider?: 'none' | 'sentence-transformers' | 'openai-compatible' | 'ollama'
  model?: string
  baseUrl?: string
  apiKeyEnv?: string
  tokenizerJson?: string
  tokenBudget?: number
  revision?: string
}

export interface VectorConfig {
  python: string
  workspaceId: string
  generation: string
  fingerprint: string
  collection: string
  dimensions: number
  qdrantUrl?: string
  apiKeyEnv?: string
  timeoutSeconds?: number
  timeoutMs?: number
  embedding?: EmbeddingConfig
}

export interface DocumentChunk {
  document_id: string
  version: number
  chunk_id: string
  source_version: string
  text: string
  metadata?: Record<string, unknown>
}

export interface VectorHit {
  id: string
  score: number
  payload: DocumentChunk & { workspace_id: string; generation: string; index_fingerprint: string }
}

export interface VectorRetrievalApi {
  search(
    query: { text: string } | { vector: number[] },
    options?: {
      limit?: number
      offset?: number
      signal?: AbortSignal
    },
  ): Promise<VectorHit[]>
  index(chunks: DocumentChunk[], options?: { vectors?: number[][]; signal?: AbortSignal }): Promise<string[]>
  deleteVersion(
    documentId: string,
    version: number,
    options?: {
      older?: boolean
      include?: boolean
      signal?: AbortSignal
    },
  ): Promise<{ completed: true }>
  embed(texts: string[], options?: { signal?: AbortSignal }): Promise<number[][]>
}

export function resolveConfig(input: VectorConfig): Readonly<VectorConfig>
export function createVectorRetrieval(input: VectorConfig): VectorRetrievalApi & { dispose(): void }

export default class VectorRetrieval extends Service implements VectorRetrievalApi {
  static Config: (input: VectorConfig) => VectorConfig
  constructor(ctx: Context, config: VectorConfig)
  search: VectorRetrievalApi['search']
  index: VectorRetrievalApi['index']
  deleteVersion: VectorRetrievalApi['deleteVersion']
  embed: VectorRetrievalApi['embed']
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    vectorRetrieval: VectorRetrievalApi
  }
}
