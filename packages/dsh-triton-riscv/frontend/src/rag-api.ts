import { AdminRequestError, KnowledgeDocument, adminError } from './admin-api'
import { authFetch } from './auth'

export interface RagDataSource {
  id: string
  label: string
  read_only: boolean
  status: 'configured'
  kind: string
}
export interface RagDataSources {
  default_source_id: string
  sources: RagDataSource[]
  origin_filters?: { id: string; label: string }[]
}
export interface RagScope {
  id: string
  label: string
  kind?: string
}
export interface RagSourceHealth {
  scope_id: string
  source_kind: string
  status: string
  reason?: string
}
export interface RagItem {
  item_id: string
  source_kind: 'knowledge_document' | 'memory_case'
  source_id: string
  scope: RagScope
  title: string
  excerpt: string
  source: string
  format: string
  latest_version: string
  serving_version: string | null
  content_state: string
  lifecycle_state: string
  index_state: string
  retrieval_state: string
  retrieval_reason: string
  cleanup_state: string
  updated_at: string
  verified_at: string | null
  revision: string
  capabilities: string[]
  chunk_count: number
  filename?: string | null
}
export interface RagCompleteness {
  complete: boolean
  source_health: RagSourceHealth[]
  as_of: string
  catalog_revision: string
  scopes: RagScope[]
}
export interface RagCatalogPage extends RagCompleteness {
  items: RagItem[]
  total: number
  next_cursor: string | null
}
export interface RagSummary extends RagCompleteness {
  total: number
  eligible: number
  pending: number
  attention: number
  excluded: number
}
export interface RagSection {
  id: string
  label: string
  text: string
  locator?: string
}
export interface RagDetail extends RagItem {
  selected_version: string
  selected_title?: string
  selected_source?: string
  content: string
  sections: RagSection[]
  source_file: null | {
    filename: string
    format: string
    sha256?: string
    reviewed?: boolean
    warnings?: string[]
    download_available?: boolean
    preview_available?: boolean
  }
  provenance: Record<string, unknown>
  corrections?: { id: string; content: string; created_at: string; actor_id?: string }[]
  shared_document_id?: string | null
}
export interface RagVersion {
  version: string
  state: string
  chunk_count: number
  is_latest: boolean
  is_serving: boolean
  created_at: string
}
export interface RagVersions {
  items: RagVersion[]
  next_cursor?: string | null
}
export interface RagChunk {
  id: string
  text: string
  position: number
  locator?: string
  generation?: string | number
  revision?: string | number
  kind?: string
}
export interface RagChunks {
  items: RagChunk[]
  next_cursor: string | null
  version: string
}
export interface RagFilters {
  q: string
  source_kind: string
  scope_id: string
  status: string
  view: string
  origin?: string
}
export interface RagVectorCheck {
  version: string
  checked_at: string
  complete: boolean
  vector_scan_performed: boolean
  reason?: string
  checks: {
    generation: string
    expected_count: number
    actual_count: number | null
    dimensions: number | null
    state: string
    reason?: string
  }[]
}
export interface RagAction {
  action: 'archive' | 'correct'
  expected_revision: string
  request_id: string
  reason?: string
  content?: string
}
export interface RagOperation {
  status: 'committed' | 'unknown'
  item_id?: string
  action?: string
}
export interface RagIndexAnomaly {
  source_kind: string
  source_id: string
  scope: RagScope
  reason: string
  index_entries: number
  index_chunks: number
}
export interface RagReconciliation {
  index_anomalies: RagIndexAnomaly[]
  index_anomalies_total: number
  index_anomalies_next_cursor: string | null
  index_scan_complete: boolean
  index_source_health: RagSourceHealth[]
  vector_scan_performed: boolean
  as_of: string
}

export const initialRagFilters: RagFilters = {
  q: '',
  source_kind: 'all',
  scope_id: 'all',
  status: 'all',
  view: 'active',
}
const base = '/api/admin/rag/catalog'
const itemPath = (id: string) => `${base}/${encodeURIComponent(id)}`
class RagSourceAccessError extends AdminRequestError {}
export function ragError(reason: unknown, onForbidden: () => void): string {
  return reason instanceof RagSourceAccessError ? reason.message : adminError(reason, onForbidden)
}

// Every read is bounded. Mutation timeouts mean an unknown outcome, never permission to replay.
async function request<T>(path: string, signal?: AbortSignal, data?: unknown, blob = false): Promise<T> {
  const controller = new AbortController()
  let timedOut = false
  const abort = () => controller.abort()
  if (signal?.aborted) controller.abort()
  signal?.addEventListener('abort', abort, { once: true })
  const timeout = window.setTimeout(
    () => {
      timedOut = true
      controller.abort()
    },
    data ? 30000 : 15000,
  )
  try {
    const response = await authFetch(path, {
      signal: controller.signal,
      ...(data === undefined
        ? {}
        : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
    })
    if (!response.ok) {
      const body = await response.json().catch(() => null)
      if (response.status === 403 && body?.error_code === 'rag_source_forbidden') {
        throw new RagSourceAccessError(
          403,
          typeof body.detail === 'string' ? body.detail : '所选资料库暂不可访问，请选择其他资料库。',
        )
      }
      throw new AdminRequestError(
        response.status,
        response.status === 403
          ? '无权访问管理后台，管理员权限可能已变更。'
          : response.status === 409
            ? '资料或访问范围已经变化，请重新读取后再操作。'
            : typeof body?.detail === 'string'
              ? body.detail
              : `资料请求失败 (${response.status})`,
      )
    }
    return (blob ? await response.blob() : await response.json()) as T
  } catch (error) {
    if (timedOut) throw new Error(data ? '操作响应超时，结果尚未确认。' : '读取资料超时，请稍后重试。')
    throw error
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abort)
  }
}

function query(filters: RagFilters, cursor?: string | null) {
  return new URLSearchParams({ ...filters, origin: filters.origin || '', limit: '25', ...(cursor ? { cursor } : {}) })
}
function inSource(path: string, sourceId: string) {
  return `${path}${path.includes('?') ? '&' : '?'}${new URLSearchParams({ source_id: sourceId })}`
}
export const ragApi = {
  sources: (signal?: AbortSignal) => request<RagDataSources>('/api/admin/rag/sources', signal),
  catalog: (filters: RagFilters, cursor?: string | null, signal?: AbortSignal, sourceId = 'current') =>
    request<RagCatalogPage>(inSource(`${base}?${query(filters, cursor)}`, sourceId), signal),
  summary: (filters: RagFilters, signal?: AbortSignal, sourceId = 'current') =>
    request<RagSummary>(inSource(`${base}/summary?${query(filters)}`, sourceId), signal),
  detail: (id: string, version?: string | null, signal?: AbortSignal, sourceId = 'current') =>
    request<RagDetail>(
      inSource(`${itemPath(id)}${version ? `?${new URLSearchParams({ version })}` : ''}`, sourceId),
      signal,
    ),
  versions: (id: string, signal?: AbortSignal, cursor?: string | null, sourceId = 'current') =>
    request<RagVersions>(
      inSource(`${itemPath(id)}/versions${cursor ? `?${new URLSearchParams({ cursor })}` : ''}`, sourceId),
      signal,
    ),
  vectors: (id: string, version: string, signal?: AbortSignal, sourceId = 'current') =>
    request<RagVectorCheck>(inSource(`${itemPath(id)}/vectors?${new URLSearchParams({ version })}`, sourceId), signal),
  chunks: (id: string, version: string, cursor?: string | null, signal?: AbortSignal, sourceId = 'current') =>
    request<RagChunks>(
      inSource(
        `${itemPath(id)}/chunks?${new URLSearchParams({ version, limit: '25', ...(cursor ? { cursor } : {}) })}`,
        sourceId,
      ),
      signal,
    ),
  original: (id: string, version: string, signal?: AbortSignal, sourceId = 'current') =>
    request<Blob>(
      inSource(`${itemPath(id)}/original?${new URLSearchParams({ version })}`, sourceId),
      signal,
      undefined,
      true,
    ),
  action: (id: string, action: RagAction, signal?: AbortSignal, sourceId = 'current') =>
    request<RagDetail>(inSource(`${itemPath(id)}/actions`, sourceId), signal, action),
  share: (
    id: string,
    action: { expected_revision: string; request_id: string },
    signal?: AbortSignal,
    sourceId = 'current',
  ) => request<KnowledgeDocument>(inSource(`${itemPath(id)}/share`, sourceId), signal, action),
  operation: (id: string, itemId: string, signal?: AbortSignal, sourceId = 'current') =>
    request<RagOperation>(
      inSource(
        `/api/admin/rag/operations/${encodeURIComponent(id)}?${new URLSearchParams({ item_id: itemId })}`,
        sourceId,
      ),
      signal,
    ),
  reconcile: (scopeId: string, cursor?: string | null, signal?: AbortSignal, sourceId = 'current') =>
    request<RagReconciliation>(
      inSource(
        `${base}/reconciliation?${new URLSearchParams({ scope_id: scopeId, limit: '25', ...(cursor ? { index_cursor: cursor } : {}) })}`,
        sourceId,
      ),
      signal,
    ),
}
