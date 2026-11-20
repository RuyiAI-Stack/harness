import { authFetch } from './auth'

export interface AdminOverview {
  offset?: number
  users: {
    id: string
    username: string
    status: string
    last_used_at: string | null
    session_count: number
    run_count: number
    failed_run_count: number
  }[]
  totals: { users: number; sessions: number; runs: number; failed_runs: number }
}

export interface KnowledgeDocument {
  id: string
  title: string
  source: string
  latest_version: number
  published_version: number | null
  state: string
  chunk_count: number
  error: string | null
  updated_at: string
  deleted_at?: string | null
  catalog_revision?: number
  format?: string
  filename?: string | null
  sync?: KnowledgeSync | null
}

export interface KnowledgeSync {
  state: 'pending' | 'indexing' | 'published' | 'failed' | 'superseded' | 'withdrawn'
  request_id: string
  job_id: string | null
  version: number
  error: string | null
}

export interface KnowledgeDetail extends KnowledgeDocument {
  content: string
  source_file?: {
    filename: string
    format: string
    sha256: string
    parser_version: string
    warnings: string[]
    reviewed: boolean
    preview_available?: boolean
  }
  sections?: KnowledgeSection[]
  chunks: { id: string; text: string; position: number; locator?: string }[]
  versions: { version: number; state: string; chunk_count: number; error: string | null }[]
}

export interface KnowledgeSection {
  id: string
  locator: string
  text: string
  edited: boolean
  bbox?: [number, number, number, number]
  image_size?: [number, number]
  ocr_confidence?: number
}

interface DraftMetadata {
  title: string
  source: string
  document_id?: string
  expected_version?: number
  catalog_revision?: number
}

export type KnowledgeDraft = DraftMetadata &
  ({ content: string; sections?: never } | { sections: { id: string; text: string }[]; content?: never })

export interface KnowledgeSearch {
  status: 'ready' | 'empty' | 'unavailable'
  items: {
    document_id: string
    version: number
    title: string
    source: string
    text: string
    chunk_id: string
    score: number
    locator?: string
  }[]
  context: string
}

export class AdminRequestError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
  }
}

export function adminError(reason: unknown, onForbidden: () => void): string {
  if (reason instanceof Error && reason.name === 'AbortError') return ''
  if (reason instanceof AdminRequestError && reason.status === 403) onForbidden()
  return reason instanceof Error ? reason.message : '请求失败，请重试。'
}

export function adminDate(value: string | null): string {
  if (!value) return '未使用'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN')
}

async function request<T>(path: string, data?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await authFetch(path, {
    signal,
    ...(data === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        }),
  })
  return readResponse<T>(response, path)
}

async function readResponse<T>(response: Response, path: string): Promise<T> {
  if (response.status === 403) throw new AdminRequestError(403, '无权访问管理后台，管理员权限可能已变更。')
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    if (path === '/api/knowledge/search' && response.status === 503 && body?.status === 'unavailable') return body as T
    throw new AdminRequestError(
      response.status,
      typeof body?.detail === 'string' ? body.detail : `请求失败 (${response.status})`,
    )
  }
  return response.json() as Promise<T>
}

const documentPath = (id: string) => `/api/admin/knowledge/${encodeURIComponent(id)}`

async function boundedRequest<T>(path: string, data: unknown, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController()
  let expired = false
  const abort = () => controller.abort()
  if (signal?.aborted) controller.abort()
  signal?.addEventListener('abort', abort, { once: true })
  const timeout = window.setTimeout(
    () => {
      expired = true
      controller.abort()
    },
    data === undefined ? 15000 : 30000,
  )
  try {
    return await request<T>(path, data, controller.signal)
  } catch (error) {
    if (expired)
      throw new Error(
        data === undefined ? '读取更新结果超时，请稍后核对。' : '确认请求超时，结果尚未确认，请先核对结果。',
      )
    throw error
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abort)
  }
}

export const adminApi = {
  overview: (offset = 0, signal?: AbortSignal) =>
    request<AdminOverview>(`/api/admin/overview?${new URLSearchParams({ offset: String(offset) })}`, undefined, signal),
  documents: (
    offset = 0,
    signal?: AbortSignal,
    filters: { q: string; status: string; view: string } = { q: '', status: 'all', view: 'active' },
  ) =>
    request<{ documents: KnowledgeDocument[]; enabled: boolean; offset?: number; total?: number }>(
      `/api/admin/knowledge?${new URLSearchParams({ offset: String(offset), limit: '50', ...filters })}`,
      undefined,
      signal,
    ),
  document: (id: string, signal?: AbortSignal) => request<KnowledgeDetail>(documentPath(id), undefined, signal),
  original: async (id: string, signal?: AbortSignal): Promise<Blob> => {
    const path = `${documentPath(id)}/original`
    const response = await authFetch(path, { signal })
    if (!response.ok) await readResponse<never>(response, path)
    return response.blob()
  },
  save: (draft: KnowledgeDraft, requestId: string, signal?: AbortSignal) =>
    request<KnowledgeDocument>('/api/admin/knowledge', { ...draft, request_id: requestId }, signal),
  confirm: (draft: KnowledgeDraft, requestId: string, signal?: AbortSignal) =>
    boundedRequest<KnowledgeDocument>('/api/admin/knowledge/confirm', { ...draft, request_id: requestId }, signal),
  confirmation: (requestId: string, signal?: AbortSignal) =>
    boundedRequest<KnowledgeDocument>(
      `/api/admin/knowledge/operations/${encodeURIComponent(requestId)}`,
      undefined,
      signal,
    ),
  upload: async (
    file: File,
    metadata: Pick<DraftMetadata, 'title' | 'source'>,
    requestId: string,
    signal?: AbortSignal,
  ) => {
    const invalid = validateSourceFile(file) || validateUploadMetadata(file, metadata)
    if (invalid) throw new Error(invalid)
    const path = `/api/admin/knowledge/files?${new URLSearchParams({
      filename: file.name,
      title: metadata.title,
      source: metadata.source,
      request_id: requestId,
    })}`
    const response = await authFetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file,
      signal,
    })
    return readResponse<KnowledgeDocument>(response, path)
  },
  index: (id: string, version: number, requestId: string, signal?: AbortSignal, revision = 0) =>
    request<KnowledgeDocument>(
      `${documentPath(id)}/index`,
      { version, request_id: requestId, catalog_revision: revision },
      signal,
    ),
  publish: (id: string, version: number, signal?: AbortSignal, revision = 0) =>
    request<KnowledgeDocument>(`${documentPath(id)}/publish`, { version, catalog_revision: revision }, signal),
  unpublish: (id: string, signal?: AbortSignal, revision = 0) =>
    request<KnowledgeDocument>(`${documentPath(id)}/unpublish`, { catalog_revision: revision }, signal),
  lifecycle: (id: string, action: 'delete' | 'restore', version: number, revision: number, signal?: AbortSignal) =>
    request<KnowledgeDocument>(`${documentPath(id)}/${action}`, { version, catalog_revision: revision }, signal),
  search: (query: string, signal?: AbortSignal) =>
    request<KnowledgeSearch>('/api/knowledge/search', { query, limit: 5 }, signal),
}

export const contentByteLimit = 256 * 1024
export const sourceFileByteLimit = 12 * 1024 * 1024
export const sectionCharacterLimit = 20000
export const characterCount = (text: string) => Array.from(text).length
export function validateSourceFile(file: File): string {
  if (!/\.(pdf|docx|png|jpe?g)$/i.test(file.name)) return '仅支持 PDF、DOCX、PNG、JPG/JPEG 文件。'
  if (file.size > sourceFileByteLimit) return '文件不能超过 12 MiB。'
  if (!file.size) return '文件不能为空。'
  return ''
}

export function validateMetadata(draft: Pick<DraftMetadata, 'title' | 'source'>): string {
  if (!draft.title.trim()) return '请输入标题。'
  if (characterCount(draft.title) > 200) return '标题不能超过 200 个字符。'
  if (characterCount(draft.source) > 1000) return '来源不能超过 1000 个字符。'
  return ''
}

export function validateUploadMetadata(file: File, metadata: Pick<DraftMetadata, 'title' | 'source'>): string {
  return validateMetadata({
    ...metadata,
    title: metadata.title.trim() ? metadata.title : file.name.replace(/\.(pdf|docx|png|jpe?g)$/i, ''),
  })
}

export function validateDraft(draft: KnowledgeDraft): string {
  const metadataError = validateMetadata(draft)
  if (metadataError) return metadataError
  if (draft.sections) {
    if (!draft.sections.length || !draft.sections.some(section => section.text.trim())) return '请输入审核正文。'
    if (draft.sections.length > 2000) return '审核正文不能超过 2000 块。'
    if (new Set(draft.sections.map(section => section.id)).size !== draft.sections.length)
      return '审核块 ID 重复，请重新读取。'
    for (const [index, section] of draft.sections.entries()) {
      if (!section.id) return '审核块缺少 ID，请重新读取。'
      if (section.text.includes('\0')) return `第 ${index + 1} 块不能包含空字符。`
      if (characterCount(section.text) > sectionCharacterLimit) return `第 ${index + 1} 块不能超过 20000 个字符。`
    }
    if (draft.sections.reduce((count, section) => count + characterCount(section.text), 0) > 1000000)
      return '审核正文不能超过 1000000 个字符。'
    return ''
  }
  if (!draft.content.trim()) return '请输入正文。'
  if (draft.content.includes('\0')) return '正文不能包含空字符。'
  if (characterCount(draft.content) > 60000) return '正文不能超过 60000 个字符。'
  if (new TextEncoder().encode(draft.content).length > contentByteLimit) return '正文不能超过 256 KB (UTF-8)。'
  return ''
}
