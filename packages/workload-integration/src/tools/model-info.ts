/**
 * `buckyball_model_info` tool: fetch a HuggingFace model repo snapshot
 * (metadata + file list) via the Hub API.
 *
 * One endpoint (HF_ENDPOINT, default https://huggingface.co), one token
 * (HF_TOKEN), proxy vars (HTTPS_PROXY / ALL_PROXY / NO_PROXY). Any failure throws.
 * @module dsh-workload-integration/tools/model-info
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { EnvHttpProxyAgent, ProxyAgent, Socks5ProxyAgent, fetch as undiciFetch, type Dispatcher } from 'undici'

/** Pick the first non-empty proxy env var. */
function proxyUrl(): string | undefined {
  for (const name of ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy']) {
    const value = process.env[name]?.trim()
    if (value) return value
  }
}

/** Build a dispatcher that honors HTTP(S) and SOCKS proxy env vars. */
function hfDispatcher(): Dispatcher {
  const proxy = proxyUrl()
  if (!proxy) return new EnvHttpProxyAgent()
  const { protocol } = new URL(proxy)
  if (protocol === 'socks5:' || protocol === 'socks:') return new Socks5ProxyAgent(proxy)
  if (protocol === 'http:' || protocol === 'https:') return new ProxyAgent(proxy)
  throw new Error(`unsupported proxy URL protocol ${protocol}: ${proxy}`)
}

const hfAgent = hfDispatcher()

/** Plugin config slice this tool needs. */
export interface ModelInfoConfig {
  /** HuggingFace fetch timeout in milliseconds. */
  fetchTimeoutMs: number
}

/**
 * Extract a HuggingFace model id from a URL or a bare id.
 * @param input - e.g. "https://huggingface.co/google/alexnet" or "google/alexnet".
 * @returns the normalized "<owner>/<name>" or single-segment id.
 */
function parseModelId(input: string): string {
  const trimmed = input
    .trim()
    .replace(/\/+$/, '')
    .replace(/\.git$/, '')
  let path = trimmed
  try {
    path = new URL(trimmed).pathname.replace(/^\/+/, '')
  } catch {
    // A bare owner/model id is the normal non-URL input.
  }
  // Keep "<owner>/<name>"; drop trailing hub paths like /tree/main or /resolve/main/file.
  return path.split('/').slice(0, 2).join('/')
}

/**
 * GET one HuggingFace URL as text.
 * @param url - URL to fetch.
 * @param config - plugin config.
 * @param signal - tool execution cancellation signal.
 * @returns the response body.
 * @throws on any non-2xx status or network failure.
 */
async function fetchText(url: string, config: ModelInfoConfig, signal: AbortSignal): Promise<string> {
  const token = process.env.HF_TOKEN
  const response = await undiciFetch(url, {
    dispatcher: hfAgent,
    signal: AbortSignal.any([signal, AbortSignal.timeout(config.fetchTimeoutMs)]),
    headers: { accept: 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
  })
  if (response.status === 401) {
    throw new Error(
      `HuggingFace returned 401 Authentication required for ${url}: ` +
        'export HF_TOKEN (a Read token) in the dsh launch environment and retry',
    )
  }
  if (response.status === 404) throw new Error(`not found on HuggingFace: ${url}`)
  if (!response.ok) throw new Error(`HuggingFace fetch failed with HTTP ${response.status} for ${url}`)
  return response.text()
}

/**
 * Build the `buckyball_model_info` tool definition.
 * @param config - plugin configuration.
 * @returns the registry-ready tool definition.
 */
export function defineModelInfoTool(config: ModelInfoConfig) {
  return defineTool({
    name: 'buckyball_model_info',
    description:
      '拉取 HuggingFace 模型仓库的 metadata 与文件列表（Hub API）。用户给出模型链接时优先调用。' +
      '接受完整 URL 或裸 model id。请求发往 HF_ENDPOINT（默认 https://huggingface.co）；',
    parameters: {
      model: {
        type: 'string',
        required: true,
        description: 'HuggingFace model id 或完整 URL',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const id = parseModelId(args.model)
      if (!id || id.includes('..')) throw new Error(`cannot parse a HuggingFace model id from: ${args.model}`)
      const endpoint = (process.env.HF_ENDPOINT ?? 'https://huggingface.co').replace(/\/+$/, '')

      const info = JSON.parse(await fetchText(`${endpoint}/api/models/${id}`, config, exec.signal)) as Record<
        string,
        unknown
      >
      const siblings = Array.isArray(info.siblings)
        ? (info.siblings as { rfilename: string }[]).map(sibling => sibling.rfilename)
        : []

      return [
        `model: ${id}`,
        `url: https://huggingface.co/${id}`,
        `pipeline_tag: ${String(info.pipeline_tag ?? 'unknown')}`,
        `library: ${String(info.library_name ?? 'unknown')}`,
        `sha: ${String(info.sha ?? 'unknown')}`,
        `files: ${siblings.join(', ')}`,
      ].join('\n')
    },
  })
}
