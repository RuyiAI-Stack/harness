import { installWorkspaceHost } from './lib/native-workspace.js'
import { resolveConfig } from './lib/config.js'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'

export const name = 'triton-riscv-native-host'
export const inject = ['tools', 'systemPrompt']

export async function apply(ctx, input = {}) {
  const config = resolveConfig(input)
  if (!config.enabled) return
  if (!ctx.baseUrl) throw new Error('Harness config-tree baseUrl is required to resolve the official MCP client')
  const require = createRequire(ctx.baseUrl)
  const mcpPath = require.resolve('@deepseek-ai/dsh-mcp-client')
  const mcp = await import(pathToFileURL(mcpPath).href)
  const scope = await import(pathToFileURL(createRequire(mcpPath).resolve('@deepseek-ai/dsh-scope')).href)
  installWorkspaceHost(ctx, input, { mcp, createScope: scope.createScope })
}
