// Loaded only by the dedicated proposal-only SDK composition, never the normal UI.
export const name = 'triton-riscv-proposal-role'
export const inject = ['tools', 'systemPrompt']

export function apply(ctx, config) {
  if (typeof ctx.tools?.guard !== 'function' || typeof ctx.systemPrompt?.section !== 'function'
      || typeof ctx.systemPrompt?.variable !== 'function')
    throw new Error('Proposal roles require host tool guards and system prompt sections')
  if (typeof config?.systemPrompt !== 'string' || !config.systemPrompt.trim())
    throw new Error('A role system prompt is required')
  ctx.effect(() => ctx.tools.guard(() => 'Proposal roles cannot execute tools'), 'triton.role-tool-guard')
  ctx.effect(() => ctx.systemPrompt.variable('triton_role_policy', () => config.systemPrompt), 'triton.role-variable')
  ctx.effect(
    () => ctx.systemPrompt.section({ name: 'triton:role', order: 900, text: '{{triton_role_policy}}', complete: true }),
    'triton.role-system',
  )
  // No exposed tools, even if the host preset contributes built-in tools.
  ctx.on('system-prompt/assemble', async (assembly, _context, next) => {
    const result = await next()
    return { ...result, tools: [] }
  })
}
