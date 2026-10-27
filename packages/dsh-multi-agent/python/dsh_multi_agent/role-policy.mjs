export const name = 'multi-agent-isolated-task-policy'
export const inject = ['tools', 'systemPrompt']

export function installPolicy(ctx, config, identity = {}) {
  const variable = identity.variable ?? 'multi_agent_task_policy'
  const section = identity.section ?? 'multi-agent:task'
  const effect = identity.effect ?? 'multi-agent.task'
  if (typeof ctx.tools?.guard !== 'function' || typeof ctx.systemPrompt?.section !== 'function'
      || typeof ctx.systemPrompt?.variable !== 'function')
    throw new Error('Isolated tasks require host tool guards and system prompt sections')
  if (typeof config?.systemPrompt !== 'string' || !config.systemPrompt.trim())
    throw new Error('A role system prompt is required')
  ctx.effect(() => ctx.tools.guard(() => identity.denial ?? 'Isolated tasks cannot execute tools'), `${effect}-tool-guard`)
  ctx.effect(() => ctx.systemPrompt.variable(variable, () => config.systemPrompt), `${effect}-variable`)
  ctx.effect(
    () => ctx.systemPrompt.section({ name: section, order: 900, text: `{{${variable}}}`, complete: true }),
    `${effect}-system`,
  )
  ctx.on('system-prompt/assemble', async (_assembly, _context, next) => {
    const result = await next()
    return { ...result, tools: [] }
  })
}

export function apply(ctx, config) {
  installPolicy(ctx, config, config?.identity)
}
