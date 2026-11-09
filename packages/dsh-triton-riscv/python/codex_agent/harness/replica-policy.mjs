export const name = 'triton-read-only-replica'
export const inject = ['tools', 'systemPrompt']

export function apply(ctx, config) {
  const allowed = new Set(config.tools.map(tool => `mcp__triton_riscv__${tool}`))
  let calls = 0
  ctx.effect(
    () =>
      ctx.tools.guard(exec => {
        if (!allowed.has(exec.name)) return 'Replica capability denied: read-only tools only'
        if (calls >= config.maxToolCalls) return 'Replica tool budget exhausted; summarize available evidence'
      }),
    'replica.read-only',
  )
  ctx.on('tools/execute', async (exec, next) => {
    if (!allowed.has(exec.name) || calls >= config.maxToolCalls) throw new Error('Replica capability/budget denied')
    calls++
    return next()
  })
  ctx.effect(() => ctx.systemPrompt.variable('replica_policy', () => config.systemPrompt), 'replica.variable')
  ctx.effect(
    () => ctx.systemPrompt.section({ name: 'replica', order: 900, text: '{{replica_policy}}', complete: true }),
    'replica.prompt',
  )
}
