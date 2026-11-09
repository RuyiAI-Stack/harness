import { OPERATOR_TOOLS } from './bundle/lib/tool-policy.js'

export const name = 'parallel-parent-capabilities'
export const inject = ['tools']

export function apply(ctx, config = {}) {
  const protectedSessions = new Set()
  ctx.effect(
    () =>
      ctx.tools.guard(exec => {
        const session = exec.agent?.session?.id
        if (!session) return 'A host session is required'
        if (exec.name === 'mcp__triton_riscv__prepare_parallel_tasks') protectedSessions.add(session)
        if ((config.typedOnly || protectedSessions.has(session)) && !OPERATOR_TOOLS.has(exec.name))
          return 'Delegation sessions require typed tools; shell/edit/nested generic delegation cannot bypass approval'
      }),
    'parallel.parent-guard',
  )
}
