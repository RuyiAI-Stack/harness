import { describe, expect, it, vi } from 'vitest'
import { apply } from '../../python/dsh_multi_agent/role-policy.mjs'

function host() {
  const hooks = new Map<string, Function>()
  const disposers: Function[] = []
  const ctx = {
    tools: { guard: vi.fn(() => vi.fn()) },
    systemPrompt: { section: vi.fn(() => vi.fn()), variable: vi.fn(() => vi.fn()) },
    effect: vi.fn((install: Function) => disposers.push(install())),
    on: vi.fn((name: string, hook: Function) => hooks.set(name, hook)),
  }
  return { ctx, hooks, disposers }
}

describe('generic tool-free task policy', () => {
  it.each(['guard', 'section', 'variable'])('fails closed without %s', capability => {
    const { ctx } = host()
    if (capability === 'guard') delete (ctx.tools as any).guard
    else delete (ctx.systemPrompt as any)[capability]
    expect(() => apply(ctx, { systemPrompt: 'Classify mail.' })).toThrow(/require host/)
    expect(ctx.effect).not.toHaveBeenCalled()
  })
  it('denies all tools and removes inherited tool definitions on every assembly', async () => {
    const { ctx, hooks, disposers } = host()
    apply(ctx, { systemPrompt: 'Classify mail.' })
    const guard = (ctx.tools.guard.mock.calls[0] as any)[0]
    for (const name of ['bash', 'write', 'subagent']) expect(guard({ name })).toContain('cannot execute tools')
    const original = { prompt: 'context', tools: [{ name: 'new-tool' }], metadata: { revision: 2 } }
    expect(await hooks.get('system-prompt/assemble')!({}, {}, async () => original)).toEqual({ ...original, tools: [] })
    expect(original.tools).toHaveLength(1)
    expect(ctx.systemPrompt.section).toHaveBeenCalledWith({
      name: 'multi-agent:task',
      order: 900,
      text: '{{multi_agent_task_policy}}',
      complete: true,
    })
    expect(disposers).toHaveLength(3)
    for (const disposer of disposers) expect(disposer).toBeTypeOf('function')
  })
  it('can retain a trusted wrapper policy identity without copying the implementation', () => {
    const { ctx } = host()
    apply(ctx, { systemPrompt: 'Classify mail.', identity: { variable: 'legacy_policy', section: 'legacy:task' } })
    expect(ctx.systemPrompt.section).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'legacy:task', text: '{{legacy_policy}}' }),
    )
  })
})
