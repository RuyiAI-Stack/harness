import { describe, expect, it, vi } from 'vitest'
import { installPolicy } from '../../../dsh-multi-agent/python/dsh_multi_agent/role-policy.mjs'

function apply(ctx: any, config: any) {
  installPolicy(ctx, config, {
    variable: 'triton_role_policy',
    section: 'triton:role',
    effect: 'triton.role',
    denial: 'Proposal roles cannot execute tools',
  })
}

function host() {
  const hooks = new Map<string, Function>()
  const disposers: Function[] = []
  const ctx = {
    tools: { guard: vi.fn((_guard: Function) => vi.fn()) },
    systemPrompt: {
      section: vi.fn((_section: object) => vi.fn()),
      variable: vi.fn((_name: string, _value: Function) => vi.fn()),
    },
    effect: vi.fn((install: Function) => disposers.push(install())),
    on: vi.fn((name: string, hook: Function) => hooks.set(name, hook)),
  }
  return { ctx, hooks, disposers }
}

describe('proposal-only role policy', () => {
  it.each(['guard', 'section', 'variable'])('fails closed without host capability %s', capability => {
    const { ctx } = host()
    if (capability === 'guard') delete (ctx.tools as any).guard
    else delete (ctx.systemPrompt as any)[capability]
    expect(() => apply(ctx, { systemPrompt: 'Review the candidate only.' })).toThrow(/require host/)
    expect(ctx.effect).not.toHaveBeenCalled()
  })

  it.each([undefined, '', '   ', 42])('rejects a missing or invalid role prompt: %s', systemPrompt => {
    const { ctx } = host()
    expect(() => apply(ctx, { systemPrompt })).toThrow(/role system prompt/)
    expect(ctx.effect).not.toHaveBeenCalled()
  })

  it('registers the assigned role prompt and lifecycle-owned cleanup', () => {
    const { ctx, disposers } = host()
    apply(ctx, { systemPrompt: 'Only propose a patch. Do not execute it.' })
    expect(ctx.systemPrompt.variable.mock.calls[0][0]).toBe('triton_role_policy')
    expect(ctx.systemPrompt.variable.mock.calls[0][1]()).toBe('Only propose a patch. Do not execute it.')
    expect(ctx.systemPrompt.section).toHaveBeenCalledWith({
      name: 'triton:role',
      order: 900,
      text: '{{triton_role_policy}}',
      complete: true,
    })
    expect(disposers).toHaveLength(3)
    for (const dispose of disposers) {
      expect(dispose).toBeTypeOf('function')
      dispose()
      expect(dispose).toHaveBeenCalledOnce()
    }
  })

  it.each(['bash', 'write', 'delegate', 'mcp__triton_riscv__apply_repair'])('denies direct execution of %s', name => {
    const { ctx } = host()
    apply(ctx, { systemPrompt: 'Propose only.' })
    const guard = ctx.tools.guard.mock.calls[0][0]
    expect(guard({ name })).toBe('Proposal roles cannot execute tools')
  })

  it('removes inherited tools on every assembly without changing other context', async () => {
    const { ctx, hooks } = host()
    apply(ctx, { systemPrompt: 'Review only.' })
    const assemble = hooks.get('system-prompt/assemble')!
    for (const name of ['bash', 'new-tool-from-host']) {
      const original = { prompt: 'Keep task constraints', tools: [{ name }], metadata: { revision: 2 } }
      const next = vi.fn(async () => original)
      const result = await assemble({}, {}, next)
      expect(next).toHaveBeenCalledOnce()
      expect(result).toEqual({ ...original, tools: [] })
      expect(original.tools).toEqual([{ name }])
    }
  })
})
