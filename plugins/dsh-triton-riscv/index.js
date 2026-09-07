import { readFileSync } from 'node:fs'

export const name = 'triton-riscv-domain-policy'

export const inject = ['systemPrompt']

export const TRITON_RISCV_SYSTEM_PROMPT = readFileSync(
  new URL('./policy.md', import.meta.url),
  'utf8',
).trim()

export function apply(ctx) {
  ctx.effect(
    () => ctx.systemPrompt.section({
      name: 'tool:triton-riscv',
      order: 180,
      text: TRITON_RISCV_SYSTEM_PROMPT,
    }),
    'triton-riscv.system-prompt',
  )
}
