import { readFileSync } from 'node:fs'

const files = ['0-hints', '1-skills', '2-failure-experience', '3-success-experience', '4-verification']
export const PROMPT_SECTIONS = files.map((file, index) => ({
  name: `tool:triton-riscv:${file.slice(2)}`,
  order: 180 + index,
  text: readFileSync(new URL(`../prompts/${file}.md`, import.meta.url), 'utf8').trim(),
}))
export const TRITON_RISCV_SYSTEM_PROMPT = PROMPT_SECTIONS.map(section => section.text).join('\n\n')
