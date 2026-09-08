import z from '@deepseek-ai/schemastery'

export interface Config {
  fetchTimeoutMs: number
  hfToken: string
  hfEndpoint: string
}

export const Config: z<Config> = z.object({
  fetchTimeoutMs: z.number().default(30000),
  hfToken: z.string().required(),
  hfEndpoint: z.string().default('https://huggingface.co'),
})
