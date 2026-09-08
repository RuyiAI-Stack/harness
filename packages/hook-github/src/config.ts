import z from '@deepseek-ai/schemastery'

export interface Config {
  secretFile: string
  path: string
}

export const Config: z<Config> = z.object({
  secretFile: z.string().required(),
  path: z.string().default('/integrations/hook-github'),
})
