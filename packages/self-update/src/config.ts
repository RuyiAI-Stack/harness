import z from '@deepseek-ai/schemastery'

export interface Config {
  /** Repo root containing config.yml, packages/ and tools/scripts/install-all.sh. Defaults to the repo root derived from this package's location. */
  root?: string
  /** Install script relative to the repo root. */
  script: string
  /** Directory under the repo root that holds the self-update log. */
  logDir: string
  /** Allow running the self-update. Defaults to true. */
  enabled: boolean
}

export const Config: z<Config> = z.object({
  root: z.string().default(''),
  script: z.string().default('tools/scripts/install-all.sh'),
  logDir: z.string().default('.dsh'),
  enabled: z.boolean().default(true),
})
