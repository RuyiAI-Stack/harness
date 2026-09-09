import z from '@deepseek-ai/schemastery'

export const DEFAULT_TITLE = 'Agent execution report: {session}'

export interface Config {
  /** github-bot org key owning the GitHub App installation used for auth. */
  org?: string
  /** Target repository in owner/name form. */
  repo?: string
  /** Issue title; the `{session}` placeholder is replaced with the session id. */
  title?: string
  /** File the report issue on session dispose. Defaults to true. */
  enabled?: boolean
}

export const Config: z<Config> = z.object({
  org: z.string().default(''),
  repo: z.string().default(''),
  title: z.string().default(DEFAULT_TITLE),
  enabled: z.boolean().default(true),
})
