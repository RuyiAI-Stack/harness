import { parseRepoRef } from './parse.ts'

export async function assertGithubRepo(ref: string): Promise<{
  owner: string
  repo: string
  url: string
  dir: string
}> {
  const parsed = parseRepoRef(ref)
  const res = await fetch(`https://api.github.com/repos/${parsed.owner}/${parsed.repo}`, {
    headers: {
      Accept: 'application/vnd.github+json',
      'User-Agent': 'dsh-env-builder',
    },
  })
  if (res.status === 404) {
    throw new Error(`env-builder: github repo not found ${parsed.owner}/${parsed.repo}`)
  }
  if (!res.ok) {
    throw new Error(`env-builder: github lookup failed ${res.status} for ${parsed.owner}/${parsed.repo}`)
  }
  return parsed
}
