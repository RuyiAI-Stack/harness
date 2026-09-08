export function parseRepoRef(input: string): {
  owner: string
  repo: string
  url: string
  dir: string
} {
  const raw = input.trim().replace(/\.git$/, '')
  const urlMatch = raw.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/?$/i)
  const shortMatch = raw.match(/^([^/]+)\/([^/]+)$/)
  let owner: string
  let repo: string
  if (urlMatch) {
    owner = urlMatch[1]
    repo = urlMatch[2]
  } else if (shortMatch) {
    owner = shortMatch[1]
    repo = shortMatch[2]
  } else {
    throw new Error(`env-builder: invalid repo ref "${input}"`)
  }
  if (owner === '.' || owner === '..' || repo === '.' || repo === '..') {
    throw new Error(`env-builder: invalid repo ref "${input}"`)
  }
  return {
    owner,
    repo,
    url: `https://github.com/${owner}/${repo}.git`,
    dir: `${owner}/${repo}`,
  }
}
