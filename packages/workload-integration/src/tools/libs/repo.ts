import type { Context } from '@deepseek-ai/cordis'

export function selectedRepoPath(ctx: Context): string {
  const repo = ctx.envBuilder.store.selectedRepo()
  if (repo === undefined) throw new Error('workload integration requires a selected repository')
  if (!ctx.role.repoList.includes(repo)) throw new Error('selected repository is not in the repoList: ' + repo)
  const separator = repo.indexOf('/')
  if (separator <= 0 || separator === repo.length - 1)
    throw new Error('selected repository must use owner/name format: ' + repo)
  return ctx.envBuilder.store.ensureComponent(repo.slice(separator + 1))
}
