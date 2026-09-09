export function verifyPromptText(env: string, repo: string): string {
  const name = repo.split('/')[1]
  return `# env-builder verify (${env} / ${repo})

Install step finished. Before telling the user you are done, run this checklist and fix anything that fails.

1. \`env_list()\` — \`${env}\` lists \`${repo}\` with status \`ready\`.
2. \`env_select({ id: "${env}" })\` then \`env_select_repo({ repo: "${repo}" })\`.
3. \`env_ensure_component({ name: "${name}" })\` — returns the checkout path; must not throw.
4. In that checkout: \`.git\` exists and \`git remote get-url origin\` matches the registered URL.

If any step fails, fix the install and call \`env_register_component\` again. Do not claim success until all four pass.`
}
