export function cleanPromptText(env: string): string {
  return `# env-builder clean (${env})

You must prepare environment \`${env}\` so it can be rebound to a new Singularity graph.

Goals:
1. Detach every session still attached to \`${env}\` (\`env_list\` / \`env_detach_session\`).
2. For each registered GitHub component whose checkout exists under the env path: reset it to a clean git tree (\`env_reset\` and/or \`git -C <dir> reset --hard\` + \`git clean -fd\`). If the checkout directory is missing, remove that component from the manifest with \`env_remove_component\` — do not invent paths.
3. Delete junk under the env directory that is not a managed component checkout (build artifacts, scratch files, orphan folders). Keep the env root directory itself.
4. When the env is rebindable, call \`env_mark_clean({ env: "${env}" })\` exactly once. That call is the only success signal.

Rules:
- Do not delete the environment id/directory (\`env_delete\` is forbidden for this task).
- Missing component directories are expected; fix the manifest, do not fail the whole clean.
- Do not claim success until \`env_mark_clean\` succeeds.`
}
