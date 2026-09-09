export function hintsPromptText(): string {
  return `# env-builder hints

Repository checkout is ready. Before build or install:

1. In the checkout you just cloned, list its \`.agents/skills\` directory — skills are \`<name>/SKILL.md\` directories or \`<name>.md\` files.
2. Read each skill frontmatter \`description\` / \`whenToUse\`. If one covers building or developing this repo, call \`skill({ name: "<exact-name>" })\` and follow it for build steps.
3. If no build skill applies, use the repository README, run bash build, then \`env_register_component\`.

Do not skip skill discovery when \`.agents/skills\` exists and lists a build-related skill.`
}
