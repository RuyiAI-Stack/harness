import { defineConfig } from 'vitest/config'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const alias = Object.fromEntries(
  ['cordis', 'schemastery', 'cosmokit'].flatMap(name => {
    const source = new URL(`../../thirdparty/deepseek-harness/vendor/${name}/src/index.ts`, import.meta.url)
    return existsSync(source) ? [[`@deepseek-ai/${name}`, fileURLToPath(source)]] : []
  }),
)

export default defineConfig({
  resolve: { alias },
  test: { include: ['tests/**/*.spec.ts'], environment: 'node', testTimeout: 30000 },
})
