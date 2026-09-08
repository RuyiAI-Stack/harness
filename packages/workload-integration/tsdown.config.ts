import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'tsdown'

const here = dirname(fileURLToPath(import.meta.url))
const require = createRequire(import.meta.url)
const undiciRoot = dirname(
  require.resolve('undici/package.json', {
    paths: [join(here, '../..'), here],
  }),
)

export default defineConfig({
  entry: ['src/index.ts'],
  outDir: 'lib',
  format: ['esm'],
  platform: 'node',
  target: 'es2022',
  dts: true,
  clean: true,
  noExternal: ['undici'],
  alias: { undici: undiciRoot },
})
