import { copyFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')
mkdirSync(join(root, 'lib'), { recursive: true })
copyFileSync(join(root, 'src/client.js'), join(root, 'lib/client.js'))
