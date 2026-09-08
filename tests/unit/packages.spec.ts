import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)))

interface PackageManifest {
  name: string
  main?: string
  types?: string
  dsh?: { bundle?: { patch?: string } }
}

function readJson(path: string): PackageManifest {
  return JSON.parse(readFileSync(resolve(root, path), 'utf8')) as PackageManifest
}

describe('local dsh package contracts', () => {
  const packages = [
    ['packages/env-builder/package.json', 'packages/env-builder/cordis.patch.yml'],
    ['packages/hook-github/package.json', 'packages/hook-github/cordis.patch.yml'],
    ['packages/pr-bot-path/package.json', 'packages/pr-bot-path/cordis.patch.yml'],
    ['packages/hook-Zulip/package.json', 'packages/hook-Zulip/cordis.patch.yml'],
    ['packages/env-sidebar/package.json', 'packages/env-sidebar/cordis.patch.yml'],
    ['packages/collaboration/package.json', 'packages/collaboration/cordis.patch.yml'],
    ['packages/singularity/package.json', 'packages/singularity/cordis.patch.yml'],
    ['packages/workload-integration/package.json', 'packages/workload-integration/cordis.patch.yml'],
  ] as const

  it.each(packages)('%s has a buildable dsh bundle declaration', (manifestPath, patchPath) => {
    const manifest = readJson(manifestPath)
    expect(manifest.name).toMatch(/^@dangosys\/dsh-/)
    expect(manifest.main).toBeTruthy()
    expect(manifest.dsh?.bundle?.patch).toBe('./cordis.patch.yml')
    const patch = readFileSync(resolve(root, patchPath), 'utf8')
    expect(patch).toContain(`name: '${manifest.name}'`)
  })
})
