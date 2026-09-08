import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)))

interface PackageManifest {
  name: string
  main?: string
  dependencies?: Record<string, string>
  dsh?: { bundle?: { patch?: string } }
}

function readJson(abs: string): PackageManifest {
  return JSON.parse(readFileSync(abs, 'utf8')) as PackageManifest
}

function discoverBundles(): { manifestPath: string; patchPath: string; manifest: PackageManifest }[] {
  const packagesDir = resolve(root, 'packages')
  const manifestPaths: string[] = []
  for (const entry of readdirSync(packagesDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue
    const top = join(packagesDir, entry.name, 'package.json')
    if (existsSync(top)) manifestPaths.push(top)
    for (const child of readdirSync(join(packagesDir, entry.name), { withFileTypes: true })) {
      if (!child.isDirectory()) continue
      const nested = join(packagesDir, entry.name, child.name, 'package.json')
      if (existsSync(nested)) manifestPaths.push(nested)
    }
  }

  const bundles = []
  for (const abs of manifestPaths) {
    const manifest = readJson(abs)
    if (!manifest.dsh?.bundle?.patch) continue
    if (manifest.dsh.bundle.patch !== './cordis.patch.yml') {
      throw new Error(
        `${relative(root, abs)}: dsh.bundle.patch must be ./cordis.patch.yml, got ${manifest.dsh.bundle.patch}`,
      )
    }
    const patchAbs = join(abs, '..', 'cordis.patch.yml')
    if (!existsSync(patchAbs)) throw new Error(`missing ${relative(root, patchAbs)}`)
    bundles.push({
      manifestPath: relative(root, abs),
      patchPath: relative(root, patchAbs),
      manifest,
    })
  }
  if (bundles.length === 0) throw new Error('no dsh bundle packages under packages/')
  return bundles
}

describe('local dsh package contracts', () => {
  const bundles = discoverBundles()

  it.each(bundles.map(b => [b.manifestPath, b.patchPath, b.manifest] as const))(
    '%s has a buildable dsh bundle declaration',
    (manifestPath, patchPath, manifest) => {
      expect(manifest.name).toMatch(/^@[^/]+\/dsh-/)
      expect(manifest.main).toBeTruthy()
      const patch = readFileSync(resolve(root, patchPath), 'utf8')
      expect(patch).toContain(`name: '${manifest.name}'`)
    },
  )

  it('osc meta-bundle patch inserts every osc plugin dependency', () => {
    const meta = bundles.find(b => b.manifest.name === '@ruyiAi/dsh-osc')
    if (!meta) throw new Error('missing @ruyiAi/dsh-osc package')
    if (!meta.manifest.dependencies) throw new Error('@ruyiAi/dsh-osc has no dependencies')
    const patch = readFileSync(resolve(root, meta.patchPath), 'utf8')
    expect(patch).toContain(`name: '${meta.manifest.name}'`)
    const plugins = Object.keys(meta.manifest.dependencies).filter(n => n.startsWith('@ruyiAi/dsh-osc-'))
    if (plugins.length === 0) throw new Error('@ruyiAi/dsh-osc has no @ruyiAi/dsh-osc-* dependencies')
    for (const name of plugins) {
      expect(patch).toContain(`name: '${name}'`)
    }
  })

  it('singularity meta-bundle patch inserts singularity-core', () => {
    const meta = bundles.find(b => b.manifest.name === '@dangosys/dsh-singularity')
    if (!meta) throw new Error('missing @dangosys/dsh-singularity package')
    if (!meta.manifest.dependencies?.['@dangosys/dsh-singularity-core']) {
      throw new Error('@dangosys/dsh-singularity must depend on @dangosys/dsh-singularity-core')
    }
    const patch = readFileSync(resolve(root, meta.patchPath), 'utf8')
    expect(patch).toContain("name: '@dangosys/dsh-singularity'")
    expect(patch).toContain("name: '@dangosys/dsh-singularity-core'")
  })
})
