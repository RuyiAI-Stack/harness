import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { nextProjectId } from './ids.ts'
import { parseRepoRef } from './parse.ts'
import { parseComponentStatus, type EnvManifest, type EnvRecord } from './types.ts'
import { cloneRepo, remoteOriginUrl, resetRepo } from './git.ts'

export class EnvStore {
  constructor(readonly root: string) {}

  private manifestPath(): string {
    return join(this.root, 'manifest.json')
  }

  private projectDirs(): string[] {
    if (!existsSync(this.root)) return []
    return readdirSync(this.root, { withFileTypes: true })
      .filter(e => e.isDirectory() && /^project\d+$/.test(e.name))
      .map(e => e.name)
  }

  load(): EnvManifest {
    const path = this.manifestPath()
    const dirs = this.projectDirs()
    if (!existsSync(path)) {
      if (dirs.length > 0) {
        throw new Error(`env-builder: project dirs exist without manifest under ${this.root}`)
      }
      return { version: 1, environments: [] }
    }
    const raw = JSON.parse(readFileSync(path, 'utf8')) as EnvManifest
    if (raw.version !== 1 || !Array.isArray(raw.environments)) {
      throw new Error(`env-builder: invalid manifest at ${path}`)
    }
    for (const env of raw.environments) {
      if (typeof env.running !== 'boolean') {
        throw new Error(`env-builder: environment ${env.id} missing boolean running`)
      }
      if (!Array.isArray(env.sessionIds)) {
        throw new Error(`env-builder: environment ${env.id} missing sessionIds array`)
      }
      for (const sid of env.sessionIds) {
        if (typeof sid !== 'string' || !sid) {
          throw new Error(`env-builder: environment ${env.id} has invalid session id`)
        }
      }
      if (!Array.isArray(env.components)) {
        throw new Error(`env-builder: environment ${env.id} missing components array`)
      }
      for (const c of env.components) {
        parseComponentStatus(c.status)
        if (c.sessionId !== undefined) {
          if (typeof c.sessionId !== 'string' || !c.sessionId) {
            throw new Error(`env-builder: environment ${env.id} component ${c.owner}/${c.repo} has invalid sessionId`)
          }
          if (!env.sessionIds.includes(c.sessionId)) {
            throw new Error(
              `env-builder: environment ${env.id} component ${c.owner}/${c.repo} sessionId ${c.sessionId} not in sessionIds`,
            )
          }
        }
      }
    }
    return raw
  }

  private save(manifest: EnvManifest): void {
    mkdirSync(this.root, { recursive: true })
    writeFileSync(this.manifestPath(), `${JSON.stringify(manifest, null, 2)}\n`)
  }

  list(): EnvRecord[] {
    return this.load().environments
  }

  get(id: string): EnvRecord {
    const env = this.load().environments.find(e => e.id === id)
    if (!env) throw new Error(`env-builder: unknown environment ${id}`)
    return env
  }

  create(): EnvRecord {
    const manifest = this.load()
    const id = nextProjectId(manifest.environments.map(e => e.id))
    const path = join(this.root, id)
    mkdirSync(path, { recursive: true })
    const env: EnvRecord = { id, path, components: [], running: false, sessionIds: [] }
    manifest.environments.push(env)
    this.save(manifest)
    return env
  }

  delete(id: string): void {
    const manifest = this.load()
    const idx = manifest.environments.findIndex(e => e.id === id)
    if (idx < 0) throw new Error(`env-builder: unknown environment ${id}`)
    const env = manifest.environments[idx]
    rmSync(env.path, { recursive: true, force: true })
    manifest.environments.splice(idx, 1)
    if (manifest.selectedId === id) {
      delete manifest.selectedId
      delete manifest.selectedRepoRef
    }
    this.save(manifest)
  }

  select(id: string): EnvRecord {
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === id)
    if (!env) throw new Error(`env-builder: unknown environment ${id}`)
    if (manifest.selectedId !== id) delete manifest.selectedRepoRef
    manifest.selectedId = id
    this.save(manifest)
    return env
  }

  selected(): EnvRecord | undefined {
    const manifest = this.load()
    if (!manifest.selectedId) return undefined
    const env = manifest.environments.find(e => e.id === manifest.selectedId)
    if (!env) throw new Error(`env-builder: selected environment missing ${manifest.selectedId}`)
    return env
  }

  selectRepo(ref: string): string {
    const env = this.selected()
    if (!env) throw new Error('env-builder: no environment selected')
    const comp = env.components.find(c => `${c.owner}/${c.repo}` === ref)
    if (!comp) throw new Error(`env-builder: environment ${env.id} has no component ${ref}`)
    const manifest = this.load()
    manifest.selectedRepoRef = ref
    this.save(manifest)
    return ref
  }

  selectedRepo(): string | undefined {
    const manifest = this.load()
    if (!manifest.selectedRepoRef) return undefined
    const env = this.selected()
    if (!env) throw new Error(`env-builder: selectedRepoRef set without selected environment`)
    const comp = env.components.find(c => `${c.owner}/${c.repo}` === manifest.selectedRepoRef)
    if (!comp) throw new Error(`env-builder: selected repository missing ${manifest.selectedRepoRef}`)
    return manifest.selectedRepoRef
  }

  componentPath(name: string): string {
    const env = this.selected()
    if (!env) throw new Error('env-builder: no environment selected')
    const comp = env.components.find(c => c.repo === name)
    if (!comp) throw new Error(`env-builder: environment ${env.id} has no component ${name}`)
    return join(env.path, comp.dir)
  }

  ensureComponent(name: string): string {
    const repo = this.selectedRepo()
    if (repo === undefined) throw new Error('env-builder: no repository selected')
    if (repo.split('/')[1] !== name) throw new Error(`env-builder: selected repository is ${repo}, not */${name}`)
    const env = this.selected()
    if (!env) throw new Error('env-builder: no environment selected')
    const comp = env.components.find(c => c.repo === name)
    if (!comp) throw new Error(`env-builder: environment ${env.id} has no component ${name}`)
    const dir = join(env.path, comp.dir)
    if (!existsSync(join(dir, '.git'))) throw new Error(`env-builder: component not cloned: ${name}`)
    const origin = remoteOriginUrl(dir)
    if (origin.replace(/\.git$/, '') !== comp.url.replace(/\.git$/, '')) {
      throw new Error(`env-builder: component ${name} origin ${origin} != ${comp.url}`)
    }
    return dir
  }

  registerComponent(envId: string, ref: string): string {
    const parsed = parseRepoRef(ref)
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    if (env.components.some(c => c.dir === parsed.dir)) {
      throw new Error(`env-builder: duplicate component dir ${parsed.dir} in ${envId}`)
    }
    const dir = join(env.path, parsed.dir)
    if (!existsSync(join(dir, '.git'))) throw new Error(`env-builder: component not present at ${dir}`)
    const origin = remoteOriginUrl(dir)
    if (origin.replace(/\.git$/, '') !== parsed.url.replace(/\.git$/, '')) {
      throw new Error(`env-builder: component ${parsed.dir} origin ${origin} != ${parsed.url}`)
    }
    env.components.push({ ...parsed, status: 'ready' })
    this.save(manifest)
    return dir
  }

  /** Record a github component to install later (no clone yet). */
  planComponent(envId: string, ref: string): void {
    const parsed = parseRepoRef(ref)
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    if (env.components.some(c => c.dir === parsed.dir)) {
      throw new Error(`env-builder: duplicate component dir ${parsed.dir} in ${envId}`)
    }
    const dir = join(env.path, parsed.dir)
    if (existsSync(dir)) throw new Error(`env-builder: path already exists ${dir}`)
    env.components.push({ ...parsed, status: 'installing' })
    this.save(manifest)
  }

  async addComponent(envId: string, ref: string): Promise<string> {
    const parsed = parseRepoRef(ref)
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    if (env.components.some(c => c.dir === parsed.dir)) {
      throw new Error(`env-builder: duplicate component dir ${parsed.dir} in ${envId}`)
    }
    const dir = join(env.path, parsed.dir)
    env.components.push({ ...parsed, status: 'installing' })
    this.save(manifest)
    try {
      await cloneRepo(parsed.url, dir)
    } catch (err) {
      const failed = this.load()
      const fe = failed.environments.find(e => e.id === envId)
      if (!fe) throw err
      fe.components = fe.components.filter(c => c.dir !== parsed.dir)
      this.save(failed)
      throw err
    }
    const done = this.load()
    const de = done.environments.find(e => e.id === envId)
    if (!de) throw new Error(`env-builder: unknown environment ${envId}`)
    const comp = de.components.find(c => c.dir === parsed.dir)
    if (!comp) throw new Error(`env-builder: component missing after clone ${parsed.dir}`)
    comp.status = 'ready'
    this.save(done)
    return dir
  }

  removeComponent(envId: string, ref: string): void {
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    const idx = env.components.findIndex(c => `${c.owner}/${c.repo}` === ref)
    if (idx < 0) throw new Error(`env-builder: environment ${envId} has no component ${ref}`)
    const comp = env.components[idx]
    if (comp.sessionId) {
      const sidx = env.sessionIds.indexOf(comp.sessionId)
      if (sidx < 0) {
        throw new Error(
          `env-builder: environment ${envId} component ${ref} sessionId ${comp.sessionId} not in sessionIds`,
        )
      }
      env.sessionIds.splice(sidx, 1)
    }
    env.components.splice(idx, 1)
    rmSync(join(env.path, comp.dir), { recursive: true, force: true })
    if (manifest.selectedRepoRef === ref) delete manifest.selectedRepoRef
    this.save(manifest)
  }

  setComponentStatus(envId: string, ref: string, status: unknown) {
    const parsed = parseComponentStatus(status)
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    const comp = env.components.find(c => `${c.owner}/${c.repo}` === ref)
    if (!comp) throw new Error(`env-builder: environment ${envId} has no component ${ref}`)
    comp.status = parsed
    this.save(manifest)
    return comp
  }

  bindComponentSession(envId: string, ref: string, sessionId: string): EnvComponent {
    if (typeof sessionId !== 'string' || !sessionId) throw new Error('env-builder: missing session id')
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    const comp = env.components.find(c => `${c.owner}/${c.repo}` === ref)
    if (!comp) throw new Error(`env-builder: environment ${envId} has no component ${ref}`)
    if (comp.sessionId) {
      throw new Error(`env-builder: environment ${envId} component ${ref} already bound to ${comp.sessionId}`)
    }
    for (const c of env.components) {
      if (c.sessionId === sessionId) {
        throw new Error(`env-builder: session ${sessionId} already bound to ${c.owner}/${c.repo}`)
      }
    }
    if (!env.sessionIds.includes(sessionId)) {
      env.sessionIds.push(sessionId)
    }
    comp.sessionId = sessionId
    this.save(manifest)
    return comp
  }

  attachSession(envId: string, sessionId: string): EnvRecord {
    if (typeof sessionId !== 'string' || !sessionId) throw new Error('env-builder: missing session id')
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    if (env.sessionIds.includes(sessionId)) {
      throw new Error(`env-builder: environment ${envId} already has session ${sessionId}`)
    }
    env.sessionIds.push(sessionId)
    this.save(manifest)
    return env
  }

  detachSession(envId: string, sessionId: string): EnvRecord {
    if (typeof sessionId !== 'string' || !sessionId) throw new Error('env-builder: missing session id')
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    for (const c of env.components) {
      if (c.sessionId === sessionId) {
        throw new Error(`env-builder: session ${sessionId} bound to ${c.owner}/${c.repo}; removeComponent instead`)
      }
    }
    const idx = env.sessionIds.indexOf(sessionId)
    if (idx < 0) throw new Error(`env-builder: environment ${envId} has no session ${sessionId}`)
    env.sessionIds.splice(idx, 1)
    this.save(manifest)
    return env
  }

  rename(id: string, newId: string): EnvRecord {
    if (typeof newId !== 'string' || !newId) throw new Error('env-builder: missing new environment id')
    if (newId.includes('/') || newId.includes('\\') || newId === '.' || newId === '..') {
      throw new Error(`env-builder: invalid environment id ${newId}`)
    }
    const manifest = this.load()
    const idx = manifest.environments.findIndex(e => e.id === id)
    if (idx < 0) throw new Error(`env-builder: unknown environment ${id}`)
    if (manifest.environments.some(e => e.id === newId)) {
      throw new Error(`env-builder: environment id conflict ${newId}`)
    }
    const env = manifest.environments[idx]
    const newPath = join(this.root, newId)
    if (existsSync(newPath)) throw new Error(`env-builder: environment path exists ${newPath}`)
    renameSync(env.path, newPath)
    env.id = newId
    env.path = newPath
    if (manifest.selectedId === id) manifest.selectedId = newId
    this.save(manifest)
    return env
  }

  setRunning(id: string, running: boolean): EnvRecord {
    if (typeof running !== 'boolean') throw new Error('env-builder: running must be boolean')
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === id)
    if (!env) throw new Error(`env-builder: unknown environment ${id}`)
    env.running = running
    this.save(manifest)
    return env
  }

  reset(envId: string): void {
    const env = this.get(envId)
    for (const c of env.components) {
      resetRepo(join(env.path, c.dir))
    }
  }

  /**
   * Certify an environment is rebindable after an agent-driven clean.
   * Clears session bindings; does not mutate the filesystem.
   */
  markClean(envId: string): EnvRecord {
    const manifest = this.load()
    const env = manifest.environments.find(e => e.id === envId)
    if (!env) throw new Error(`env-builder: unknown environment ${envId}`)
    if (!existsSync(env.path)) throw new Error(`env-builder: environment path missing ${env.path}`)
    env.sessionIds = []
    for (const c of env.components) delete c.sessionId
    env.running = false
    this.save(manifest)
    return this.get(envId)
  }
}
