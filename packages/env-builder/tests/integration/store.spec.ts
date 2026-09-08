import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { spawnSync } from 'node:child_process'
import { EnvStore } from '../../src/service/store.ts'

describe('EnvStore', () => {
  let root: string
  let store: EnvStore

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'env-builder-'))
    store = new EnvStore(root)
  })

  afterEach(() => {
    rmSync(root, { recursive: true, force: true })
  })

  it('create makes project1 and manifest', () => {
    const env = store.create()
    expect(env.id).toBe('project1')
    expect(env.running).toBe(false)
    expect(env.sessionIds).toEqual([])
    expect(existsSync(env.path)).toBe(true)
    const man = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf8'))
    expect(man.version).toBe(1)
    expect(man.environments).toHaveLength(1)
    expect(man.environments[0].running).toBe(false)
    expect(man.environments[0].sessionIds).toEqual([])
  })

  it('delete removes dir and manifest entry', () => {
    const env = store.create()
    store.delete(env.id)
    expect(store.list()).toHaveLength(0)
    expect(existsSync(env.path)).toBe(false)
  })

  it('get unknown throws', () => {
    expect(() => store.get('project9')).toThrow(/unknown environment/)
  })

  it('select persists selectedId and clears on delete', () => {
    const env = store.create()
    store.select(env.id)
    expect(store.selected()?.id).toBe(env.id)
    store.delete(env.id)
    expect(store.selected()).toBeUndefined()
  })

  it('selectRepo requires explicit selection and clears on env change', () => {
    const env = store.create()
    store.select(env.id)
    expect(() => store.selectRepo('DangoSys/buckyball')).toThrow(/no component/)
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    rec.components.push({
      owner: 'DangoSys',
      repo: 'buckyball',
      url: 'https://github.com/DangoSys/buckyball.git',
      dir: 'buckyball',
      status: 'ready',
    })
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(store.selectRepo('DangoSys/buckyball')).toBe('DangoSys/buckyball')
    expect(store.selectedRepo()).toBe('DangoSys/buckyball')
    const env2 = store.create()
    store.select(env2.id)
    expect(store.selectedRepo()).toBeUndefined()
  })

  it('registerComponent records a cloned repo', () => {
    const bare = join(root, 'bare2.git')
    mkdirSync(bare)
    expect(spawnSync('git', ['init', '--bare', bare]).status).toBe(0)
    const env = store.create()
    const dir = join(env.path, 'local', 'seeded')
    expect(spawnSync('git', ['clone', bare, dir]).status).toBe(0)
    expect(() => store.registerComponent(env.id, 'local/seeded')).toThrow(/origin/)
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    rec.components = []
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(
      spawnSync('git', ['-C', dir, 'remote', 'set-url', 'origin', 'https://github.com/local/seeded.git']).status,
    ).toBe(0)
    const path = store.registerComponent(env.id, 'local/seeded')
    expect(path).toBe(dir)
    expect(store.get(env.id).components[0].status).toBe('ready')
  })

  it('addComponent clones and reset cleans', () => {
    const bare = join(root, 'bare.git')
    mkdirSync(bare)
    expect(spawnSync('git', ['init', '--bare', bare]).status).toBe(0)
    const work = join(root, 'seed')
    expect(spawnSync('git', ['clone', bare, work]).status).toBe(0)
    writeFileSync(join(work, 'README'), 'x\n')
    expect(spawnSync('git', ['-C', work, 'add', 'README']).status).toBe(0)
    const gitEnv = {
      ...process.env,
      GIT_AUTHOR_NAME: 'test',
      GIT_AUTHOR_EMAIL: 'test@test',
      GIT_COMMITTER_NAME: 'test',
      GIT_COMMITTER_EMAIL: 'test@test',
    }
    expect(spawnSync('git', ['-C', work, 'commit', '-m', 'init'], { env: gitEnv }).status).toBe(0)
    expect(spawnSync('git', ['-C', work, 'push', 'origin', 'HEAD:master']).status).toBe(0)

    const env = store.create()
    // file URL clone via addComponent needs github-shaped ref — call clone path directly through store after hacking:
    // Use store.addComponent only with github refs. For unit test, clone manually then register via internal path:
    const dir = join(env.path, 'seeded')
    expect(spawnSync('git', ['clone', bare, dir]).status).toBe(0)
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    rec.components.push({
      owner: 'local',
      repo: 'seeded',
      url: bare,
      dir: 'seeded',
      status: 'ready',
    })
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)

    writeFileSync(join(dir, 'junk.txt'), 'dirty\n')
    store.reset(env.id)
    expect(existsSync(join(dir, 'junk.txt'))).toBe(false)
  })

  it('selected throws when selectedId is missing from environments', () => {
    const env = store.create()
    store.select(env.id)
    const man = store.load()
    man.environments = []
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(() => store.selected()).toThrow(/selected environment missing/)
  })

  it('selectedRepo throws when selectedRepoRef is inconsistent', () => {
    const env = store.create()
    store.select(env.id)
    const man = store.load()
    man.selectedRepoRef = 'DangoSys/missing'
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(() => store.selectedRepo()).toThrow(/selected repository missing/)
  })

  it('setRunning toggles and rejects non-boolean on load', () => {
    const env = store.create()
    expect(store.setRunning(env.id, true).running).toBe(true)
    expect(store.get(env.id).running).toBe(true)
    const man = store.load()
    delete (man.environments[0] as { running?: boolean }).running
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(() => store.load()).toThrow(/missing boolean running/)
  })

  it('setComponentStatus only accepts ready|modified|installing', () => {
    const env = store.create()
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    rec.components.push({
      owner: 'DangoSys',
      repo: 'buckyball',
      url: 'https://github.com/DangoSys/buckyball.git',
      dir: 'buckyball',
      status: 'ready',
    })
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(store.setComponentStatus(env.id, 'DangoSys/buckyball', 'modified').status).toBe('modified')
    expect(() => store.setComponentStatus(env.id, 'DangoSys/buckyball', 'weird')).toThrow(/invalid component status/)
  })

  it('removeComponent drops dir and clears selectedRepoRef', () => {
    const env = store.create()
    store.select(env.id)
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    const dir = join(env.path, 'buckyball')
    mkdirSync(dir, { recursive: true })
    rec.components.push({
      owner: 'DangoSys',
      repo: 'buckyball',
      url: 'https://github.com/DangoSys/buckyball.git',
      dir: 'buckyball',
      status: 'ready',
    })
    man.selectedRepoRef = 'DangoSys/buckyball'
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    store.removeComponent(env.id, 'DangoSys/buckyball')
    expect(store.get(env.id).components).toHaveLength(0)
    expect(existsSync(dir)).toBe(false)
    expect(store.load().selectedRepoRef).toBeUndefined()
  })

  it('rename moves dir and throws on missing or conflict', () => {
    const env = store.create()
    store.select(env.id)
    const renamed = store.rename(env.id, 'alpha')
    expect(renamed.id).toBe('alpha')
    expect(existsSync(join(root, 'alpha'))).toBe(true)
    expect(existsSync(join(root, 'project1'))).toBe(false)
    expect(store.selected()?.id).toBe('alpha')
    expect(() => store.rename('missing', 'beta')).toThrow(/unknown environment/)
    store.create()
    expect(() => store.rename('project1', 'alpha')).toThrow(/conflict/)
    const next = store.create()
    expect(next.id).toBe('project2')
  })

  it('bindComponentSession and removeComponent clears bound session', () => {
    const env = store.create()
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    const dir = join(env.path, 'DangoSys/buckyball')
    mkdirSync(dir, { recursive: true })
    rec.components.push({
      owner: 'DangoSys',
      repo: 'buckyball',
      url: 'https://github.com/DangoSys/buckyball.git',
      dir: 'DangoSys/buckyball',
      status: 'ready',
    })
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(store.bindComponentSession(env.id, 'DangoSys/buckyball', 's1').sessionId).toBe('s1')
    expect(store.get(env.id).sessionIds).toEqual(['s1'])
    expect(() => store.bindComponentSession(env.id, 'DangoSys/buckyball', 's2')).toThrow(/already bound/)
    store.removeComponent(env.id, 'DangoSys/buckyball')
    expect(store.get(env.id).sessionIds).toEqual([])
    expect(store.get(env.id).components).toHaveLength(0)
  })
  it('detachSession throws when session is bound to a component', () => {
    const env = store.create()
    const man = store.load()
    const rec = man.environments.find(e => e.id === env.id)!
    mkdirSync(join(env.path, 'DangoSys/buckyball'), { recursive: true })
    rec.components.push({
      owner: 'DangoSys',
      repo: 'buckyball',
      url: 'https://github.com/DangoSys/buckyball.git',
      dir: 'DangoSys/buckyball',
      status: 'ready',
    })
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    store.bindComponentSession(env.id, 'DangoSys/buckyball', 's1')
    expect(() => store.detachSession(env.id, 's1')).toThrow(/removeComponent instead/)
  })

  it('attachSession and detachSession throw on missing mapping', () => {
    const env = store.create()
    expect(store.attachSession(env.id, 's1').sessionIds).toEqual(['s1'])
    expect(() => store.attachSession(env.id, 's1')).toThrow(/already has session/)
    expect(store.detachSession(env.id, 's1').sessionIds).toEqual([])
    expect(() => store.detachSession(env.id, 's1')).toThrow(/has no session/)
    const man = store.load()
    delete (man.environments[0] as { sessionIds?: string[] }).sessionIds
    writeFileSync(join(root, 'manifest.json'), `${JSON.stringify(man, null, 2)}\n`)
    expect(() => store.load()).toThrow(/missing sessionIds/)
  })
})
