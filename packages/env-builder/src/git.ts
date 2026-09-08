import { spawn, spawnSync } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'

export function cloneRepo(url: string, dir: string): Promise<void> {
  if (existsSync(dir)) throw new Error(`env-builder: path exists ${dir}`)
  return new Promise((resolve, reject) => {
    const child = spawn('git', ['clone', url, dir], { stdio: 'inherit' })
    child.on('error', err => {
      if (existsSync(dir)) rmSync(dir, { recursive: true, force: true })
      reject(err)
    })
    child.on('exit', code => {
      if (code !== 0) {
        if (existsSync(dir)) rmSync(dir, { recursive: true, force: true })
        reject(new Error(`env-builder: git clone failed for ${url}`))
        return
      }
      resolve()
    })
  })
}

export function resetRepo(dir: string): void {
  if (!existsSync(join(dir, '.git'))) throw new Error(`env-builder: not a git repo ${dir}`)
  for (const args of [
    ['reset', '--hard'],
    ['clean', '-fd'],
  ] as const) {
    const r = spawnSync('git', [...args], { cwd: dir, stdio: 'inherit' })
    if (r.status !== 0) {
      throw new Error(`env-builder: git ${args.join(' ')} failed in ${dir}`)
    }
  }
}

export function remoteOriginUrl(dir: string): string {
  const r = spawnSync('git', ['-C', dir, 'remote', 'get-url', 'origin'], { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`env-builder: git remote get-url origin failed in ${dir}`)
  const url = (r.stdout || '').trim()
  if (!url) throw new Error(`env-builder: empty origin remote in ${dir}`)
  return url
}
