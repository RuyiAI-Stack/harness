import { createHash, randomUUID } from 'node:crypto'
import { readFileSync, writeFileSync, mkdirSync, renameSync, rmSync, realpathSync } from 'node:fs'
import { isAbsolute, join, resolve } from 'node:path'
import { configFromEnvironment, resolveConfig } from './config.js'

// Domain state lives beside (not inside) the host's versioned conversation log.
// The host refuses unknown event types on restore; never extend its core log ad hoc.
export function createStateStore(input = process.env) {
  const config = Object.hasOwn(input, 'repoRoot') ? input : resolveConfig(configFromEnvironment(input))
  const configured = config.repoRoot
  if (!configured || !isAbsolute(configured)) throw new Error('Set an absolute TRITON_RISCV_REPO_ROOT')
  const root = realpathSync(configured)
  const directory = join(resolve(root, config.stateDir || 'agent-results'), 'native-harness')
  const path = id => join(directory, createHash('sha256').update(`${root}\0${id}`).digest('hex') + '.json')
  return {
    load(id) {
      try {
        const data = JSON.parse(readFileSync(path(id), 'utf8'))
        if (data.schema !== 1 || data.repository !== root || data.session_id !== id)
          throw new Error('Native state identity mismatch')
        return data
      } catch (error) {
        if (error.code !== 'ENOENT') throw error
        return { schema: 1, repository: root, session_id: id, artifacts: {}, task: {}, contract: null, memory: null }
      }
    },
    save(state) {
      mkdirSync(directory, { recursive: true, mode: 0o700 })
      const destination = path(state.session_id)
      const temporary = destination + '.' + randomUUID() + '.tmp'
      try {
        writeFileSync(temporary, JSON.stringify(state), { encoding: 'utf8', mode: 0o600, flag: 'wx' })
        renameSync(temporary, destination)
      } finally {
        rmSync(temporary, { force: true })
      }
    },
  }
}
