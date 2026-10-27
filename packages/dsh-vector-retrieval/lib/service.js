import { spawn } from 'node:child_process'
import { bridgeEnvironment, fields, resolveConfig } from './config.js'

const MAX_BYTES = 2_000_000
const failure = () => new Error('Vector retrieval operation failed')

export function createVectorRetrieval(input) {
  const config = resolveConfig(input)
  const active = new Set()
  let disposed = false
  function invoke(request, signal) {
    if (disposed) return Promise.reject(new Error('Vector retrieval service disposed'))
    if (signal?.aborted) return Promise.reject(new Error('Vector retrieval cancelled'))
    const body = JSON.stringify(request)
    if (Buffer.byteLength(body) > MAX_BYTES) return Promise.reject(new Error('Vector retrieval request exceeds limit'))
    return new Promise((resolve, reject) => {
      const child = spawn(config.python, ['-I', '-B', '-m', 'dsh_vector_retrieval'], {
        env: bridgeEnvironment(config),
        stdio: ['pipe', 'pipe', 'pipe'],
      })
      let output = '',
        bytes = 0,
        error
      const stop = reason => {
        error ||= reason
        child.kill('SIGKILL')
      }
      active.add(stop)
      const abort = () => stop(new Error('Vector retrieval cancelled'))
      const timer = setTimeout(() => stop(new Error('Vector retrieval timed out')), config.timeoutMs)
      signal?.addEventListener('abort', abort, { once: true })
      child.stdout.setEncoding('utf8')
      child.stdout.on('data', data => {
        bytes += Buffer.byteLength(data)
        if (bytes > MAX_BYTES) stop(new Error('Vector retrieval response exceeds limit'))
        else output += data
      })
      child.stderr.resume()
      child.stdin.on('error', () => {
        error ||= failure()
      })
      child.on('error', () => {
        error ||= failure()
      })
      child.on('close', code => {
        active.delete(stop)
        clearTimeout(timer)
        signal?.removeEventListener('abort', abort)
        if (error) return reject(error)
        try {
          const message = JSON.parse(output)
          if (code !== 0 || !message || message.error || !Object.hasOwn(message, 'result')) throw failure()
          resolve(message.result)
        } catch {
          reject(failure())
        }
      })
      child.stdin.end(body)
    })
  }
  return Object.freeze({
    search(query, options = {}) {
      fields(query, ['text', 'vector'])
      fields(options, ['limit', 'offset', 'signal'])
      if (Object.hasOwn(query, 'text') === Object.hasOwn(query, 'vector')) throw new Error('Provide text or vector')
      const { signal, ...page } = options
      return invoke({ action: 'search', ...query, ...page }, signal)
    },
    index(chunks, options = {}) {
      fields(options, ['vectors', 'signal'])
      return invoke({ action: 'index', chunks, vectors: options.vectors }, options.signal)
    },
    deleteVersion(documentId, version, options = {}) {
      fields(options, ['older', 'include', 'signal'])
      const { signal, ...mode } = options
      return invoke({ action: 'deleteVersion', documentId, version, ...mode }, signal)
    },
    embed(texts, options = {}) {
      fields(options, ['signal'])
      return invoke({ action: 'embed', texts }, options.signal)
    },
    dispose() {
      disposed = true
      for (const stop of active) stop(new Error('Vector retrieval service disposed'))
    },
  })
}
