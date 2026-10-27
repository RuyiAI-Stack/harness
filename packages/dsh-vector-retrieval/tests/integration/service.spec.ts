import { createServer, type Server } from 'node:http'
import { once } from 'node:events'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { createVectorRetrieval } from '../../lib/service.js'
import { python } from '../probe'

let server: Server, url: string, requests: any[], points: Map<string, any>, mode: string
const config = () => ({
  python,
  workspaceId: 'support-a',
  generation: 'faq-v1',
  fingerprint: 'fp',
  collection: 'faq',
  qdrantUrl: url,
  dimensions: 4,
  timeoutMs: 5000,
})
const chunk = {
  document_id: 'returns-policy',
  version: 1,
  chunk_id: 'paragraph-1',
  source_version: 'sha',
  text: 'Returns within 30 days.',
}

beforeEach(async () => {
  requests = []
  points = new Map()
  mode = 'ok'
  server = createServer(async (req, res) => {
    const body: Buffer[] = []
    for await (const data of req) body.push(data)
    const data = body.length ? JSON.parse(Buffer.concat(body).toString()) : undefined
    requests.push({ path: req.url, method: req.method, data })
    if (mode === 'hang') return
    if (mode === 'error') {
      res.writeHead(500)
      res.end('SECRET provider response')
      return
    }
    let result: any = { status: 'completed' }
    if (req.method === 'GET') result = { config: { params: { vectors: { size: 4, distance: 'Cosine' } } } }
    else if (req.url?.includes('/points?')) for (const point of data.points) points.set(point.id, point)
    else if (req.url?.includes('/points/query')) {
      result = { points: [...points.values()].map(p => ({ id: p.id, payload: p.payload, score: 0.8 })) }
    } else if (req.url?.includes('/points/delete')) {
      for (const [id, point] of points)
        if (
          data.filter.must.every((c: any) =>
            c.match ? point.payload[c.key] === c.match.value : point.payload[c.key] <= c.range.lte,
          )
        )
          points.delete(id)
    }
    res.setHeader('content-type', 'application/json')
    res.end(JSON.stringify({ status: 'ok', result }))
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  url = `http://127.0.0.1:${(server.address() as any).port}`
})

afterEach(async () => {
  server.closeAllConnections()
  await new Promise<void>(resolve => server.close(() => resolve()))
})

describe('installed Python CLI behind the host service', () => {
  it('indexes and retrieves a non-operator document, fencing foreign scopes and deletion', async () => {
    const a = createVectorRetrieval(config())
    const b = createVectorRetrieval({ ...config(), workspaceId: 'support-b' })
    const ids = await a.index([chunk], { vectors: [[1, 0, 0, 0]] })
    expect(await a.index([chunk], { vectors: [[1, 0, 0, 0]] })).toEqual(ids)
    const foreign = await b.index([chunk], { vectors: [[1, 0, 0, 0]] })
    expect(foreign).not.toEqual(ids)
    expect(points.size).toBe(2)
    const hits = (await a.search({ vector: [1, 0, 0, 0] })) as any[]
    expect(hits).toHaveLength(1)
    expect(hits[0].payload).toMatchObject({ workspace_id: 'support-a', document_id: 'returns-policy' })
    expect(await a.deleteVersion('returns-policy', 1)).toEqual({ completed: true })
    expect([...points.values()].map(p => p.payload.workspace_id)).toEqual(['support-b'])
    expect(requests.at(-1).data.filter.must.map((c: any) => c.key)).toEqual([
      'workspace_id',
      'document_id',
      'version',
      'generation',
      'index_fingerprint',
    ])
  })

  it('rejects malicious chunks and invalid vectors without touching the transport', async () => {
    const client = createVectorRetrieval(config())
    await expect(client.index([{ ...chunk, workspace_id: 'other' }], { vectors: [[1, 0, 0, 0]] })).rejects.toThrow(
      'operation failed',
    )
    await expect(client.search({ vector: [0, 0, 0, 0] })).rejects.toThrow('operation failed')
    await expect(client.search({ vector: [1, 0, 0, 0] }, { limit: 100, offset: 101 })).rejects.toThrow(
      'operation failed',
    )
    expect(requests).toEqual([])
  })

  it('returns safe errors for provider failures and unavailable runtimes', async () => {
    mode = 'error'
    const client = createVectorRetrieval(config())
    await expect(client.search({ vector: [1, 0, 0, 0] })).rejects.toThrow(/^Vector retrieval operation failed$/)
    const missing = createVectorRetrieval({ ...config(), python: '/not-present/SECRET-python' })
    await expect(missing.search({ vector: [1, 0, 0, 0] })).rejects.toThrow(/^Vector retrieval operation failed$/)
  })

  it('cancels hung children on timeout, abort and disposal', async () => {
    mode = 'hang'
    const client = createVectorRetrieval({ ...config(), timeoutMs: 100 })
    await expect(client.search({ vector: [1, 0, 0, 0] })).rejects.toThrow('timed out')
    const controller = new AbortController()
    controller.abort()
    await expect(client.search({ vector: [1, 0, 0, 0] }, { signal: controller.signal })).rejects.toThrow('cancelled')
    const pending = client.search({ vector: [1, 0, 0, 0] })
    client.dispose()
    await expect(pending).rejects.toThrow('disposed')
  })
})
