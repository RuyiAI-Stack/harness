import { createServer } from 'node:http'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createMultiAgent } from '../../index.js'
import { python } from '../probe'

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof import('node:child_process')>()
  return { ...actual, spawn: vi.fn(actual.spawn) }
})

afterEach(() => {
  vi.unstubAllEnvs()
  vi.mocked(spawn).mockClear()
})

function cleaned() {
  for (const call of vi.mocked(spawn).mock.calls) expect(existsSync((call[2] as any).cwd)).toBe(false)
  for (const result of vi.mocked(spawn).mock.results) {
    const pid = result.value.pid
    if (pid && process.platform !== 'win32') expect(() => process.kill(-pid, 0)).toThrow()
  }
}

async function fixture(mode = 'valid') {
  const requests: any[] = []
  const server = createServer(async (req, res) => {
    let body = ''
    for await (const chunk of req) body += chunk
    requests.push(JSON.parse(body))
    res.writeHead(200, { 'Content-Type': 'text/event-stream' })
    const text = mode === 'invalid' ? '{"category":"unknown"}' : '{"category":"question"}'
    const item = {
      type: 'message',
      id: 'msg_fixture',
      role: 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text, annotations: [] }],
    }
    const write = (event: any) => res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    write({ type: 'response.created', response: { id: 'resp_fixture', status: 'in_progress', output: [] } })
    if (mode === 'stall') return
    write({
      type: 'response.output_item.added',
      output_index: 0,
      item: { ...item, status: 'in_progress', content: [] },
    })
    write({
      type: 'response.content_part.added',
      item_id: item.id,
      output_index: 0,
      content_index: 0,
      part: { type: 'output_text', text: '', annotations: [] },
    })
    write({ type: 'response.output_text.delta', item_id: item.id, output_index: 0, content_index: 0, delta: text })
    write({ type: 'response.output_text.done', item_id: item.id, output_index: 0, content_index: 0, text })
    write({
      type: 'response.content_part.done',
      item_id: item.id,
      output_index: 0,
      content_index: 0,
      part: item.content[0],
    })
    write({ type: 'response.output_item.done', output_index: 0, item })
    write({
      type: 'response.completed',
      response: {
        id: 'resp_fixture',
        status: 'completed',
        output: [item],
        usage: {
          input_tokens: 32,
          output_tokens: 16,
          total_tokens: 48,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens_details: { reasoning_tokens: 0 },
        },
      },
    })
    res.end()
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  vi.stubEnv('MULTI_AGENT_FIXTURE_API_KEY', 'fixture-only-no-paid-call')
  vi.stubEnv('NO_PROXY', '127.0.0.1,localhost')
  vi.stubEnv('DSH_HOME', '/parent-profile-must-not-leak')
  const api = createMultiAgent({
    python,
    model: {
      provider: 'fixture',
      model: 'mail-classifier',
      apiKeyEnv: 'MULTI_AGENT_FIXTURE_API_KEY',
      requestTimeoutSeconds: 10,
      baseURL: `http://127.0.0.1:${(server.address() as any).port}/v1`,
    },
    tasks: {
      classify: {
        systemPrompt: 'Classify mail. No tools are available.',
        resultSchema: {
          type: 'object',
          additionalProperties: false,
          required: ['category'],
          properties: { category: { enum: ['question', 'feedback'] } },
        },
      },
    },
  })
  return {
    api,
    requests,
    async close() {
      await api.close()
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    },
  }
}

describe('installed Python bridge with a loopback provider', () => {
  it('runs an independent typed task with fresh sessions and measured usage', async () => {
    const f = await fixture()
    try {
      const first = f.api.run({ task: 'classify', input: { message: 'When will it arrive?' } })
      const result = await f.api.result(first.id)
      expect(result, JSON.stringify(result)).toMatchObject({ outcome: 'validated', result: { category: 'question' } })
      expect(result.usage).toMatchObject({ model: 'mail-classifier', totals: { total_tokens: 48 }, attempt_count: 1 })
      expect(f.api.usage(first.id)).toEqual(result.usage)
      expect(f.api.status(first.id).status).toBe('validated')
      const second = f.api.run({ task: 'classify', input: { message: 'A separate question' } })
      expect((await f.api.result(second.id)).session_id).not.toBe(result.session_id)
      expect(f.requests).toHaveLength(2)
      for (const request of f.requests) {
        expect(request.model).toBe('mail-classifier')
        expect(request.tools ?? []).toEqual([])
        expect(JSON.stringify(request)).not.toContain('parent-profile-must-not-leak')
        expect(request).not.toHaveProperty('temperature')
      }
      expect(JSON.stringify(f.requests[1])).not.toContain('When will it arrive?')
      cleaned()
    } finally {
      await f.close()
    }
  })
  it('retains rejected output and observed usage without retrying', async () => {
    const f = await fixture('invalid')
    try {
      const { id } = f.api.run({ task: 'classify', input: { message: 'A question' } })
      const result = await f.api.result(id)
      expect(result).toMatchObject({
        outcome: 'invalid_output',
        result: null,
        harness: { final_response: '{"category":"unknown"}' },
        usage: { totals: { total_tokens: 48 } },
      })
      expect(f.requests).toHaveLength(1)
      cleaned()
    } finally {
      await f.close()
    }
  })
  it('cancels a running SDK invocation, reaps the process and keeps unknown usage honest', async () => {
    const f = await fixture('stall')
    try {
      const { id } = f.api.run({ task: 'classify', input: { message: 'A question' } })
      await expect.poll(() => f.requests.length, { timeout: 15000 }).toBe(1)
      expect(f.api.cancel(id)).toBe(true)
      const result = await f.api.result(id)
      expect(result).toMatchObject({ outcome: 'cancelled', result: null })
      expect(result.usage?.totals.total_tokens ?? null).toBeNull()
      expect(f.api.status(id).status).toBe('cancelled')
      expect(f.api.cancel(id)).toBe(false)
      cleaned()
    } finally {
      await f.close()
    }
  })
})
