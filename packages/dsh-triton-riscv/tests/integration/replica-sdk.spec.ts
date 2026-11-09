import { expect, it } from 'vitest'
import { createServer } from 'node:http'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const python =
  process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url))
const driver = `
import json, os, sys, tempfile
from pathlib import Path
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.delegation import Plans, host_scope
from codex_agent.harness.runtime import HarnessAgent
data = json.load(sys.stdin)
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    examples = root / "python/examples/flaggems"; examples.mkdir(parents=True)
    (examples / "fixture_square.py").write_text("import triton\\n@triton.jit\\ndef fixture_square_kernel(X, Y):\\n    pass\\ndef fixture_square(x):\\n    return x * x\\n")
    settings = HarnessSettings.from_env(root, {"ISRC_API_KEY":"loopback-fixture-only", "ISRC_BASE_URL":data["url"], "DSH_MODEL":"replica-fixture"})
    if data["routing"]:
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(dict(schemaVersion=1,repoRoot=folder,stateDir=folder,modelRouting={
            "enabled":True,"fastModel":"fast-fixture","strongModel":"strong-fixture","fastMaxInputBytes":64000}))
    plans = Plans(root)
    backend = HarnessAgent(settings)
    try:
        with host_scope("s", "r", "List independent source facts"):
            backend.run("Prepare independent read-only replica tasks. Wait for human approval.", session_id="parent-fixture")
        row = plans.pending("s")[0]
        before = {"status":row["status"], "results":len(row["results"])}
        plans.decide(row["plan_id"], "s", row["digest"], True, "test-human")
        with host_scope("s", "r2", "Continue the approved plan"):
            result = backend.run("Execute the approved plan " + row["plan_id"], session_id="parent-fixture-next" if data["routing"] else "parent-fixture")
        print(json.dumps({**plans.public(plans.load(row["plan_id"], "s")), "before":before, "parent_response":result.final_response,
            "parent_finish":result.finish_reason,"parent_events":result.events if not result.final_response else []}))
    finally:
        backend.close()
`

function streamReply(
  tool: boolean,
  index: number,
  name = 'mcp__triton_riscv__discover_operator',
  args = '{"operator_name":"fixture_square"}',
) {
  const response = { id: `resp_${index}`, status: 'in_progress', output: [] }
  const item = tool
    ? {
        type: 'function_call',
        id: `fc_${index}`,
        call_id: `call_${index}`,
        name,
        arguments: args,
        status: 'completed',
      }
    : {
        type: 'message',
        id: `msg_${index}`,
        role: 'assistant',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'Inspected fixture_square.py. This is analysis, not test success.',
            annotations: [],
          },
        ],
      }
  const events: any[] = [
    { type: 'response.created', response },
    {
      type: 'response.output_item.added',
      output_index: 0,
      item: { ...item, status: 'in_progress', ...(tool ? { arguments: '' } : { content: [] }) },
    },
  ]
  if (tool)
    events.push(
      { type: 'response.function_call_arguments.delta', item_id: item.id, output_index: 0, delta: item.arguments },
      { type: 'response.function_call_arguments.done', item_id: item.id, output_index: 0, arguments: item.arguments },
    )
  else {
    const part = item.content![0]
    events.push(
      {
        type: 'response.content_part.added',
        item_id: item.id,
        output_index: 0,
        content_index: 0,
        part: { ...part, text: '' },
      },
      { type: 'response.output_text.delta', item_id: item.id, output_index: 0, content_index: 0, delta: part.text },
      { type: 'response.output_text.done', item_id: item.id, output_index: 0, content_index: 0, text: part.text },
      { type: 'response.content_part.done', item_id: item.id, output_index: 0, content_index: 0, part },
    )
  }
  events.push(
    { type: 'response.output_item.done', output_index: 0, item },
    {
      type: 'response.completed',
      response: {
        ...response,
        status: 'completed',
        output: [item],
        usage: {
          input_tokens: 100,
          output_tokens: 20,
          total_tokens: 120,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens_details: { reasoning_tokens: 0 },
        },
      },
    },
  )
  return events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join('')
}

it.each([false, true])(
  'runs actual SDK model selection and approved isolated tool loops (routing=%s)',
  async routing => {
    const requests: any[] = []
    let active = 0,
      peak = 0
    let parentRequests = 0
    const server = createServer(async (req, res) => {
      const chunks: Buffer[] = []
      for await (const chunk of req) chunks.push(chunk)
      const body = JSON.parse(Buffer.concat(chunks).toString())
      requests.push(body)
      const index = requests.length
      if (body.tools.some(tool => tool.name === 'mcp__triton_riscv__prepare_parallel_tasks')) {
        parentRequests++
        const tool = parentRequests === 1 || parentRequests === 3
        const args =
          parentRequests === 1
            ? JSON.stringify({
                reason: 'Separate independent read-only investigations',
                context: 'Only inspect current sources; do not run tests',
                tasks: [{ goal: 'GOAL_ALPHA list forward function names' }, { goal: 'GOAL_BETA review test coverage' }],
              })
            : JSON.stringify({ plan_id: JSON.stringify(body.input).match(/parallel-[a-f0-9]{32}/)?.[0] })
        res.writeHead(200, { 'Content-Type': 'text/event-stream' })
        res.end(
          streamReply(
            tool,
            index,
            parentRequests === 1
              ? 'mcp__triton_riscv__prepare_parallel_tasks'
              : 'mcp__triton_riscv__execute_parallel_tasks',
            args,
          ),
        )
        return
      }
      active++
      peak = Math.max(peak, active)
      await new Promise(resolve => setTimeout(resolve, 300))
      active--
      const tool = !JSON.stringify(body.input).includes('function_call_output')
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      res.end(streamReply(tool, index))
    })
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(0, '127.0.0.1', resolve)
    })
    try {
      const address: any = server.address()
      const data = await new Promise<any>((accept, reject) => {
        const env = Object.fromEntries(
          Object.entries(process.env).filter(
            ([key]) => !/^(TRITON_|RISCV_|DSH_)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
          ),
        )
        env.NO_PROXY = '127.0.0.1,localhost'
        const child = spawn(python, ['-I', '-B', '-c', driver], { env })
        let stdout = '',
          stderr = ''
        child.stdout.on('data', value => {
          stdout += value
        })
        child.stderr.on('data', value => {
          stderr += value
        })
        const timer = setTimeout(() => child.kill('SIGKILL'), 35000)
        child.on('error', reject)
        child.on('close', code => {
          clearTimeout(timer)
          if (code) reject(new Error(stderr + stdout))
          else {
            try {
              accept(JSON.parse(stdout))
            } catch (error) {
              reject(error)
            }
          }
        })
        child.stdin.end(JSON.stringify({ url: `http://127.0.0.1:${address.port}/v1`, routing }))
      })
      expect(data.status, JSON.stringify(data)).toBe('completed')
      expect(data.before).toEqual({ status: 'pending', results: 0 })
      expect(data.parent_response).toContain('not test success')
      expect(data.results).toHaveLength(2)
      expect(new Set(data.results.map(row => row.session_id)).size).toBe(2)
      expect(requests).toHaveLength(8)
      expect(parentRequests).toBe(4)
      if (routing) {
        expect(requests.filter(x => x.model === 'fast-fixture')).toHaveLength(4)
        expect(requests.filter(x => x.model === 'strong-fixture')).toHaveLength(4)
        expect(data.model).toBe('replica-fixture')
        expect(data.tasks.map(x => x.route.model)).toEqual(['fast-fixture', 'strong-fixture'])
      } else expect(requests.every(x => x.model === 'replica-fixture')).toBe(true)
      expect(peak).toBe(2)
      for (const body of requests) {
        if (body.tools.some(tool => tool.name === 'mcp__triton_riscv__prepare_parallel_tasks')) continue
        expect(body.tools.map(tool => tool.name).sort()).toEqual(
          ['discover_operator', 'inspect_project', 'read_operator_file', 'retrieve_operator_memory'].map(
            name => 'mcp__triton_riscv__' + name,
          ),
        )
        const text = JSON.stringify(body)
        expect(text.includes('GOAL_ALPHA') !== text.includes('GOAL_BETA')).toBe(true)
        expect(text).not.toContain('loopback-fixture-only')
        const outputs = body.input.filter(item => item.type === 'function_call_output')
        if (outputs.length) {
          expect(JSON.stringify(outputs)).toContain('fixture_square.py')
          expect(JSON.stringify(outputs)).toContain('found')
        }
      }
      for (const row of data.results) {
        expect(row.response).toContain('not test success')
        expect(row.usage.totals.total_tokens).toBe(240)
        expect(row.usage.attempt_count).toBe(2)
      }
    } finally {
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  },
  45000,
)
