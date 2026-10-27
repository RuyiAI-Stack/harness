import { describe, expect, it } from 'vitest'
import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python')
const driver = `
import json, os, sys, tempfile
from pathlib import Path
from unittest.mock import patch
from deepseek_harness import DeepSeekHarness
from codex_agent.harness.config import HarnessSettings
from codex_agent.harness.runtime import DeepSeekHarnessBackend
from codex_agent.orchestration.runtime import RoleAgent
from codex_agent.platform.conversation_context import ContextPolicy, ConversationContextManager
data = json.load(sys.stdin)
homes, profiles = [], []
def factory(**kwargs):
    home = Path(kwargs["dsh_home"])
    homes.append(home)
    profiles.append(dict(manifest=json.loads((home / "profiles/triton/package.json").read_text()),
                         patch=json.loads((home / "profiles/triton/cordis.patch.yml").read_text())))
    return DeepSeekHarness(**kwargs)
with tempfile.TemporaryDirectory() as temporary:
    workspace = Path(temporary)
    (workspace / "AGENTS.md").write_text("WORKSPACE_INSTRUCTION_MARKER")
    settings = HarnessSettings.from_env(workspace, {"ISRC_API_KEY":"fixture-key", "DSH_MODEL":"fixture-model",
        "ISRC_BASE_URL":data["url"], "DSH_REQUEST_TIMEOUT_SECONDS":"8"})
    # Attempt ambient-profile contamination: explicit SDK home must win.
    ambient = workspace / "ambient"
    ambient.mkdir()
    (ambient / "cordis.patch.yml").write_text("not a valid patch: PRIVATE_AMBIENT_HOME_MARKER")
    with patch.dict(os.environ, {"DSH_HOME":str(ambient), "NO_PROXY":"127.0.0.1,localhost"}), \\
         patch("deepseek_harness.DeepSeekHarness", factory):
        if data.get("chat"):
            backend = DeepSeekHarnessBackend(settings)
            try:
                runs = [backend.run("Discover fixture_square using the repository tool.", "chat-fixture")]
                runs.append(backend.run("Continue the same conversation; only summarize.", "chat-fixture"))
                backend.close()
                backend = DeepSeekHarnessBackend(settings)
                collision = backend.run("Do not overwrite a persisted session.", "chat-fixture")
                history = [dict(id=f"history-{i}", role="assistant", content=run.final_response)
                           for i, run in enumerate(runs)]
                # The product rotates SDK IDs on restart, then supplies managed history.
                bundle = ConversationContextManager(ContextPolicy()).build(
                    current_request="Continue the prior discussion after restarting.", messages=history)
                runs.append(backend.run(bundle.prompt, "chat-fixture-new-runtime"))
                result = dict(responses=[run.final_response for run in runs],
                    events=[event for run in runs for event in run.events],
                    collision_reason=collision.finish_reason, collision_events=collision.events)
            finally:
                backend.close()
            result["session_files"] = [str(p.relative_to(settings.session_root)) for p in settings.session_root.rglob("*") if p.is_file()]
        else:
            request = dict(task_id="task", step_id="step", input_epoch=1, role="main", phase="preparing",
                kind="create", operator="fixture_square", requirement="Square x.", versions={})
            usages = []
            try:
                value = RoleAgent(settings).run(request, on_usage=usages.append)
                result = dict(outcome=value.outcome, usage=value.usage, events=value.harness.events)
            except Exception as error:
                result = dict(error_type=type(error).__name__, usage=usages[-1] if usages else None)
        result.update(profiles=profiles, homes_removed=all(not home.exists() for home in homes))
    result["ambient_unchanged"] = (ambient / "cordis.patch.yml").read_text() == "not a valid patch: PRIVATE_AMBIENT_HOME_MARKER"
print(json.dumps(result))
`
const env = Object.fromEntries(
  Object.entries(process.env).filter(
    ([key]) => !/^(TRITON_|RISCV_|DSH_)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
  ),
)
env.NO_PROXY = '127.0.0.1,localhost'
const answer = JSON.stringify({
  status: 'completed',
  summary: 'Fixture only, no real validation.',
  decision: 'plan_tests',
})
type Usage = {
  input_tokens: number
  output_tokens: number
  total_tokens: number
  input_tokens_details?: { cached_tokens: number }
  output_tokens_details?: { reasoning_tokens: number }
}
const usage: Usage = {
  input_tokens: 120,
  output_tokens: 30,
  total_tokens: 150,
  input_tokens_details: { cached_tokens: 80 },
  output_tokens_details: { reasoning_tokens: 10 },
}
type Reply = { text?: string; usage?: Usage; tool?: string; truncate?: boolean; reject?: boolean; incomplete?: boolean }

function events(reply: Reply, index: number) {
  const text = reply.text ?? answer
  const id = `msg_${index}`
  const response = { id: `resp_${index}`, status: 'in_progress', output: [] }
  if (reply.tool) {
    const item = {
      type: 'function_call',
      id,
      call_id: `call_${index}`,
      name: reply.tool,
      arguments: '{"operator_name":"fixture_square"}',
      status: 'completed',
    }
    return [
      { type: 'response.created', response },
      { type: 'response.output_item.added', output_index: 0, item: { ...item, arguments: '', status: 'in_progress' } },
      { type: 'response.function_call_arguments.delta', item_id: id, output_index: 0, delta: item.arguments },
      { type: 'response.function_call_arguments.done', item_id: id, output_index: 0, arguments: item.arguments },
      { type: 'response.output_item.done', output_index: 0, item },
      {
        type: 'response.completed',
        response: { ...response, status: 'completed', output: [item], usage: reply.usage },
      },
    ]
  }
  const part = { type: 'output_text', text, annotations: [] }
  const item = { type: 'message', id, role: 'assistant', status: 'completed', content: [part] }
  const chunks = [
    { type: 'response.created', response },
    { type: 'response.output_item.added', output_index: 0, item: { ...item, status: 'in_progress', content: [] } },
    {
      type: 'response.content_part.added',
      item_id: id,
      output_index: 0,
      content_index: 0,
      part: { ...part, text: '' },
    },
    { type: 'response.output_text.delta', item_id: id, output_index: 0, content_index: 0, delta: text },
    { type: 'response.output_text.done', item_id: id, output_index: 0, content_index: 0, text },
    { type: 'response.content_part.done', item_id: id, output_index: 0, content_index: 0, part },
    { type: 'response.output_item.done', output_index: 0, item },
    {
      type: reply.incomplete ? 'response.incomplete' : 'response.completed',
      response: {
        ...response,
        status: reply.incomplete ? 'incomplete' : 'completed',
        incomplete_details: reply.incomplete ? { reason: 'max_output_tokens' } : undefined,
        output: [item],
        usage: reply.usage,
      },
    },
  ]
  return reply.truncate ? chunks.slice(0, 4) : chunks
}

async function run(replies: Reply[], chat = false) {
  const requests: Record<string, any>[] = []
  const server = createServer(async (req, res) => {
    const chunks: Buffer[] = []
    for await (const chunk of req) chunks.push(chunk)
    requests.push(JSON.parse(Buffer.concat(chunks).toString()))
    const reply = replies[requests.length - 1]
    if (!reply || reply.reject) {
      res.writeHead(400, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: { message: 'Fixture rejects this request', type: 'invalid_request_error' } }))
      return
    }
    res.writeHead(200, { 'Content-Type': 'text/event-stream' })
    for (const event of events(reply, requests.length))
      res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    res.end()
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  try {
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('Missing fixture port')
    const result = await new Promise<any>((resolve, reject) => {
      const child = spawn(
        python,
        [
          '-I',
          '-B',
          '-c',
          'import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))',
          join(root, 'python'),
          driver,
        ],
        { cwd: root, env },
      )
      let stdout = '',
        stderr = ''
      const timer = setTimeout(() => child.kill('SIGKILL'), 25000)
      child.stdout.on('data', data => {
        stdout += data
      })
      child.stderr.on('data', data => {
        stderr += data
      })
      child.on('error', error => {
        clearTimeout(timer)
        reject(error)
      })
      child.on('close', code => {
        clearTimeout(timer)
        if (code !== 0) reject(new Error(stderr || `SDK fixture exit ${code}`))
        else {
          try {
            resolve(JSON.parse(stdout))
          } catch (error) {
            reject(error)
          }
        }
      })
      child.stdin.end(JSON.stringify({ url: `http://127.0.0.1:${address.port}/v1`, chat }))
    })
    return { result, requests }
  } finally {
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

describe('published SDK accounting and isolation', () => {
  it('preserves authoritative total and positive cached input without double counting', async () => {
    const { result, requests } = await run([{ usage }])
    expect(result.outcome).toBe('validated')
    expect(requests).toHaveLength(1)
    expect(result.usage.totals).toMatchObject({
      uncached_input_tokens: 40,
      input_tokens: 120,
      cache_read_tokens: 80,
      output_tokens: 30,
      total_tokens: 150,
      known_tokens: 150,
      usage_status: 'complete',
    })
    expect(result.usage.totals.reasoning_tokens).toBeNull()
    console.info('SDK_ACCOUNTING', JSON.stringify({ wire: usage, measured: result.usage.totals }))
  }, 30000)
  it('preserves totals when the SDK omits zero cache fields', async () => {
    const { result } = await run([{ usage: { ...usage, input_tokens_details: { cached_tokens: 0 } } }])
    expect(result.usage.totals).toMatchObject({
      input_tokens: 120,
      output_tokens: 30,
      total_tokens: 150,
      cache_read_tokens: null,
    })
  }, 30000)
  it('does not turn absent provider usage into a claim of free execution', async () => {
    const { result } = await run([{}])
    expect(result.outcome).toBe('validated')
    expect(result.usage.totals.total_tokens).toBeNull()
  }, 30000)
  it('retains tokens for an invalid role answer', async () => {
    const { result } = await run([{ text: 'not JSON', usage }])
    expect(result.outcome).toBe('invalid_output')
    expect(result.usage.totals.total_tokens).toBe(150)
  }, 30000)
  it('counts supplied usage even when completion stops at the output limit', async () => {
    const { result } = await run([{ usage, incomplete: true }])
    expect(result.outcome).toBe('incomplete')
    expect(result.usage.totals.total_tokens).toBe(150)
  }, 30000)
  it('keeps rejected and truncated requests unknown, closes their private runtime', async () => {
    for (const reply of [{ reject: true }, { truncate: true }]) {
      const { result } = await run([reply])
      expect(result.usage.totals.total_tokens).toBeNull()
      expect(result.homes_removed).toBe(true)
    }
  }, 60000)
  it('isolates the role policy, workspace, ambient home and tool roster', async () => {
    const { result, requests } = await run([{ usage }])
    expect(requests[0].tools ?? []).toHaveLength(0)
    expect(JSON.stringify(requests)).not.toContain('WORKSPACE_INSTRUCTION_MARKER')
    expect(JSON.stringify(requests)).not.toContain('PRIVATE_AMBIENT_HOME_MARKER')
    expect(JSON.stringify(requests)).toContain('You are the coordinator')
    expect(result.ambient_unchanged).toBe(true)
    expect(result.homes_removed).toBe(true)
    expect(result.profiles[0].manifest.dsh.profile.bundles).toEqual([])
    const config = JSON.stringify(result.profiles)
    expect(config).not.toContain('fixture-key')
    for (const name of ['mcp-client', 'bash-local', 'fs-local', 'agent-instructions'])
      expect(config).not.toContain(name)
  }, 30000)
  it('cannot execute a model-invented domain tool even when it is requested', async () => {
    const { result, requests } = await run([{ tool: 'mcp__triton_riscv__discover_operator', usage }, { usage }])
    expect(requests.every(request => !request.tools?.length)).toBe(true)
    expect(result.homes_removed).toBe(true)
    const serialized = JSON.stringify(result.events ?? [])
    expect(serialized).not.toContain('"discovery_status":"found"')
    expect(serialized).toMatch(/not found|not available|cannot execute|Unknown tool/i)
  }, 30000)
  it('keeps MCP chat working and uses managed history with a new ID after restart', async () => {
    const { result, requests } = await run(
      [
        { tool: 'mcp__triton_riscv__discover_operator', usage },
        { text: 'Repository discovery completed.', usage },
        { text: 'Same session continued.', usage },
        { text: 'Managed conversation continued.', usage },
      ],
      true,
    )
    expect(requests).toHaveLength(4)
    expect(result.collision_reason).toBe('error')
    expect(JSON.stringify(result.collision_events)).toContain('id collision')
    expect(
      requests[0].tools.some((tool: { name: string }) => tool.name === 'mcp__triton_riscv__discover_operator'),
    ).toBe(true)
    expect(JSON.stringify(requests[0])).toContain('WORKSPACE_INSTRUCTION_MARKER')
    expect(JSON.stringify(requests[1])).toContain('function_call_output')
    expect(JSON.stringify(requests[2])).toContain('Repository discovery completed.')
    expect(JSON.stringify(requests[3])).toContain('Repository discovery completed.')
    expect(JSON.stringify(requests[3])).toContain('Same session continued.')
    expect(result.responses).toEqual([
      'Repository discovery completed.',
      'Same session continued.',
      'Managed conversation continued.',
    ])
    expect(result.profiles).toHaveLength(2)
    expect(result.session_files.length).toBeGreaterThan(0)
    expect(result.homes_removed).toBe(true)
    expect(result.ambient_unchanged).toBe(true)
  }, 30000)
})
