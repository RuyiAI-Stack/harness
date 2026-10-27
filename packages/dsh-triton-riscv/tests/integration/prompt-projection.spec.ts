import { describe, expect, it } from 'vitest'
import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createServer } from 'node:http'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python')
const driver = `
import importlib.util, json, sys, tempfile
from pathlib import Path
from codex_agent.orchestration.prompts import task_prompt, system_prompt, PROMPT_VERSION
from codex_agent.orchestration.roles import RoleRequest
from codex_agent.orchestration.runtime import RoleAgent
from codex_agent.harness.config import HarnessSettings
data = json.load(sys.stdin)
try:
    request = RoleRequest.model_validate(data["request"])
    original = request.model_dump(mode="json")
    if data.get("mutate"):
        request.files[0] = request.files[0].model_copy(update={
            "content": request.files[0].content + "mutated after validation"})
    prompt = task_prompt(request)
    result = dict(prompt=prompt, projection=json.loads(prompt.split("\\n", 1)[1]),
                  original=original, after=request.model_dump(mode="json"),
                  policy=system_prompt(request.role), version=PROMPT_VERSION)
    if data.get("baseline"):
        spec = importlib.util.spec_from_file_location("codex_agent.orchestration._prompt_baseline", data["baseline"])
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
        try:
            old = before.task_prompt(request)
            result["before"] = dict(prompt=old, policy=before.system_prompt(request.role))
        except ValueError as error:
            result["before_error"] = str(error)
    if data.get("base_url"):
        with tempfile.TemporaryDirectory() as folder:
            settings = HarnessSettings.from_env(Path(folder), {"ISRC_API_KEY": "fixture-key",
                "DSH_MODEL": "fixture-model", "ISRC_BASE_URL": data["base_url"],
                "DSH_REQUEST_TIMEOUT_SECONDS": "15"})
            value = RoleAgent(settings).run(request)
            result["outcome"] = value.outcome
            result["verdict"] = value.result.verdict if value.result else None
    print(json.dumps(result))
except ValueError as error:
    print(json.dumps(dict(error=str(error))))
`
const args = [
  '-I',
  '-B',
  '-c',
  'import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))',
  join(root, 'python'),
  driver,
]
const env = Object.fromEntries(
  Object.entries(process.env).filter(
    ([key]) => !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) && !/^(TRITON_|RISCV_|DSH_)/.test(key),
  ),
)
env.NO_PROXY = '127.0.0.1,localhost'

function invoke(request: object, extra = {}) {
  const result = spawnSync(python, args, {
    input: JSON.stringify({ request, ...extra }),
    cwd: root,
    env,
    encoding: 'utf8',
    timeout: 30000,
    maxBuffer: 8 * 1024 * 1024,
  })
  expect(result.error).toBeUndefined()
  expect(result.status, result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

const hash = (content: string) => createHash('sha256').update(content).digest('hex')
const sourcePath = 'python/examples/flaggems/fixture_square.py'
const testPath = 'python/examples/flaggems/test_fixture_square.py'
const tests = 'def test_fixed():\n    assert fixture_square(2) == 4\n'
const source = 'def fixture_square(x):\n    # NEW_SOURCE_MARKER\n    return x * x\n'
const snapshot = (path: string, content: string) => ({ path, content, sha256: hash(content) })
function review(kind: 'create' | 'repair' = 'create', content = source) {
  const baseline =
    kind === 'repair' ? snapshot(sourcePath, 'def fixture_square(x):\n    return x  # OLD_BASELINE\n') : null
  return {
    task_id: 'task',
    step_id: 'review-step',
    input_epoch: 1,
    role: 'reviewer',
    phase: 'reviewing',
    kind,
    operator: 'fixture_square',
    requirement: 'Square x; preserve the fixed tests and tolerance.',
    versions: { contract_sha256: hash('contract'), source_sha256: hash(content), tests_sha256: hash(tests) },
    evidence: [
      {
        id: 'contract',
        kind: 'contract',
        text: 'y=x*x; float32; rtol=1e-4, atol=1e-4',
        source: 'fixture contract',
        verified: false,
      },
      {
        id: 'failure',
        kind: 'log',
        text: 'Original test: expected 4, got 2. Not a result for this proposal.',
        source: 'fixture old receipt',
        verified: false,
      },
      {
        id: 'history',
        kind: 'historical',
        text: 'Recommendation: check boundary masks. Not a verified fix.',
        source: 'fixture RAG',
        verified: false,
      },
    ],
    files: [snapshot(sourcePath, content), snapshot(testPath, tests)],
    candidate: { path: sourcePath, content, base_sha256: baseline?.sha256 ?? null },
    review_baseline: baseline,
  }
}
function expand(projection: ReturnType<typeof invoke>['projection']) {
  const result = structuredClone(projection)
  for (const file of result.files) {
    if (file.content_ref) {
      expect(file.content_ref).toBe('/candidate/content')
      file.content = result.candidate.content
      delete file.content_ref
    }
  }
  return result
}

describe('lossless role prompt projection', () => {
  it.each(['create', 'repair'] as const)(
    'sends candidate text once for %s and reconstructs the full original',
    kind => {
      const result = invoke(review(kind))
      expect(result.error).toBeUndefined()
      expect(result.prompt.match(/NEW_SOURCE_MARKER/g)).toHaveLength(1)
      expect(result.projection.files[0]).toEqual({
        path: sourcePath,
        sha256: hash(source),
        content_ref: '/candidate/content',
      })
      expect(expand(result.projection)).toEqual(result.original)
      expect(result.after).toEqual(result.original)
      expect(result.projection.files[1].content).toBe(tests)
      expect(result.projection.review_baseline).toEqual(result.original.review_baseline)
      expect(result.projection.evidence).toEqual(result.original.evidence)
    },
  )
  it('does not expand a very short body into a longer reference', () => {
    const result = invoke(review('create', 'x'))
    expect(result.projection.files[0].content).toBe('x')
    expect(result.projection.files[0]).not.toHaveProperty('content_ref')
  })
  it.each(['中'.repeat(2000), 'x'.repeat(8000), 'line\n"quoted"\\path\n'.repeat(1000)])(
    'preserves Unicode, unspaced and escaped bodies',
    content => {
      const result = invoke(review('repair', content))
      expect(expand(result.projection)).toEqual(result.original)
    },
  )
  it('resolves by the validated path, not the position of a file', () => {
    const request = review()
    request.files.reverse()
    const result = invoke(request)
    expect(result.projection.files[0].content).toBe(tests)
    expect(result.projection.files[1].content_ref).toBe('/candidate/content')
    expect(expand(result.projection)).toEqual(result.original)
  })
  it.each(['body', 'base', 'digest', 'path', 'missing', 'foreign_ref'])(
    'rejects an inconsistent %s before projection',
    field => {
      const request = review('repair')
      if (field === 'body') request.candidate.content = 'different'
      if (field === 'base') request.candidate.base_sha256 = hash('wrong base')
      if (field === 'digest') request.files[0].sha256 = hash('wrong digest')
      if (field === 'path') request.candidate.path = 'another/operator.py'
      if (field === 'missing') request.files = request.files.slice(1)
      if (field === 'foreign_ref') Object.assign(request.files[0], { content_ref: '/another_task/content' })
      expect(invoke(request).error).toBeTruthy()
    },
  )
  it('revalidates nested objects mutated after Pydantic construction', () => {
    expect(invoke(review(), { mutate: true }).error).toContain('digest')
  })
  it('keeps large unique evidence subject to the existing byte limit', () => {
    const request = review('repair', 'x'.repeat(47000))
    request.evidence = Array.from({ length: 4 }, (_, i) => ({
      id: `e${i}`,
      kind: 'log',
      text: 'z'.repeat(7500),
      source: 'fixture',
      verified: false,
    }))
    expect(invoke(request).error).toContain('64 KiB')
  })
  it.each(['main', 'semantic', 'reviewer', 'implementer'])(
    'preserves the existing non-review projection for %s',
    role => {
      const request = {
        ...review('repair'),
        role,
        phase: role === 'implementer' ? 'implementing' : role === 'reviewer' ? 'planning_tests' : 'preparing',
        candidate: null,
        review_baseline: null,
      }
      const result = invoke(request)
      const expected = structuredClone(result.original)
      if (role !== 'implementer')
        expected.files = expected.files.filter((file: { path: string }) => file.path === testPath)
      if (role === 'main') expected.files = []
      expect(result.projection).toEqual(expected)
    },
  )
  it('leaves legacy candidate-less review readable without creating a reference', () => {
    const result = invoke({ ...review(), candidate: null })
    expect(result.projection).toEqual(result.original)
  })
  it('keeps isolation, reference interpretation and approval rules in the system policy', () => {
    const result = invoke(review())
    expect(result.version).toBe('operator-roles-v8')
    for (const instruction of [
      'cannot approve changes',
      'No tools are available',
      'Do not weaken acceptance criteria',
      'already in this same input',
      'Do not rewrite acceptance tests',
    ])
      expect(result.policy).toContain(instruction)
  })
})

describe('SDK prompt transport', () => {
  it('sends the projected review through the installed SDK without exposing tools', async () => {
    const requests: object[] = []
    const text = JSON.stringify({ status: 'completed', summary: 'Synthetic review only', verdict: 'acceptable' })
    const item = {
      id: 'msg_projection',
      type: 'message',
      role: 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text, annotations: [] }],
    }
    const server = createServer(async (req, res) => {
      const body: Buffer[] = []
      for await (const chunk of req) body.push(chunk)
      requests.push(JSON.parse(Buffer.concat(body).toString()))
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      const events = [
        { type: 'response.created', response: { id: 'resp_projection', status: 'in_progress', output: [] } },
        { type: 'response.output_item.added', output_index: 0, item: { ...item, status: 'in_progress', content: [] } },
        {
          type: 'response.content_part.added',
          item_id: item.id,
          output_index: 0,
          content_index: 0,
          part: { type: 'output_text', text: '', annotations: [] },
        },
        { type: 'response.output_text.delta', item_id: item.id, output_index: 0, content_index: 0, delta: text },
        { type: 'response.output_text.done', item_id: item.id, output_index: 0, content_index: 0, text },
        {
          type: 'response.content_part.done',
          item_id: item.id,
          output_index: 0,
          content_index: 0,
          part: item.content[0],
        },
        { type: 'response.output_item.done', output_index: 0, item },
        {
          type: 'response.completed',
          response: {
            id: 'resp_projection',
            status: 'completed',
            output: [item],
            usage: { input_tokens: 32, output_tokens: 16, total_tokens: 48 },
          },
        },
      ]
      for (const event of events) res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      res.end()
    })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const address = server.address()
      if (!address || typeof address === 'string') throw new Error('Missing fixture listener')
      const result = await new Promise<ReturnType<typeof invoke>>((resolve, reject) => {
        const child = spawn(python, args, { cwd: root, env })
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
          if (code !== 0) reject(new Error(stderr || `Child exit ${code}`))
          else {
            try {
              resolve(JSON.parse(stdout))
            } catch (error) {
              reject(error)
            }
          }
        })
        child.stdin.end(JSON.stringify({ request: review('repair'), base_url: `http://127.0.0.1:${address.port}/v1` }))
      })
      expect(result.error).toBeUndefined()
      expect(result.outcome).toBe('validated')
      expect(result.verdict).toBe('acceptable')
      expect(requests).toHaveLength(1)
      expect((requests[0] as { tools?: object[] }).tools ?? []).toHaveLength(0)
      expect(JSON.stringify(requests[0]).match(/NEW_SOURCE_MARKER/g)).toHaveLength(1)
      expect(JSON.stringify(requests[0])).toContain('OLD_BASELINE')
      expect(JSON.stringify(requests[0])).toContain('content_ref')
    } finally {
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }, 35000)
})

describe.runIf(Boolean(process.env.TRITON_PROMPT_BASELINE))(
  'frozen v7 comparison (synthetic, not quality evaluation)',
  () => {
    it.each([
      ['short', 'x'],
      ['medium', source.repeat(64)],
      ['long', source.repeat(240)],
      ['chinese', '# 注释\n'.repeat(1000)],
    ])('%s: retains all evidence and reports actual serialized bytes', (name, content) => {
      const result = invoke(review('repair', content), { baseline: process.env.TRITON_PROMPT_BASELINE })
      expect(result.error).toBeUndefined()
      expect(result.before_error).toBeUndefined()
      const before = JSON.parse(result.before.prompt.split('\n').slice(1).join('\n'))
      expect(expand(result.projection)).toEqual(before)
      const bytes = (text: string) => Buffer.byteLength(text, 'utf8')
      const measurement = {
        name,
        body_bytes: bytes(content),
        before_task_bytes: bytes(result.before.prompt),
        after_task_bytes: bytes(result.prompt),
        before_combined_bytes: bytes(result.before.policy + '\n' + result.before.prompt),
        after_combined_bytes: bytes(result.policy + '\n' + result.prompt),
        evidence_equal: true,
      }
      console.info('PROJECTION_COMPARISON', JSON.stringify(measurement))
      expect(measurement.after_task_bytes).toBeLessThanOrEqual(measurement.before_task_bytes)
      if (name !== 'short') expect(measurement.after_combined_bytes).toBeLessThan(measurement.before_combined_bytes)
    })
    it('allows a duplicated large body to fit without raising the 64-KiB limit or losing bytes', () => {
      const result = invoke(review('create', 'x'.repeat(40000)), { baseline: process.env.TRITON_PROMPT_BASELINE })
      expect(result.error).toBeUndefined()
      expect(result.before_error).toContain('64 KiB')
      expect(expand(result.projection)).toEqual(result.original)
      expect(result.projection.candidate.content).toHaveLength(40000)
    })
  },
)
