// Run with the target Harness source's Vitest configuration (see scripts/test-native-host.mjs).
import { afterEach, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { ToolCallId, createUserMessage } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId } from '@deepseek-ai/dsh-session'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SystemPrompt, { renderContextSnapshot } from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import AgentRegistry, { assembleContextFor, type Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import ApprovalService from '@deepseek-ai/dsh-user-approval'
import {
  mkdtempSync,
  rmSync,
  mkdirSync,
  existsSync,
  readFileSync,
  writeFileSync,
  symlinkSync,
  unlinkSync,
} from 'node:fs'
import { bindScopeParent } from '@deepseek-ai/dsh-scope'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
import { execFileSync, spawn } from 'node:child_process'
// This test is copied into the pinned host's core/tools/tests directory.
import { MockAdapter, toolCallResponse, textResponse } from '../../agent-loop/tests/mock-adapter.ts'

const plugin = process.env.TRITON_PLUGIN_TEST_PACKAGE!
const python = process.env.TRITON_PLUGIN_TEST_PYTHON!
// The host tests use source aliases; resolve the plugin's compiled scope import
// to that same real module, not a second copy with a different scope Symbol.
const hostRequire = createRequire(pathToFileURL(join(process.cwd(), 'packages/mcp/mcp-client/package.json')))
vi.doMock(hostRequire.resolve('@deepseek-ai/dsh-scope'), () => import('@deepseek-ai/dsh-scope'))
const Native = await import(pathToFileURL(join(plugin, 'native.js')).href)
const { configFromEnvironment } = await import(pathToFileURL(join(plugin, 'lib/config.js')).href)
const { createStateStore } = await import(pathToFileURL(join(plugin, 'lib/native-state.js')).href)
const { resolveSessionConfig } = await import(pathToFileURL(join(plugin, 'lib/native-workspace.js')).href)
const { OPERATOR_TOOLS } = await import(pathToFileURL(join(plugin, 'lib/tool-policy.js')).href)
const cleanups: (() => unknown)[] = []
function checkout() {
  const root = mkdtempSync(join(tmpdir(), 'triton-other-checkout-'))
  cleanups.push(() => rmSync(root, { recursive: true, force: true }))
  mkdirSync(join(root, 'python/examples/flaggems'), { recursive: true })
  return root
}
function data(result: any) {
  expect(result.isError, JSON.stringify(result)).toBe(false)
  return result.value.structuredContent
}
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup()
})

async function setup(
  outcome: 'allowed-once' | 'rejected',
  allowValidation = false,
  options = { noRepoRoot: false, initialize: true },
) {
  const root = mkdtempSync(join(tmpdir(), 'triton-native-test-'))
  cleanups.push(() => rmSync(root, { recursive: true, force: true }))
  mkdirSync(join(root, 'python/examples/flaggems'), { recursive: true })
  const env = {
    ...process.env,
    TRITON_RISCV_REPO_ROOT: root,
    TRITON_RISCV_STATE_DIR: join(root, 'state'),
    TRITON_RISCV_MEMORY_DB: join(root, 'state/memory.sqlite3'),
    TRITON_RISCV_MCP_PYTHON: python,
    TRITON_RISCV_ALLOW_DEVELOPMENT_APPLY: '1',
    TRITON_RISCV_ALLOW_VALIDATION: allowValidation ? '1' : '0',
    TRITON_RISCV_EMBEDDING_PROVIDER: 'none',
    TRITON_RISCV_MEMORY_RETRIEVAL_MODE: 'legacy',
  }
  const ctx = new Context()
  cleanups.push(() => ctx.fiber.dispose())
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(SessionStore)
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(AgentRegistry)
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(ApprovalService)
  const questions: string[] = []
  ctx.on('approval/request', request => {
    questions.push(request.reason || '')
    return Promise.resolve(outcome)
  })
  ctx.baseUrl = pathToFileURL(join(process.cwd(), 'packages/mcp/mcp-client/package.json')).href
  const input = configFromEnvironment(env)
  if (options.noRepoRoot) {
    delete input.repoRoot
    delete input.stateDir
    delete input.memory.database
  }
  const nativeFiber = await ctx.plugin(Native, input)
  const handles = new Map()
  async function makeAgent(id: string, cwd = root) {
    const handle = await ctx.agents.create({
      sessionId: SessionId(id),
      meta: { cwd },
      agentOptions: { provider: 'test', model: 'no-live-model' },
    })
    handles.set(handle.agent, handle)
    handle.agent.session.append('turn/start', { turn: 1 })
    return handle.agent
  }
  const agent = await makeAgent('native-test-session')
  let sequence = 0
  async function call(name: string, args: object, owner = agent) {
    await ctx.systemPrompt.assemble(assembleContextFor(owner))
    const result = await ctx.tools.execute({
      name: 'mcp__triton_riscv__' + name,
      arguments: args,
      agent: owner,
      callId: ToolCallId(`native-test-${++sequence}`),
      signal: new AbortController().signal,
    })
    return result
  }
  const fixture = JSON.parse(
    execFileSync(
      python,
      [
        '-I',
        '-c',
        'import json; from codex_agent.tests.test_operator_development import valid_spec, IMPLEMENTATION, TEST_SOURCE; print(json.dumps(dict(spec=valid_spec(),implementation=IMPLEMENTATION,test=TEST_SOURCE)))',
      ],
      { encoding: 'utf8' },
    ),
  )
  execFileSync(
    python,
    [
      '-I',
      '-c',
      [
        'import os',
        'from pathlib import Path',
        'from codex_agent.memory import MemoryRecord, MemoryStore',
        'with MemoryStore(Path(os.environ["TRITON_RISCV_MEMORY_DB"])) as store:',
        '    store.add(MemoryRecord(memory_type="failure-diagnosis", operator="square_new", semantics="Compute the elementwise square of the input tensor.", pytorch_reference="torch.square(x)", summary="Synthetic native integration evidence; not a real validation result.", outcome="failed", confidence_grade="C", source_run="synthetic:native-integration", evidence={"recommended_actions":["Check the mask on the last block."]}))',
      ].join('\n'),
    ],
    { env, encoding: 'utf8' },
  )
  if (options.initialize) await ctx.systemPrompt.assemble(assembleContextFor(agent))
  return { ctx, agent, call, fixture, root, questions, env, input, nativeFiber, makeAgent, handles }
}

it('disabling the configured plugin removes its MCP tools and context effects', async () => {
  const h = await setup('rejected')
  expect(
    h.ctx.tools
      .schemas(h.agent)
      .filter(t => t.name.startsWith('mcp__triton_riscv__'))
      .map(t => t.name)
      .sort(),
  ).toEqual([...OPERATOR_TOOLS].sort())
  await h.nativeFiber.dispose()
  expect(h.ctx.tools.schemas(h.agent).filter(t => t.name.startsWith('mcp__triton_riscv__'))).toHaveLength(0)
  expect(renderContextSnapshot(await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent)))).not.toContain(
    'triton_task_evidence',
  )
}, 60_000)

it('real host guard denies generic bypass after activation, survives restore and disposes cleanly', async () => {
  const h = await setup('allowed-once')
  let calls = 0
  for (const name of ['bash', 'edit', 'delegate', 'renamed_shell', 'mcp__triton_riscv__fake']) {
    h.ctx.tools.register({
      name,
      description: 'Inert bypass fixture',
      parameters: { type: 'object', properties: {} },
      output: { schema: { type: 'null' }, render: () => [] },
      async execute() {
        calls++
        return null
      },
    })
  }
  // An extensible policy returning allow cannot override a monotonic denial.
  h.ctx.on('tools/pre-execute', async () => ({ kind: 'allow' as const }))
  const raw = (name: string, owner = h.agent) =>
    h.ctx.tools.execute({
      name,
      arguments: {},
      agent: owner,
      callId: ToolCallId('bypass-' + name),
      signal: new AbortController().signal,
    })
  expect((await raw('bash')).isError).toBe(false)
  expect((await h.call('inspect_project', {})).isError).toBe(false)
  for (const name of ['bash', 'edit', 'delegate', 'renamed_shell', 'mcp__triton_riscv__fake']) {
    const result = await raw(name)
    expect(result.isError).toBe(true)
    expect(JSON.stringify(result)).toContain('restricted to typed')
  }
  expect(calls).toBe(1)
  await h.handles.get(h.agent).dispose()
  const restored = await h.makeAgent(h.agent.session.id)
  await h.ctx.systemPrompt.assemble(assembleContextFor(restored))
  expect((await raw('bash', restored)).isError).toBe(true)
  expect(createStateStore(h.env).load(h.agent.session.id).operator_tools_only).toBe(true)
  const other = await h.makeAgent('unrelated')
  expect((await raw('bash', other)).isError).toBe(false)
  await h.nativeFiber.dispose()
  expect((await raw('bash', restored)).isError).toBe(false)
  expect(calls).toBe(3)
}, 60_000)

it('real host + real stdio MCP: prepares, approves and applies in an isolated checkout', async () => {
  const h = await setup('allowed-once')
  expect(h.ctx.tools.schemas(h.agent).filter(t => t.name.startsWith('mcp__triton_riscv__'))).toHaveLength(20)
  const prepare = await h.call('prepare_operator_development', {
    specification: h.fixture.spec,
  })
  expect(prepare.isError).toBe(false)
  const developmentId = (prepare as any).value.structuredContent.development_id
  expect(developmentId).toBeTruthy()
  const initialContext = renderContextSnapshot(await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent)))
  expect(initialContext).toContain('synthetic:native-integration')
  expect(initialContext).toContain('Check the mask')
  const proposed = await h.call('propose_operator_implementation', {
    development_id: developmentId,
    implementation_source: h.fixture.implementation,
    test_source: h.fixture.test,
    rationale: 'host integration fixture',
  })
  expect(proposed.isError).toBe(false)
  const proposalId = (proposed as any).value.structuredContent.proposal_id
  expect(existsSync(join(h.root, 'python/examples/flaggems/square_new.py'))).toBe(false)
  const result = await h.call('apply_development_proposal', {
    proposal_id: proposalId,
  })
  expect(result.isError).toBe(false)
  expect((result as any).value.structuredContent.status).toBe('applied')
  expect(readFileSync(join(h.root, 'python/examples/flaggems/square_new.py'), 'utf8')).toContain('@triton.jit')
  const source = await h.call('read_operator_file', { path: 'python/examples/flaggems/square_new.py' })
  expect(source.isError).toBe(false)
  expect((source as any).value.structuredContent.content).toContain('@triton.jit')
  expect(h.questions).toHaveLength(1)
  expect(h.questions[0]).toContain('test_square_new')
  const replay = await h.call('apply_development_proposal', { proposal_id: proposalId })
  expect(replay.isError).toBe(false)
  expect((replay as any).value.structuredContent).toEqual((result as any).value.structuredContent)
  expect(h.questions).toHaveLength(1)
  const assembly = await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent))
  const text = renderContextSnapshot(assembly)
  expect(text).toContain(proposalId)
  expect(text).toContain('torch.square(x)')
  expect(text).toContain('not_validated_after_change')
  expect(h.agent.session.snapshotEvents().some(e => e.type === 'approval/decided')).toBe(true)
  const restored = createStateStore(h.env).load(h.agent.session.id)
  expect(restored.artifacts[proposalId]).toBe('development')
  expect(restored.task.validation_status).toBe('not_validated_after_change')
  expect(createStateStore(h.env).load('another-session').artifacts).toEqual({})
  restored.memory = { text: 'literal compiler tokens {{unknown}} remain data' }
  createStateStore(h.env).save(restored)
  expect(renderContextSnapshot(await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent)))).toContain('{{unknown}}')
  const planned = await h.call('validate_operator', { operator_name: 'square_new', execute: false })
  expect(planned.isError).toBe(false)
  const planId = (planned as any).value.structuredContent.run_id
  const beforeStatus = createStateStore(h.env).load(h.agent.session.id)
  const status = await h.call('get_validation_status', { run_id: planId })
  expect(status.isError).toBe(false)
  expect((status as any).value.structuredContent.execution_state).toBe('not-started')
  expect((status as any).value.structuredContent.test_status).toBe(null)
  expect(createStateStore(h.env).load(h.agent.session.id)).toEqual(beforeStatus)
  expect(h.questions).toHaveLength(1)
  const foreign = await h.makeAgent('foreign-status')
  expect((await h.call('get_validation_status', { run_id: planId }, foreign)).isError).toBe(true)
}, 60_000)

it('real host waits for another writer, asks for the new baseline and never applies the old proposal', async () => {
  const h = await setup('allowed-once', true)
  h.agent.session.append(
    'user/message',
    createUserMessage({
      source: { kind: 'user' },
      content: [{ type: 'text', text: "Create square_new without overwriting someone else's work." }],
    }),
    { surfaceOp: 'append' },
  )
  const prepare = await h.call('prepare_operator_development', { specification: h.fixture.spec })
  const proposed = await h.call('propose_operator_implementation', {
    development_id: (prepare as any).value.structuredContent.development_id,
    implementation_source: h.fixture.implementation,
    test_source: h.fixture.test,
    rationale: 'stale B proposal fixture',
  })
  expect(proposed.isError).toBe(false)
  const id = (proposed as any).value.structuredContent.proposal_id
  const implementationPath = join(h.root, 'python/examples/flaggems/square_new.py')
  const testPath = join(h.root, 'python/examples/flaggems/test_square_new.py')
  const writer = spawn(
    python,
    [
      '-I',
      '-u',
      '-c',
      [
        'import sys,time,json',
        'from pathlib import Path',
        'from codex_agent.execution_guard import resource_locks',
        'root=Path(sys.argv[1]); paths=[Path(sys.argv[2]),Path(sys.argv[3])]',
        'with resource_locks(root, ["file:"+str(p.resolve()) for p in paths]):',
        '    print("locked",flush=True)',
        '    time.sleep(0.8)',
        '    paths[0].write_text(sys.argv[4]+"\\n# A completed a newer version\\n")',
        '    paths[1].write_text(sys.argv[5])',
      ].join('\n'),
      h.root,
      implementationPath,
      testPath,
      h.fixture.implementation,
      h.fixture.test,
    ],
    { env: h.env },
  )
  cleanups.push(() => {
    if (writer.exitCode === null) writer.kill('SIGKILL')
  })
  const finished = new Promise<number | null>((resolve, reject) => {
    writer.once('error', reject)
    writer.once('exit', resolve)
  })
  await new Promise<void>((resolve, reject) => {
    writer.stdout!.once('data', () => resolve())
    writer.once('error', reject)
    writer.once('exit', code => reject(new Error('Writer exited before locking: ' + code)))
  })
  const result = await h.call('apply_development_proposal', { proposal_id: id })
  expect(await finished).toBe(0)
  expect(result.isError, JSON.stringify(result)).toBe(false)
  const data = (result as any).value.structuredContent
  expect(data.status).toBe('replanned')
  expect(data.followup_plan.status).toBe('planned')
  expect(data.followup_plan.run_id).toBeTruthy()
  expect(h.questions).toHaveLength(1)
  expect(h.questions[0]).toContain('replan_from_latest_sources')
  expect(h.questions[0]).toContain('Create square_new without overwriting')
  expect(readFileSync(implementationPath, 'utf8')).toContain('A completed a newer version')
  expect(existsSync(join(h.root, 'tasks/operators/square_new.md'))).toBe(false)
  expect(createStateStore(h.env).load(h.agent.session.id).active.plan).toBe(data.followup_plan.run_id)
  expect((await h.call('apply_development_proposal', { proposal_id: id })).isError).toBe(true)
}, 60_000)

it('real MCP project validation requires native approval and keeps a failing result', async () => {
  const h = await setup('allowed-once', true)
  writeFileSync(join(h.root, 'python/examples/test_local.py'), 'def test_failure():\n    assert False\n')
  const planned = await h.call('prepare_validation_job', {
    targets: ['pytest::python/examples/test_local.py'],
    kind: 'project',
    source_env: false,
  })
  expect(planned.isError).toBe(false)
  const id = (planned as any).value.structuredContent.job_id
  expect(h.questions).toHaveLength(0)
  const other = await h.makeAgent('foreign-job-session')
  expect((await h.call('execute_validation_job', { job_id: id }, other)).isError).toBe(true)
  expect(h.questions).toHaveLength(0)
  const result = await h.call('execute_validation_job', { job_id: id })
  expect(result.isError).toBe(false)
  expect(h.questions).toHaveLength(1)
  const data = (result as any).value.structuredContent
  expect(data.status).toBe('failed')
  expect(data.results[0].exit_code).toBe(1)
  expect(readFileSync(data.results[0].log_path, 'utf8')).toContain('1 failed')
  const repeated = await h.call('execute_validation_job', { job_id: id })
  expect(repeated.isError).toBe(false)
  expect((repeated as any).value.structuredContent).toEqual(data)
  expect(h.questions).toHaveLength(1)
}, 60_000)

it('real native rejection and session isolation leave target files untouched', async () => {
  const h = await setup('rejected')
  const prepare = await h.call('prepare_operator_development', {
    specification: h.fixture.spec,
  })
  const proposed = await h.call('propose_operator_implementation', {
    development_id: (prepare as any).value.structuredContent.development_id,
    implementation_source: h.fixture.implementation,
    test_source: h.fixture.test,
    rationale: 'rejection fixture',
  })
  const id = (proposed as any).value.structuredContent.proposal_id
  const other = await h.makeAgent('other-session')
  const foreign = await h.call('apply_development_proposal', { proposal_id: id }, other)
  expect(foreign.isError).toBe(true)
  expect(h.questions).toHaveLength(0)
  const rejected = await h.call('apply_development_proposal', {
    proposal_id: id,
  })
  expect(rejected.isError).toBe(true)
  expect(h.questions).toHaveLength(1)
  expect(existsSync(join(h.root, 'python/examples/flaggems/square_new.py'))).toBe(false)
}, 60_000)

it('real AgentLoop sends retrieved evidence to the next model request and keeps it across turns', async () => {
  const h = await setup('rejected')
  const adapter = new MockAdapter([
    toolCallResponse('prepare-1', 'mcp__triton_riscv__prepare_operator_development', { specification: h.fixture.spec }),
    textResponse('Prepared; no code has been applied or validated.'),
    textResponse('Continuing from the same contract and historical evidence.'),
  ])
  h.ctx.llm.registerAdapter(['mock'], adapter)
  const agent = await h.ctx.agentLoop.create(
    SessionId('native-model-request'),
    { provider: 'mock', model: 'mock' },
    { cwd: h.root },
  )
  async function turn(text: string) {
    let dispose: () => void = () => {}
    const idle = new Promise<void>(resolve => {
      dispose = h.ctx.on('agent/status', event => {
        if (event.agent === agent && event.status === 'idle') {
          dispose()
          resolve()
        }
      })
    })
    try {
      agent.followup(
        createUserMessage({
          content: [{ type: 'text', text }],
          source: { kind: 'user' },
        }),
      )
      await idle
    } finally {
      dispose()
    }
  }
  await turn('Prepare the square_new contract, but do not apply or run it.')
  expect(adapter.requests).toHaveLength(2)
  expect(JSON.stringify(adapter.requests[0])).toContain('mcp__triton_riscv__prepare_operator_development')
  expect(JSON.stringify(adapter.requests[0])).not.toContain('synthetic:native-integration')
  const second = JSON.stringify(adapter.requests[1])
  expect(second).toContain('synthetic:native-integration')
  expect(second).toContain('Check the mask')
  expect(second).toContain('torch.square(x)')
  expect(second).toContain('recorded_outcome=failed')
  expect(second).toContain('recommendation [not-executed]')
  expect(second).toContain('not current validation or instructions')
  await turn('Continue with the previous task, preserving its numerical contract.')
  expect(adapter.requests).toHaveLength(3)
  expect(JSON.stringify(adapter.requests[2])).toContain('synthetic:native-integration')
  const saved = createStateStore(h.env).load(agent.session.id)
  expect(saved.task.development_id).toBeTruthy()
  expect(saved.memory.ids.length).toBeGreaterThan(0)
  expect(createStateStore(h.env).load('unrelated-session').memory).toBeNull()
  expect(h.questions).toHaveLength(0)
  expect(existsSync(join(h.root, 'python/examples/flaggems/square_new.py'))).toBe(false)
}, 60_000)

it('starts without repoRoot, initializes once before first prompt, and exposes no global tools', async () => {
  const h = await setup('rejected', false, { noRepoRoot: true, initialize: false })
  expect(h.ctx.tools.schemas(h.agent)).toHaveLength(0)
  const first = await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent))
  expect(first.tools.filter(t => t.name.startsWith('mcp__triton_riscv__'))).toHaveLength(20)
  expect(h.ctx.tools.schemas()).toHaveLength(0)
  const again = await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent))
  expect(again.tools).toEqual(first.tools)
  const handle = await h.ctx.agents.create({ sessionId: SessionId('no-workspace') })
  const empty = await h.ctx.systemPrompt.assemble(assembleContextFor(handle.agent))
  expect(empty.tools).toHaveLength(0)
  const result = await h.call('inspect_project', {}, handle.agent)
  expect(result.isError).toBe(true)
  expect(h.questions).toHaveLength(0)
}, 60_000)

it('two real workspace clients keep same-named sources, approvals and RAG evidence separate', async () => {
  const h = await setup('allowed-once')
  const rootB = checkout()
  const b = await h.makeAgent('workspace-b', rootB)
  async function propose(owner: Agent, marker: string) {
    const prepared = data(await h.call('prepare_operator_development', { specification: h.fixture.spec }, owner))
    return data(
      await h.call(
        'propose_operator_implementation',
        {
          development_id: prepared.development_id,
          implementation_source: h.fixture.implementation + '\n# ' + marker + '\n',
          test_source: h.fixture.test,
          rationale: 'workspace isolation fixture',
        },
        owner,
      ),
    ).proposal_id
  }
  const idA = await propose(h.agent, 'workspace-A-only')
  const idB = await propose(b, 'workspace-B-only')
  const contextA = renderContextSnapshot(await h.ctx.systemPrompt.assemble(assembleContextFor(h.agent)))
  const contextB = renderContextSnapshot(await h.ctx.systemPrompt.assemble(assembleContextFor(b)))
  expect(contextA).toContain('synthetic:native-integration')
  expect(contextB).not.toContain('synthetic:native-integration')
  expect(contextB).not.toContain(idA)
  expect(resolveSessionConfig(h.input, b.session).memory.database).not.toBe(h.env.TRITON_RISCV_MEMORY_DB)
  const foreign = await h.call('apply_development_proposal', { proposal_id: idA }, b)
  expect(foreign.isError).toBe(true)
  expect(h.questions).toHaveLength(0)
  expect(data(await h.call('apply_development_proposal', { proposal_id: idA })).status).toBe('applied')
  expect(existsSync(join(rootB, 'python/examples/flaggems/square_new.py'))).toBe(false)
  expect(data(await h.call('apply_development_proposal', { proposal_id: idB }, b)).status).toBe('applied')
  expect(h.questions).toHaveLength(2)
  const args = { path: 'python/examples/flaggems/square_new.py' }
  const both = await Promise.all([h.call('read_operator_file', args), h.call('read_operator_file', args, b)])
  expect(data(both[0]).content).toContain('workspace-A-only')
  expect(data(both[0]).content).not.toContain('workspace-B-only')
  expect(data(both[1]).content).toContain('workspace-B-only')
  expect(data(both[1]).content).not.toContain('workspace-A-only')
  await h.handles.get(h.agent).dispose()
  expect(data(await h.call('read_operator_file', args, b)).content).toContain('workspace-B-only')
  const restartedElsewhere = await h.makeAgent(h.agent.session.id, rootB)
  const moved = await h.call('apply_development_proposal', { proposal_id: idA }, restartedElsewhere)
  expect(moved.isError).toBe(true)
  expect(h.questions).toHaveLength(2)
}, 60_000)

it('executes the same relative pytest target in two workspaces with different real results', async () => {
  const h = await setup('allowed-once', true, { noRepoRoot: true, initialize: true })
  const rootB = checkout(),
    b = await h.makeAgent('workspace-b-tests', rootB)
  writeFileSync(join(h.root, 'python/examples/test_local.py'), 'def test_workspace():\n    assert True\n')
  writeFileSync(join(rootB, 'python/examples/test_local.py'), 'def test_workspace():\n    assert False\n')
  const request = { targets: ['pytest::python/examples/test_local.py'], kind: 'project', source_env: false }
  const planA = data(await h.call('prepare_validation_job', request))
  const planB = data(await h.call('prepare_validation_job', request, b))
  const cross = await h.call('execute_validation_job', { job_id: planA.job_id }, b)
  expect(cross.isError).toBe(true)
  expect(h.questions).toHaveLength(0)
  const aResult = data(await h.call('execute_validation_job', { job_id: planA.job_id }))
  const bResult = data(await h.call('execute_validation_job', { job_id: planB.job_id }, b))
  expect(aResult.status).toBe('passed')
  expect(bResult.status).toBe('failed')
  expect(aResult.results[0].exit_code).toBe(0)
  expect(bResult.results[0].exit_code).toBe(1)
  expect(readFileSync(aResult.results[0].log_path, 'utf8')).toContain('1 passed')
  expect(readFileSync(bResult.results[0].log_path, 'utf8')).toContain('1 failed')
  expect(h.questions).toHaveLength(2)
}, 60_000)

it('rejects a workspace symlink retarget instead of using stale tools or approvals', async () => {
  const h = await setup('rejected')
  const other = checkout(),
    alias = join(h.root, 'workspace-alias')
  symlinkSync(h.root, alias)
  const owner = await h.makeAgent('symlink-session', alias)
  await h.ctx.systemPrompt.assemble(assembleContextFor(owner))
  unlinkSync(alias)
  symlinkSync(other, alias)
  await expect(h.ctx.systemPrompt.assemble(assembleContextFor(owner))).rejects.toThrow('workspace changed')
  const result = await h.ctx.tools.execute({
    name: 'mcp__triton_riscv__inspect_project',
    arguments: {},
    agent: owner,
    callId: ToolCallId('retarget'),
    signal: new AbortController().signal,
  })
  expect(result.isError).toBe(true)
  expect(JSON.stringify(result)).toContain('workspace changed')
  expect(h.questions).toHaveLength(0)
}, 60_000)

it('does not let a nested agent borrow its parent workspace connection or context', async () => {
  const h = await setup('rejected')
  await h.call('prepare_operator_development', { specification: h.fixture.spec })
  const rootB = checkout(),
    child = await h.makeAgent('nested-workspace', rootB)
  bindScopeParent(child, h.agent)
  const before = await h.ctx.tools.execute({
    name: 'mcp__triton_riscv__inspect_project',
    arguments: {},
    agent: child,
    callId: ToolCallId('before-child-mount'),
    signal: new AbortController().signal,
  })
  expect(before.isError).toBe(true)
  const assembly = await h.ctx.systemPrompt.assemble(assembleContextFor(child))
  expect(assembly.tools.filter(t => t.name.startsWith('mcp__triton_riscv__'))).toHaveLength(20)
  expect(renderContextSnapshot(assembly)).not.toContain('synthetic:native-integration')
  expect(data(await h.call('discover_operator', { operator_name: 'square_new' }, child)).status).toBe('not_found')
}, 60_000)
