import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import { test } from 'vitest'

import { TRITON_RISCV_SYSTEM_PROMPT, apply, inject, name, resolveWorkbenchLaunch } from '../../index.js'
import { PROMPT_SECTIONS } from '../../lib/prompts.js'

const packageRoot = new URL('../../', import.meta.url)

test('keeps the native, workbench and domain paths without the retired orchestrator', async () => {
  const backend = new URL('python/codex_agent/', packageRoot)
  const files = await readdir(backend, { recursive: true })
  const sources = files.filter(file => !file.split('/').includes('__pycache__'))
  assert.ok(
    !sources.some(file => /(^|\/)langgraph_agent(\/|$)|test_langgraph_agent\.py$|langgraph-agent\.md$/.test(file)),
  )
  const metadata = await readFile(new URL('python/pyproject.toml', packageRoot), 'utf8')
  assert.doesNotMatch(metadata, /langgraph|langchain/i)
  for (const required of [
    'harness/mcp_server.py',
    'harness/native_bridge.py',
    'platform/api.py',
    'platform/conversation_context.py',
    'operator_development.py',
    'operator_lifecycle.py',
    'remote_executor.py',
    'memory.py',
    'memory_chunking.py',
    'memory_evidence.py',
    'memory_selection.py',
    'core/workflow.py',
    'adapters/codex_cli.py',
  ])
    assert.ok(sources.includes(required), `Missing retained capability: ${required}`)
  for (const file of sources.filter(file => file.endsWith('.py'))) {
    const source = await readFile(new URL(file, backend), 'utf8')
    assert.doesNotMatch(source, /^\s*(?:from|import)\s+[^\n]*(?:langgraph|langchain)/m, file)
  }
})

test('declares an installable Harness bundle', async () => {
  const manifest = JSON.parse(await readFile(new URL('package.json', packageRoot), 'utf8'))

  assert.equal(manifest.name, 'dsh-triton-riscv')
  assert.equal(manifest.private, true)
  assert.equal(manifest.dsh.bundle.patch, './cordis.patch.yml')
  assert.ok(manifest.files.includes('index.js'))
  assert.ok(manifest.files.includes('lib/client.js'))
  assert.ok(manifest.files.includes('lib/tool-policy.js'))
  assert.ok(manifest.files.includes('cordis.patch.yml'))
  assert.ok(manifest.files.includes('policy.md'))
  assert.ok(manifest.files.includes('README.md'))
  assert.ok(!manifest.files.includes('README.zh.md'))
  assert.equal(manifest.exports['./client'], './lib/client.js')
  assert.equal(manifest.dsh.client.platform, 'web')
  assert.equal(manifest.scripts.build, 'node scripts/build.mjs')

  const lockfile = await readFile(new URL('pnpm-lock.yaml', packageRoot), 'utf8')
  assert.match(lockfile, /lockfileVersion: '9\.0'/)
})

test('builds a bounded local workbench launch', () => {
  const launch = resolveWorkbenchLaunch({
    enabled: true,
    repoRoot: '/tmp/triton-riscv',
    python: '/tmp/venv/bin/python',
    port: 9000,
  })

  assert.deepEqual(launch, {
    command: '/tmp/venv/bin/python',
    args: ['-I', '-m', 'codex_agent.platform', '--host', '127.0.0.1', '--port', '9000'],
    cwd: '/tmp/triton-riscv',
    url: 'http://127.0.0.1:9000',
  })
  assert.equal(resolveWorkbenchLaunch({}), null)
  assert.throws(() => resolveWorkbenchLaunch({ enabled: true }), /repoRoot/)
  assert.throws(
    () =>
      resolveWorkbenchLaunch({
        enabled: true,
        repoRoot: '/tmp/repo',
        port: 70000,
      }),
    /between 1 and 65535/,
  )
})

test('mounts the domain policy and the official MCP client', async () => {
  const patch = await readFile(new URL('cordis.patch.yml', packageRoot), 'utf8')

  assert.match(patch, /id: triton-riscv-domain-policy/)
  assert.match(patch, /name: '@deepseek-ai\/dsh-mcp-client'/)
  assert.match(patch, /id: triton-riscv-native-host/)
  assert.match(patch, /enabled: false/)
  assert.match(patch, /disabled: true/)
  assert.doesNotMatch(patch, /process\.env|!!js/)
})

test('registers lifecycle guidance through the system prompt service', async () => {
  let effectName
  const sections = []
  let disposed = 0
  const ctx = {
    effect(setup, label) {
      effectName = label
      const dispose = setup()
      dispose()
    },
    systemPrompt: {
      section(value) {
        sections.push(value)
        return () => {
          disposed++
        }
      },
    },
  }

  const previousAutostart = process.env.TRITON_RISCV_WORKBENCH_AUTOSTART
  process.env.TRITON_RISCV_WORKBENCH_AUTOSTART = '0'
  try {
    apply(ctx)
  } finally {
    if (previousAutostart === undefined) delete process.env.TRITON_RISCV_WORKBENCH_AUTOSTART
    else process.env.TRITON_RISCV_WORKBENCH_AUTOSTART = previousAutostart
  }

  assert.equal(name, 'triton-riscv-domain-policy')
  assert.deepEqual(inject, ['systemPrompt'])
  assert.equal(effectName, 'tool:triton-riscv:verification')
  assert.deepEqual(sections, PROMPT_SECTIONS)
  const section = { text: sections.map(s => s.text).join('\n\n') }
  assert.equal(section.text, TRITON_RISCV_SYSTEM_PROMPT)
  assert.equal(section.text, (await readFile(new URL('policy.md', packageRoot), 'utf8')).trim())
  assert.equal(disposed, 5)
  assert.match(section.text, /prepare_operator_development/)
  assert.match(section.text, /propose_operator_implementation/)
  assert.match(section.text, /apply_development_proposal/)
  assert.match(section.text, /execute_approved_validation/)
  assert.match(section.text, /retrieve_operator_memory/)
  assert.match(section.text, /Never weaken or\s+replace an acceptance test/)
  assert.match(section.text, /untrusted historical evidence/)
  assert.match(section.text, /verified-passed/)
})
