import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'

import {
  TRITON_RISCV_SYSTEM_PROMPT,
  apply,
  inject,
  name,
  resolveWorkbenchLaunch,
} from '../index.js'

const packageRoot = new URL('../', import.meta.url)

test('declares an installable Harness bundle', async () => {
  const manifest = JSON.parse(
    await readFile(new URL('package.json', packageRoot), 'utf8'),
  )

  assert.equal(manifest.name, 'dsh-triton-riscv')
  assert.equal(manifest.dsh.bundle.patch, './cordis.patch.yml')
  assert.ok(manifest.files.includes('index.js'))
  assert.ok(manifest.files.includes('lib/client.js'))
  assert.ok(manifest.files.includes('cordis.patch.yml'))
  assert.ok(manifest.files.includes('policy.md'))
  assert.equal(manifest.exports['./client'], './lib/client.js')
  assert.equal(manifest.dsh.client.platform, 'web')
})

test('builds a bounded local workbench launch', () => {
  const launch = resolveWorkbenchLaunch({
    TRITON_RISCV_CHECKOUT: '/tmp/triton-riscv',
    TRITON_RISCV_MCP_PYTHON: '/tmp/venv/bin/python',
    TRITON_RISCV_WORKBENCH_PORT: '9000',
  })

  assert.deepEqual(launch, {
    command: '/tmp/venv/bin/python',
    args: ['-m', 'codex_agent.platform', '--host', '127.0.0.1', '--port', '9000'],
    cwd: '/tmp/triton-riscv',
    url: 'http://127.0.0.1:9000',
  })
  assert.equal(resolveWorkbenchLaunch({ TRITON_RISCV_WORKBENCH_AUTOSTART: '0' }), null)
  assert.throws(() => resolveWorkbenchLaunch({}), /TRITON_RISCV_CHECKOUT/)
  assert.throws(
    () => resolveWorkbenchLaunch({ TRITON_RISCV_CHECKOUT: '/tmp/repo', TRITON_RISCV_WORKBENCH_PORT: '70000' }),
    /between 1 and 65535/,
  )
})

test('mounts the domain policy and the official MCP client', async () => {
  const patch = await readFile(
    new URL('cordis.patch.yml', packageRoot),
    'utf8',
  )

  assert.match(patch, /id: triton-riscv-domain-policy/)
  assert.match(patch, /name: '@deepseek-ai\/dsh-mcp-client'/)
  assert.match(patch, /serverName: triton_riscv/)
  assert.match(patch, /codex_agent\.harness\.mcp_server/)
  assert.match(patch, /TRITON_RISCV_REPO_ROOT:.*TRITON_RISCV_CHECKOUT/)
  assert.match(patch, /cwd:.*TRITON_RISCV_CHECKOUT/)
  assert.match(patch, /TRITON_RISCV_ALLOW_VALIDATION.*'0'/)
  assert.match(patch, /TRITON_RISCV_REQUIRE_APPROVED_VALIDATION.*'1'/)
  assert.match(patch, /failOnStartupError: true/)
})

test('registers lifecycle guidance through the system prompt service', async () => {
  let effectName
  let section
  let disposed = false
  const ctx = {
    effect(setup, label) {
      effectName = label
      const dispose = setup()
      dispose()
    },
    systemPrompt: {
      section(value) {
        section = value
        return () => {
          disposed = true
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
  assert.equal(effectName, 'triton-riscv.system-prompt')
  assert.equal(section.name, 'tool:triton-riscv')
  assert.equal(section.order, 180)
  assert.equal(section.text, TRITON_RISCV_SYSTEM_PROMPT)
  assert.equal(
    section.text,
    (await readFile(new URL('policy.md', packageRoot), 'utf8')).trim(),
  )
  assert.equal(disposed, true)
  assert.match(section.text, /prepare_operator_development/)
  assert.match(section.text, /Never weaken or replace an acceptance test/)
  assert.match(section.text, /verified-passed/)
})

test('publishes a root-slot Harness client plugin', async () => {
  const source = await readFile(new URL('../lib/client.js', import.meta.url), 'utf8')
  let definition
  const sandbox = {
    URL,
    URLSearchParams,
    window: {
      __ModuleLoader__: {
        load(value) {
          definition = value
        },
      },
    },
  }
  vm.runInNewContext(source, sandbox)

  assert.equal(definition.id, 'dsh-triton-riscv')
  const client = definition.factory(specifier => {
    assert.equal(specifier, 'react')
    return { createElement: () => null }
  })
  let registration
  const ctx = {
    effect(setup) {
      setup()
    },
    slots: {
      register(options, component) {
        registration = { options, component }
        return () => {}
      },
    },
  }
  client.apply(ctx)

  assert.deepEqual(Array.from(client.inject), ['slots'])
  assert.equal(registration.options.name, 'root')
  assert.equal(registration.options.priority, -100)
  assert.equal(typeof registration.component, 'function')
})
