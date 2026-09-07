import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

import {
  TRITON_RISCV_SYSTEM_PROMPT,
  apply,
  inject,
  name,
} from '../index.js'

const packageRoot = new URL('../', import.meta.url)

test('declares an installable Harness bundle', async () => {
  const manifest = JSON.parse(
    await readFile(new URL('package.json', packageRoot), 'utf8'),
  )

  assert.equal(manifest.name, 'dsh-triton-riscv')
  assert.equal(manifest.dsh.bundle.patch, './cordis.patch.yml')
  assert.ok(manifest.files.includes('index.js'))
  assert.ok(manifest.files.includes('cordis.patch.yml'))
  assert.ok(manifest.files.includes('policy.md'))
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

  apply(ctx)

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
