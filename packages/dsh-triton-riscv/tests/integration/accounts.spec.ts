import { beforeAll, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
let result: Record<string, unknown>
beforeAll(() => {
  for (const key of ['TRITON_MYSQL_URL', 'TRITON_AUTH_REDIS_URL', 'TRITON_TEST_AMQP_URL'])
    expect(process.env[key], `${key} must target disposable services`).toBeTruthy()
  const env = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) =>
        !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key) && !/^(DSH_|RISCV_|TRITON_RISCV_|TRITON_WORKBENCH_)/.test(key),
    ),
  )
  const output = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python'),
    ['-I', '-B', join(root, 'tests/integration/accounts-fixture.py')],
    {
      encoding: 'utf8',
      env: { ...env, TRITON_RISCV_CONFIG: '{"schemaVersion":1}' },
      timeout: 110000,
      maxBuffer: 8 * 1024 * 1024,
    },
  )
  expect(output.error, output.stderr).toBeUndefined()
  expect(output.status, output.stdout + output.stderr).toBe(0)
  result = JSON.parse(output.stdout)
}, 120000)

it('requires login and rejects host/origin/CSRF/identity spoofing', () => {
  expect(result.anonymous).toBe(401)
  expect(result.bad_host).toBe(400)
  expect(result.bad_origin).toBe(403)
  expect(result.missing_csrf).toBe(403)
  expect(result.spoof_registration).toBe(422)
})
it('registers with a single-use invitation and hashes passwords with Argon2id', () => {
  expect(result.registration).toBe(201)
  expect(result.invite_replay).toBe(400)
  expect(result.argon2id).toBe(true)
  expect(result.cookie_flags).toEqual({ httponly: true, strict: true })
  expect(result.invalid_credentials).toEqual([401, 401])
})
it('rejects stale tab reads, writes, streams and logout without affecting the new account', () => {
  expect(result.stale_tab).toEqual([401, 401, 401, 401])
  expect(result.stale_tab_no_effects).toBe(true)
  expect(result.bound_owner_read).toBe(200)
})
it('isolates conversation reads, mutations, approvals and event streams', () => {
  expect(result.owner_read).toBe(200)
  expect(result.owner_sse).toBe(200)
  expect(result.bob_sessions).toEqual([])
  expect(result.cross_reads).toEqual([404, 404, 404, 404, 404, 404, 404])
  expect(result.cross_delete).toBe(404)
  expect(result.cross_pin).toBe(404)
})
it('isolates task claims and idempotency domains', () => {
  expect(result.distinct_jobs).toBe(true)
  expect(result.foreign_envelope).toBe(true)
  expect(result.foreign_job).toBe(true)
  expect(result.owner_artifact).toBe('private-artifact-a')
  expect(result.foreign_artifact_http).toBe(404)
})
it('isolates context, memories, cache keys, artifacts and source files', () => {
  expect(result.checkpoint_b).toBeNull()
  expect(result.memory_a).toBe(1)
  expect(result.memory_b).toBe(0)
  for (const key of ['cache_isolated', 'artifact_not_in_b', 'files_isolated', 'template_unchanged'])
    expect(result[key], key).toBe(true)
})
it('prevents bypass through a private backend, legacy entry or model tools', () => {
  expect(result.private_bypass).toBe(401)
  for (const key of ['marker_blocks_legacy', 'wrong_workspace', 'reviewer_bound', 'restricted_tools'])
    expect(result[key], key).toBe(true)
})
it('revokes cookies on logout, expiry, password change and disable', () => {
  expect(result.logout).toBe(200)
  expect(result.logout_replay).toBe(401)
  for (const key of [
    'password_revokes_all',
    'old_password_rejected',
    'expired',
    'session_cap',
    'disabled_session',
    'disabled_login',
    'disabled_worker',
  ])
    expect(result[key], key).toBe(true)
})
it('fails closed on Redis loss and rejects evicting auth storage', () => {
  expect(result.redis_outage).toBe(503)
  expect(result.lfu_rejected).toBe(true)
})
it('limits requests, does not renew from polling and requires public HTTPS', () => {
  for (const key of ['rate_limited', 'poll_no_renewal', 'https_required', 'secure_cookie'])
    expect(result[key], key).toBe(true)
})
it('uses a single invitation only once under concurrent registrations', () => {
  expect(result.concurrent_invite).toEqual([201, 400, 400, 400])
})
it('stops an already-open stream after its cookie is revoked', () => {
  expect(result.revoked_stream).toBe('data: before-logout\n\n')
})
it('starts private queue workers and stops them when their account is disabled', () => {
  expect(result.account_workers).toBe(true)
  expect(result.disable_stops_workers).toBe(true)
})
