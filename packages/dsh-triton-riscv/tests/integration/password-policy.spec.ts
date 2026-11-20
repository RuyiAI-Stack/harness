import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

it('keeps six-character registration and password-change rules consistent without weakening login checks', () => {
  const driver = `
import json
from pydantic import ValidationError
from codex_agent.accounts.store import password_ok, AccountError
from codex_agent.accounts.api import Registration, PasswordChange, Credentials
rows = []
for password in ["12345", "123456", "abcdef", "密" * 6, "a" * 128, "a" * 129]:
    row = []
    for check in [lambda: password_ok(password),
                  lambda: Registration(username="test_user", password=password, invitation="test"),
                  lambda: PasswordChange(old_password="old", new_password=password)]:
        try:
            check()
            row.append(True)
        except (AccountError, ValidationError):
            row.append(False)
    rows.append(row)
print(json.dumps({"rows": rows, "login": Credentials(username="test", password="x").password}))
`
  const result = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url)),
    ['-I', '-B', '-c', driver],
    { encoding: 'utf8', timeout: 10000 },
  )
  expect(result.status, result.stderr).toBe(0)
  expect(JSON.parse(result.stdout)).toEqual({
    rows: [
      [false, false, false],
      [true, true, true],
      [true, true, true],
      [true, true, true],
      [true, true, true],
      [false, false, false],
    ],
    login: 'x',
  })
})
