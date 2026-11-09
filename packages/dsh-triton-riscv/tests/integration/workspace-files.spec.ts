import { beforeAll, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

let result: any
beforeAll(() => {
  const driver = String.raw`
import asyncio, json, os, tempfile, threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from codex_agent.platform.workspace_files import router, MAX_FILE_BYTES
from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.store import AccountError

with tempfile.TemporaryDirectory() as folder:
    base = Path(folder)
    a, b = base / 'alice', base / 'bob'
    relative = 'python/examples/flaggems/demo.py'
    def write(root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path
    source = write(a, relative, b'def demo():\n    return 1\n')
    write(b, relative, b'def demo():\n    return 2\n')
    receipt = 'agent-results/operator-lifecycle/receipts/run-demo.json'
    log = 'agent-results/operator-lifecycle/validation/logs/demo.log'
    write(a, receipt, b'{"status":"failed","exit_code":1}')
    write(a, log, b'60 failed\n')
    write(a, 'tasks/operators/demo.md', b'# Demo\n')
    secret = write(base, 'secret.txt', b'never expose')
    write(a, '.env', b'never expose')
    write(a, 'python/examples/flaggems/binary.py', b'\x00binary')
    (a / 'python/examples/flaggems/link.py').symlink_to(secret)
    (a / 'python/examples/flaggems/linked').symlink_to(b / 'python/examples/flaggems', target_is_directory=True)
    os.link(secret, a / 'python/examples/flaggems/hard.py')
    os.mkfifo(a / 'python/examples/flaggems/pipe.py')
    huge = write(a, 'agent-results/operator-lifecycle/validation/logs/huge.log', b'')
    with huge.open('wb') as f: f.truncate(MAX_FILE_BYTES + 1)
    apps = {}
    for name, root in [('alice', a), ('bob', b)]:
        app = FastAPI()
        app.include_router(router(root))
        apps[name] = app
    out = {}
    with TestClient(apps['alice']) as client:
        def get(path): return client.get('/api/workspace/files', params={'path': str(path)})
        r = get(relative)
        out['source'] = [r.status_code, r.text, r.headers['content-type'], r.headers['cache-control'], r.headers['content-security-policy']]
        out['absolute'] = get(source).status_code
        out['evidence'] = [get(receipt).json(), get(log).text, get('tasks/operators/demo.md').text]
        out['denied'] = [get(p).status_code for p in [
            '../secret.txt', '.env', b / relative, '/etc/passwd',
            'python/examples/flaggems/../flaggems/demo.py',
            'python/examples/flaggems/%2e%2e/demo.py',
            'python/examples/flaggems/link.py', 'python/examples/flaggems/linked/demo.py',
            'python/examples/flaggems/hard.py', 'python/examples/flaggems/pipe.py',
            'python/examples/flaggems/binary.py', 'python/examples/flaggems/missing.py',
            'agent-results/accounts.json', 'python/examples/flaggems/demo.py\x00',
        ]]
        out['too_large'] = get(huge).status_code
        out['write_denied'] = client.post('/api/workspace/files', params={'path':relative}).status_code
    class Accounts:
        def health(self): pass
        def close(self): pass
        def authenticate(self, token, **kw):
            if token not in apps: raise AccountError(401, 'Please sign in')
            return {'id': token, 'csrf_token': 'fixture'}
        def limit(self, *args): pass
    class Workbenches:
        def start(self): pass
        def close(self): pass
        async def monitor(self):
            await asyncio.sleep(3600)
        def get(self, uid): return {'socket':uid, 'key':'fixture', 'slots':threading.BoundedSemaphore(4)}
    config = SimpleNamespace(origin='http://testserver', cookie_name='session', secure_cookie=False)
    gateway = create_gateway(config, accounts=Accounts(), workbenches=Workbenches())
    original_client = httpx.AsyncClient
    def private_client(*args, **kw):
        # The gateway still performs authentication and per-account dispatch.
        uid = str(kw['transport']._pool._uds)
        return original_client(transport=httpx.ASGITransport(app=apps[uid]), base_url='http://private')
    with patch('codex_agent.accounts.api.httpx.AsyncClient', side_effect=private_client):
        with TestClient(gateway) as client:
            def linked(token=None, path=relative, account=None):
                params = {'path':str(path)}
                if account: params['account_id'] = account
                return client.get('/api/workspace/files', params=params, headers={'Cookie':'session='+token} if token else {})
            out['anonymous'] = linked().status_code
            own = linked('alice', account='alice')
            out['owner'] = [own.status_code, own.text, own.headers['content-security-policy']]
            out['bob'] = linked('bob', account='bob').text
            out['cross_account'] = linked('bob', path=source).status_code
            out['stale_tab'] = linked('bob', account='alice').status_code
            out['foreign_origin'] = client.get('/api/workspace/files', params={'path':relative}, headers={'Cookie':'session=alice','Sec-Fetch-Site':'cross-site'}).status_code
    print(json.dumps(out))
`
  const output = spawnSync(
    process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url)),
    ['-I', '-B', '-c', driver],
    { encoding: 'utf8', timeout: 15000 },
  )
  expect(output.status, output.stderr + output.stdout).toBe(0)
  result = JSON.parse(output.stdout)
})

it('returns real source and evidence as read-only plain text, never executable HTML', () => {
  expect(result.source[0]).toBe(200)
  expect(result.source[1]).toContain('return 1')
  expect(result.source[2]).toContain('text/plain')
  expect(result.source[3]).toBe('no-store')
  expect(result.source[4]).toContain("default-src 'none'; sandbox")
  expect(result.absolute).toBe(200)
  expect(result.evidence).toEqual([{ status: 'failed', exit_code: 1 }, '60 failed\n', '# Demo\n'])
  expect(result.write_denied).toBe(405)
})

it('rejects secrets, traversal, foreign workspaces, symlinks, hardlinks, FIFOs and binary files', () => {
  expect(result.denied).toHaveLength(14)
  expect(result.denied.every((status: number) => status === 404)).toBe(true)
  expect(result.too_large).toBe(413)
})

it('requires login, binds links to the displayed account and preserves sandbox headers through the gateway', () => {
  expect(result.anonymous).toBe(401)
  expect(result.owner[0]).toBe(200)
  expect(result.owner[1]).toContain('return 1')
  expect(result.owner[2]).toContain('sandbox')
  expect(result.bob).toContain('return 2')
  expect(result.cross_account).toBe(404)
  expect(result.stale_tab).toBe(401)
  expect(result.foreign_origin).toBe(403)
})
