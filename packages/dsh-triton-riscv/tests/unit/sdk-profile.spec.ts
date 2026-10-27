import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const python = process.env.TRITON_PLUGIN_TEST_PYTHON || join(root, '.venv/bin/python')
const driver = `
import json, sys, tempfile
from pathlib import Path
from unittest.mock import patch
from codex_agent.harness.sdk_profile import kernel, stage_composition, write_profile
from codex_agent.harness.runtime import DeepSeekHarnessBackend
from codex_agent.harness.config import HarnessSettings
data = json.load(sys.stdin)
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    home = root / "home"
    source = root / "source.yml"
    source.write_text(data.get("source", "[]"))
    profile = write_profile(home, kernel())
    try:
        if data.get("fail_constructor"):
            homes = []
            def fail(**kwargs):
                homes.append(Path(kwargs["dsh_home"]))
                raise RuntimeError("fixture startup failure")
            with patch("deepseek_harness.DeepSeekHarness", fail):
                backend = DeepSeekHarnessBackend(HarnessSettings.from_env(root, {"ISRC_API_KEY":"fixture"}))
                try:
                    backend._runtime()
                except RuntimeError:
                    pass
            result = dict(removed=bool(homes) and all(not p.exists() for p in homes), runtime=backend._harness)
        else:
            stage_composition(source, home)
            result = dict(text=(home / "profiles/triton/domain.yml").read_text(), source_unchanged=source.read_text()==data.get("source", "[]"),
                manifest=json.loads((home / "profiles/triton/package.json").read_text()), root=str(root))
    except Exception as error:
        result = dict(error=type(error).__name__)
    print(json.dumps(result))
`
function invoke(data: object) {
  const child = spawnSync(
    python,
    [
      '-I',
      '-B',
      '-c',
      'import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))',
      join(root, 'python'),
      driver,
    ],
    {
      input: JSON.stringify(data),
      encoding: 'utf8',
      timeout: 15000,
      cwd: root,
      env: Object.fromEntries(
        Object.entries(process.env).filter(
          ([key]) => !/^(TRITON_|RISCV_|DSH_)/.test(key) && !/(API_KEY|TOKEN|PASSWORD|SECRET)$/.test(key),
        ),
      ),
    },
  )
  expect(child.error).toBeUndefined()
  expect(child.status, child.stderr).toBe(0)
  return JSON.parse(child.stdout)
}

describe('isolated SDK configuration', () => {
  it('retains trusted JS expressions as data, without evaluating them in Python', () => {
    const result = invoke({
      source: '- id: sample\n  name: "@deepseek-ai/sample"\n  config:\n    value: !!js process.env.FIXTURE_KEY\n',
    })
    expect(result.text).toContain('!!js')
    expect(result.text).toContain('process.env.FIXTURE_KEY')
    expect(result.source_unchanged).toBe(true)
    expect(result.manifest.dsh.profile).toEqual({ bundles: [], patchReload: 'startup' })
  })
  it('anchors relative external plugin paths to the source directory, including groups', () => {
    const result = invoke({
      source:
        '- id: external\n  name: ./plugin.js\n- id: nested\n  group:\n    - id: child\n      name: ../sibling.js\n',
    })
    expect(result.text).toContain(join(result.root, 'plugin.js'))
    expect(result.text).toContain(resolve(result.root, '../sibling.js'))
    expect(result.source_unchanged).toBe(true)
  })
  it.each(['not: a-list', '- 42', '- !!python/object/apply:os.system [false]'])(
    'rejects malformed or Python-executable YAML: %s',
    source => {
      expect(invoke({ source }).error).toBeTruthy()
    },
  )
  it('cleans a generated home if the SDK constructor fails', () => {
    expect(invoke({ fail_constructor: true })).toEqual({ removed: true, runtime: null })
  })
})
