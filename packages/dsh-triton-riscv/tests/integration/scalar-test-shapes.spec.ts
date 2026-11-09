import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const python =
  process.env.TRITON_PLUGIN_TEST_PYTHON || fileURLToPath(new URL('../../.venv/bin/python', import.meta.url))
function shapes(body: string, values = '[1, 17, 1023, 1024]') {
  const source = `import pytest\nimport torch\n@pytest.mark.parametrize("size", ${values})\ndef test_example(size):\n${body}`
  const result = spawnSync(
    python,
    [
      '-I',
      '-B',
      '-c',
      'import ast,json,sys\nfrom codex_agent.develop_operator import collect_literal_shapes\nprint(json.dumps(sorted(collect_literal_shapes(ast.parse(sys.stdin.read())))))',
    ],
    { input: source, encoding: 'utf8', timeout: 5000 },
  )
  expect(result.status, result.stderr).toBe(0)
  return JSON.parse(result.stdout)
}

it('recognizes scalar pytest sizes actually used by 1D tensor factories', () => {
  expect(shapes('    x = torch.randn(size, device="cpu")\n')).toEqual(
    expect.arrayContaining([[1], [17], [1023], [1024]]),
  )
})
it('recognizes a literal vector repeated to sufficient capacity then sliced to the parameter', () => {
  const actual = shapes(
    '    values = torch.tensor([-2.0, 0.0, 3.0])\n' +
      '    x = values.repeat((size + values.numel() - 1) // values.numel())[:size]\n',
  )
  expect(actual).toEqual(expect.arrayContaining([[1], [17], [1023], [1024]]))
})
it.each([
  '    x = torch.randn(4)\n',
  '    x = torch.randn(size, size)\n',
  '    values = torch.tensor([-2.0, 0.0, 3.0])\n    x = values.repeat(1)[:size]\n',
  '    values = torch.tensor([[-2.0, 0.0, 3.0]])\n    x = values.repeat(size)[:size]\n',
  '    values = torch.tensor([-2.0, 0.0, 3.0])\n    x = values.repeat(__import__("os").system("never execute"))[:size]\n',
])('does not infer complete 1D coverage from unused sizes or unsupported construction: %s', body => {
  expect(shapes(body)).not.toContainEqual([1024])
})
it('does not treat boolean pytest parameters as integer dimensions', () => {
  expect(shapes('    x = torch.randn(size)\n', '[True, False]')).not.toContainEqual([1])
})
