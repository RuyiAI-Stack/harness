import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const result = spawnSync('npm', ['pack', '--ignore-scripts', '--dry-run', '--json'], {
  cwd: root,
  encoding: 'utf8',
  maxBuffer: 16 * 1024 * 1024,
})
if (result.status !== 0) throw new Error(result.stderr || 'npm package inspection failed')
const files = JSON.parse(result.stdout)[0].files.map(file => file.path)
const forbidden = files.filter(file =>
  /(^|\/)(node_modules|\.venv|__pycache__|\.state|\.runtime|\.env)(\/|$)|\.(sqlite3?|db|log|pyc|pem)$/.test(file),
)
if (forbidden.length)
  throw new Error(`Package includes runtime/private artifacts: ${forbidden.slice(0, 10).join(', ')}`)
const retired = files.filter(file =>
  /(^|\/)(langgraph_agent\/|test_langgraph_agent\.py$|langgraph-agent\.md$)/.test(file),
)
if (retired.length) throw new Error(`Package includes retired LangGraph files: ${retired.join(', ')}`)
for (const file of [
  'native.js',
  'lib/native-adapter.js',
  'lib/tool-policy.js',
  'lib/native-bridge.js',
  'lib/native-state.js',
  'lib/native-workspace.js',
  'lib/config.js',
  'lib/prompts.js',
  'prompts/0-hints.md',
  'prompts/1-skills.md',
  'prompts/2-failure-experience.md',
  'prompts/3-success-experience.md',
  'prompts/4-verification.md',
  'python/codex_agent/harness/native_bridge.py',
  'python/codex_agent/harness/mcp_server.py',
  'python/codex_agent/runtime_config.py',
  'python/codex_agent/execution_guard.py',
  'python/codex_agent/process_control.py',
  'python/codex_agent/operator_tools.py',
  'python/codex_agent/source_risk.py',
  'python/codex_agent/linux_sandbox.py',
  'python/codex_agent/remote_job.py',
  'python/codex_agent/remote_capacity.py',
  'python/codex_agent/remote_maintenance.py',
  'python/codex_agent/retention.py',
  'python/codex_agent/resource_probe.py',
  'python/codex_agent/remote_retry.py',
  'python/codex_agent/remote_control.py',
  'python/codex_agent/remote_process.py',
  'python/codex_agent/batch_recovery.py',
  'python/codex_agent/frontend/dist/index.html',
  'python/codex_agent/harness/bundle/policy.md',
  'python/codex_agent/harness/bundle/lib/prompts.js',
  'python/codex_agent/harness/bundle/lib/tool-policy.js',
  'python/codex_agent/harness/bundle/lib/native-workspace.js',
  'python/codex_agent/harness/bundle/prompts/4-verification.md',
]) {
  if (!files.includes(file)) throw new Error(`Package is missing ${file}`)
}
console.log(
  `Package audit passed: ${files.length} files; backend, UI and policy included; runtime/private artifacts excluded.`,
)
