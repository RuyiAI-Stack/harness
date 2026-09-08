import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { defineTool } from '@deepseek-ai/dsh-tools'

export function defineExistingModelsIndexTool(repoPath: () => string) {
  return defineTool({
    name: 'buckyball_existing_models_index',
    description:
      '列出本地已有 ModelTest e2e 模型及其粗特征，并给出新 workload 必须改动的' +
      '精确注册点。用于挑选同类最近邻参考，并正确完成新 workload 的注册。' +
      '需要先 env_ensure_component({ name: "buckyball" })。',
    parameters: {},
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(_args, exec) {
      const root = repoPath()
      const modelsDir = join(root, 'bb-tests/workloads/src/ModelTest/e2e/models/models')
      const models = (await readdir(modelsDir, { withFileTypes: true }))
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
        .sort()

      const out = [`repo: ${root}`, '', '## Existing ModelTest e2e models']
      for (const model of models) {
        const entries = await readdir(join(modelsDir, model), { withFileTypes: true })
        const traits = [
          entries.some(entry => entry.isFile() && entry.name.endsWith('.py')) ? 'python' : undefined,
          entries.some(entry => entry.isDirectory() && entry.name === 'images') ? 'images' : undefined,
          entries.some(entry => entry.isFile() && entry.name.endsWith('.mlir')) ? 'checked-in mlir' : undefined,
          entries.some(entry => entry.isDirectory() && entry.name === 'specs')
            ? 'rax/spec workflow (not a template)'
            : undefined,
        ].filter(Boolean)
        out.push(`- ${model}: ${traits.join(', ') || 'CMake/runtime assets'}`)
      }

      out.push('', '## Registration points for a new workload')
      out.push('- e2e/models/CMakeLists.txt: MODEL reset list (foreach(model_flag IN ITEMS ...))')
      out.push('- e2e/models/models/CMakeLists.txt: MODEL_<X>_DIR set + if(MODEL_<X>) add_subdirectory(<X>) branch')
      out.push('- bbdev/api/steps/workload/01_build_event.step.py: MODEL_CMAKE dict (CLI alias -> -DMODEL value)')
      out.push('archs/buckyball/<chip>/<X> layout registration is a later stage; do not add it now.')

      exec.signal.throwIfAborted()
      return out.join('\n')
    },
  })
}
