import { readFile, readdir, stat } from 'node:fs/promises'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'
import { defineTool } from '@deepseek-ai/dsh-tools'

function isSafeModelName(value: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)
}

function isWithin(parent: string, child: string): boolean {
  const path = relative(parent, child)
  return path !== '' && !path.startsWith(`..${sep}`) && path !== '..' && !isAbsolute(path)
}

function resolveWithin(parent: string, value: string, label: string): string {
  const path = resolve(parent, value)
  if (!isWithin(parent, path)) throw new Error(`${label} must be inside ${parent}`)
  return path
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false
    throw error
  }
}

async function names(dir: string): Promise<string[]> {
  try {
    return (await readdir(dir, { withFileTypes: true }))
      .filter(entry => entry.isFile())
      .map(entry => entry.name)
      .sort()
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw error
  }
}

async function readText(path: string): Promise<string | undefined> {
  try {
    return await readFile(path, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}

/** Case- and punctuation-insensitive model key: "MiniMaxH3FL2VA" → "minimaxh3fl2va". */
function normalizeModelKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}

/** Items of the `foreach(model_flag IN ITEMS ...)` reset list. */
function modelResetListItems(content: string): string[] {
  const block = /foreach\(model_flag IN ITEMS([\s\S]*?)\)/.exec(content)?.[1]
  return block === undefined ? [] : block.split(/\s+/).filter(Boolean)
}

/** Quoted strings inside the bbdev `MODEL_CMAKE = { ... }` dict. */
function modelCmakeEntries(content: string): string[] {
  const block = /MODEL_CMAKE\s*=\s*\{([\s\S]*?)\}/.exec(content)?.[1]
  if (block === undefined) return []
  return [...block.matchAll(/"([^"]+)"/g)].map(match => match[1])
}

const HANDOFF_SECTIONS = ['Artifacts', 'Canonical Reference', 'Local Run', 'Build Binding', 'Known Limitation']

export function defineWorkloadAuditTool(repoPath: () => string) {
  return defineTool({
    name: 'buckyball_workload_audit',
    description:
      '对一个初始适配阶段的 ModelTest e2e workload 做只读终检：模型目录存在、' +
      'canonical 参考输出产物存在、三处注册项齐全、HANDOFF.md 含规定小节。' +
      '在本地 CPU（或超时后的 GPU）跑通并与参考对齐之后调用；交付要求 ACCEPT。' +
      '需要先 env_ensure_component({ name: "buckyball" })。',
    parameters: {
      model: {
        type: 'string',
        required: true,
        description: '新建的 models/models 目录名，例如 "MobileNetV2"',
      },
      referenceArtifact: {
        type: 'string',
        required: true,
        description: '固定用例的完整 canonical 输出产物路径，绝对路径或相对 bb-tests',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      if (!isSafeModelName(args.model)) throw new Error(`invalid model directory name: ${args.model}`)

      const root = repoPath()
      const bbTests = resolve(root, 'bb-tests')
      const referenceArtifact = resolveWithin(bbTests, args.referenceArtifact, 'referenceArtifact')

      const modelDir = join(root, 'bb-tests/workloads/src/ModelTest/e2e/models/models', args.model)
      const lines = [`# Buckyball workload audit: ${args.model}`]
      const failures: string[] = []
      const sourceNames = await names(modelDir)
      if (sourceNames.length === 0) {
        return `${lines.join('\n')}\n\nresult: REQUIRES FIX\n- missing workload directory: ${modelDir}`
      }

      lines.push('', '## Model directory')
      lines.push(`- ${modelDir}: ${sourceNames.length} files`)

      lines.push('', '## Canonical reference')
      const referenceName = relative(bbTests, referenceArtifact)
      const referencePresent = await exists(referenceArtifact)
      lines.push(`- ${referenceName}: ${referencePresent ? 'present' : 'missing'}`)
      if (!referencePresent) failures.push(`canonical full-output artifact is missing: ${referenceName}`)

      lines.push('', '## Registration')
      const modelKey = normalizeModelKey(args.model)
      const e2eModels = join(root, 'bb-tests/workloads/src/ModelTest/e2e/models')
      const registrations: { label: string; path: string; ok: boolean }[] = []

      const modelsSub = await readText(join(e2eModels, 'models/CMakeLists.txt'))
      registrations.push({
        label: `models/models/CMakeLists.txt add_subdirectory(${args.model})`,
        path: join(e2eModels, 'models/CMakeLists.txt'),
        ok: modelsSub?.includes(`add_subdirectory(${args.model})`) ?? false,
      })

      const modelsGate = await readText(join(e2eModels, 'CMakeLists.txt'))
      const resetItems = modelsGate === undefined ? [] : modelResetListItems(modelsGate)
      registrations.push({
        label: 'e2e/models/CMakeLists.txt MODEL reset list',
        path: join(e2eModels, 'CMakeLists.txt'),
        ok: resetItems.some(item => normalizeModelKey(item) === modelKey),
      })

      const bbdevStep = join(root, 'bbdev/api/steps/workload/01_build_event.step.py')
      const bbdevContent = await readText(bbdevStep)
      const cmakeEntries = bbdevContent === undefined ? [] : modelCmakeEntries(bbdevContent)
      registrations.push({
        label: 'bbdev MODEL_CMAKE dict',
        path: bbdevStep,
        ok: cmakeEntries.some(entry => normalizeModelKey(entry) === modelKey),
      })

      for (const registration of registrations) {
        lines.push(`- ${registration.label}: ${registration.ok ? 'present' : 'MISSING'}`)
        if (!registration.ok) failures.push(`registration missing: ${registration.label} (${registration.path})`)
      }

      lines.push('', '## HANDOFF.md')
      const handoffPath = join(modelDir, 'HANDOFF.md')
      const handoff = await readText(handoffPath)
      if (handoff === undefined) {
        lines.push('- missing')
        failures.push(`HANDOFF.md is missing: ${handoffPath}`)
      } else {
        lines.push('- present')
        const lower = handoff.toLowerCase()
        const missing = HANDOFF_SECTIONS.filter(section => !lower.includes(section.toLowerCase()))
        if (missing.length > 0) {
          lines.push(`- missing sections: ${missing.join(', ')}`)
          failures.push(`HANDOFF.md is missing sections: ${missing.join(', ')}`)
        } else {
          lines.push('- all five sections present')
        }
      }

      exec.signal.throwIfAborted()
      if (failures.length === 0) return `${lines.join('\n')}\n\nresult: ACCEPT`
      return `${lines.join('\n')}\n\nresult: REQUIRES FIX\n${failures.map(item => `- ${item}`).join('\n')}`
    },
  })
}
