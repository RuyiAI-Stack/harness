import type { ReactNode } from 'react'
import type { ModelRoute } from './types'
import { accountStreamUrl } from './auth'

export function shortDate(value?: string): string {
  return value ? value.replace('T', ' ').slice(5, 16) : '刚刚'
}

export function ModelRouteSummary({ route }: { route: Partial<ModelRoute> }) {
  const tiers: Record<ModelRoute['tier'], string> = { fast: '快速', strong: '强力', default: '默认' }
  const reasons: Record<string, string> = {
    routing_disabled: '自动路由未启用，沿用当前模型',
    unmanaged_session_keeps_model: '旧式连续会话保留原模型和历史',
    bounded_information_task: '有明确范围的资料整理或字段提取',
    code_or_review_role: '代码实现、审查或主任务规划',
    code_execution_or_complex_task: '涉及代码、执行或复杂判断',
    unknown_task: '任务不明确，保守选择强模型',
    input_exceeds_fast_budget: '输入超过快速模型路由预算',
    fast_model_not_configured: '未配置快速模型，使用强模型',
  }
  return (
    <>
      模型：{route.model || '未提供'}（{route.tier ? tiers[route.tier] || '未知档位' : '未提供档位'}）
      {route.provider && ` · 提供方：${route.provider}`} · 超时：
      {route.timeout_seconds === undefined ? '未提供' : `${route.timeout_seconds} 秒`}
      <br />
      选择原因：{route.reason ? reasons[route.reason] || route.reason : '未提供'}
    </>
  )
}

function fileHref(value: string): string | undefined {
  const path = value.replace(/\\([_*])/g, '$1')
  if (!/^(?:\.?\/?python\/examples\/flaggems\/|\.?\/?tasks\/operators\/|\.?\/?agent-results\/|\/)/.test(path))
    return undefined
  if (path.startsWith('//') || !/\.(?:py|md|json|log|txt|diff)$/.test(path)) return undefined
  return accountStreamUrl(`/api/workspace/files?path=${encodeURIComponent(path)}`)
}

function linkHref(value: string): string | undefined {
  const file = fileHref(value)
  if (file) return file
  try {
    const url = new URL(value)
    if (['https:', 'http:'].includes(url.protocol) && !url.username && !url.password) return url.href
  } catch {
    // Unsupported or incomplete links remain ordinary text while streaming.
  }
  return undefined
}

function inlineMarkup(value: string): ReactNode[] {
  const parts = value.split(
    /(\[[^\]\n]+\]\((?:<[^>\n]+>|[^\s)]+)\)|`[^`]+`|\*\*[^*]+\*\*|(?:\.?\/?python\/examples\/flaggems\/|\.?\/?tasks\/operators\/|\.?\/?agent-results\/|\/)[^\s`<>"'，。；：、()\[\]]+\.(?:py|md|json|log|txt|diff)(?![\w.]))/g,
  )
  return parts.filter(Boolean).map((part, index) => {
    const markdown = part.match(/^\[([^\]]+)\]\((.+)\)$/)
    if (markdown) {
      const href = linkHref(markdown[2].replace(/^<|>$/g, ''))
      const label = markdown[1].replace(/^`|`$/g, '')
      return href ? (
        <a key={index} href={href} target="_blank" rel="noopener noreferrer">
          {label}
        </a>
      ) : (
        label
      )
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      const text = part.slice(1, -1)
      const href = fileHref(text)
      return href ? (
        <a key={index} href={href} target="_blank" rel="noopener noreferrer">
          <code>{text}</code>
        </a>
      ) : (
        <code key={index}>{text}</code>
      )
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{inlineMarkup(part.slice(2, -2))}</strong>
    }
    const href = fileHref(part)
    if (href)
      return (
        <a key={index} href={href} target="_blank" rel="noopener noreferrer">
          {part}
        </a>
      )
    return part
  })
}

export function MessageText({ value }: { value: string }) {
  const blocks: ReactNode[] = []
  let list: string[] = []
  let code: string[] | null = null

  const flushList = () => {
    if (!list.length) return
    blocks.push(
      <ul key={`list-${blocks.length}`}>
        {list.map((item, index) => (
          <li key={index}>{inlineMarkup(item)}</li>
        ))}
      </ul>,
    )
    list = []
  }

  value.split('\n').forEach(line => {
    if (line.startsWith('```')) {
      flushList()
      if (code !== null) {
        blocks.push(
          <pre key={`code-${blocks.length}`}>
            <code>{code.join('\n')}</code>
          </pre>,
        )
        code = null
      } else code = []
      return
    }
    if (code !== null) {
      code.push(line)
      return
    }
    if (line.startsWith('- ')) {
      list.push(line.slice(2))
      return
    }
    flushList()
    if (!line.trim()) return
    if (line.startsWith('### ')) {
      blocks.push(<h3 key={`heading-${blocks.length}`}>{inlineMarkup(line.slice(4))}</h3>)
    } else {
      blocks.push(<p key={`paragraph-${blocks.length}`}>{inlineMarkup(line)}</p>)
    }
  })
  flushList()
  if (code !== null)
    blocks.push(
      <pre key="streaming-code">
        <code>{(code as string[]).join('\n')}</code>
      </pre>,
    )
  return <>{blocks}</>
}
