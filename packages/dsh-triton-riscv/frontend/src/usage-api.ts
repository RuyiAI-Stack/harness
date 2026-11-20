import { authFetch } from './auth'

export interface AccountUsage {
  account: { id: string; username: string; status: string }
  totals: { sessions: number; runs: number; failed_runs: number }
  last_used_at: string | null
  recent_runs: { id: string; status: string; created_at: string; updated_at: string }[]
  as_of: string
}

export async function readAccountUsage(accountId: string, signal: AbortSignal): Promise<AccountUsage> {
  const controller = new AbortController()
  const abort = () => controller.abort()
  signal.addEventListener('abort', abort, { once: true })
  if (signal.aborted) controller.abort()
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error('使用情况读取超时，请重试。'))
      controller.abort()
    }, 15000)
  })
  try {
    return await Promise.race([
      timeout,
      (async () => {
        const response = await authFetch('/api/account/usage', { signal: controller.signal })
        if (!response.ok) {
          if (response.status === 401) throw new Error('登录已失效，请重新登录。')
          if (response.status === 403) throw new Error('无法读取当前账号的使用情况，请重新登录后重试。')
          if (response.status === 429) throw new Error('刷新过于频繁，请稍后重试。')
          throw new Error('使用情况暂时无法读取，请重试。')
        }
        const body = (await response.json()) as AccountUsage
        if (body.account?.id !== accountId) throw new Error('账号已变化，请重新打开使用情况。')
        return body
      })(),
    ])
  } finally {
    clearTimeout(timer)
    signal.removeEventListener('abort', abort)
  }
}

export function accountStatus(status: string): string {
  return (
    ({ active: '正常', disabled: '已停用', suspended: '已暂停', locked: '已锁定' } as Record<string, string>)[status] ||
    '状态待确认'
  )
}

export function runStatus(status: string): string {
  return (
    (
      {
        queued: '等待执行',
        running: '执行中',
        completed: '已完成',
        succeeded: '已完成',
        failed: '失败',
        cancelled: '已取消',
        'awaiting-confirmation': '待确认',
        'needs-reconciliation': '待核对',
        paused: '已暂停',
        dead: '已停止',
        interrupted: '已中断',
      } as Record<string, string>
    )[status] || '状态待确认'
  )
}
