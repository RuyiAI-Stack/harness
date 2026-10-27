export interface AccountUser {
  id: string
  username: string
  csrf_token: string
}
let currentUser: AccountUser | null = null
let accountRevision = 0
export function setAccount(user: AccountUser | null) {
  if (user?.id !== currentUser?.id || user?.csrf_token !== currentUser?.csrf_token) accountRevision += 1
  currentUser = user
}
export function accountName() {
  return currentUser?.username
}

export function accountStreamUrl(path: string) {
  if (!currentUser) return path
  const url = new URL(path, window.location.origin)
  // EventSource cannot set custom headers. The account ID is not a credential.
  url.searchParams.set('account_id', currentUser.id)
  return url.pathname + url.search
}

export async function authFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const revision = accountRevision
  const assertCurrent = () => {
    if (revision !== accountRevision) throw new DOMException('Account changed; response discarded', 'AbortError')
  }
  const headers = new Headers(options.headers)
  if (currentUser && path !== '/api/auth/me') headers.set('X-Triton-Account', currentUser.id)
  if (!['GET', 'HEAD', 'OPTIONS'].includes((options.method || 'GET').toUpperCase()) && currentUser) {
    headers.set('X-CSRF-Token', currentUser.csrf_token)
  }
  const response = await fetch(path, { ...options, headers, credentials: 'same-origin' })
  assertCurrent()
  // Also check after body decoding: headers may arrive before an account switch.
  const json = response.json.bind(response)
  response.json = async () => {
    const body = await json()
    assertCurrent()
    return body
  }
  if (response.status === 401 && !['/api/auth/login', '/api/auth/me'].includes(path)) {
    window.dispatchEvent(new Event('triton:signout'))
  }
  return response
}

export async function authRequest<T>(path: string, data?: unknown): Promise<T> {
  const response = await authFetch(
    '/api/auth/' + path,
    data === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        },
  )
  const body = await response.json()
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '账号请求失败')
  return body as T
}
