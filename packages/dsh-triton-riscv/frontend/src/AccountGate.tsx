import { KeyRound, LogOut, UserRound, X } from 'lucide-react'
import { FormEvent, ReactNode, useEffect, useRef, useState } from 'react'
import { AccountUser, authFetch, authRequest, setAccount } from './auth'
import './account.css'

export function AccountGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<'loading' | 'login' | 'ready' | 'error'>('loading')
  const [user, setUser] = useState<AccountUser | null>(null)
  const [register, setRegister] = useState(false)
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)
  const [passwordDialog, setPasswordDialog] = useState(false)
  const [notice, setNotice] = useState('')
  const generation = useRef(0)
  const authPending = useRef(false)
  const accountChannel = useRef<BroadcastChannel | null>(null)

  function signedOut() {
    generation.current += 1
    authPending.current = false
    setPending(false)
    setAccount(null)
    setUser(null)
    setPasswordDialog(false)
    setState('login')
  }
  async function check() {
    if (authPending.current) return
    const version = ++generation.current
    try {
      const response = await authFetch('/api/auth/me')
      if (version !== generation.current) return
      if (response.status === 401) {
        signedOut()
        return
      }
      if (!response.ok) throw new Error('账号服务暂不可用，请稍后重试')
      const result = (await response.json()) as { enabled: boolean; user: AccountUser | null }
      if (version !== generation.current) return
      if (typeof result.enabled !== 'boolean' || (result.enabled && (!result.user?.id || !result.user?.csrf_token)))
        throw new Error('无法确认登录状态')
      setAccount(result.user)
      setUser(result.user)
      setState('ready')
      setError('')
    } catch (reason) {
      if (version !== generation.current) return
      setAccount(null)
      setUser(null)
      setError(reason instanceof Error ? reason.message : '无法连接账号服务')
      setState('error')
    }
  }
  useEffect(() => {
    void check()
    const channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel('triton-account')
    accountChannel.current = channel
    if (channel)
      channel.onmessage = () => {
        signedOut()
        void check()
      }
    const expire = () => signedOut()
    const restore = () => void check()
    window.addEventListener('triton:signout', expire)
    window.addEventListener('pageshow', restore)
    const timer = window.setInterval(restore, 60000)
    return () => {
      generation.current += 1
      clearInterval(timer)
      channel?.close()
      accountChannel.current = null
      window.removeEventListener('triton:signout', expire)
      window.removeEventListener('pageshow', restore)
    }
  }, [])
  function notifyTabs() {
    // Reuse the receiving channel so the sender does not invalidate its own login.
    accountChannel.current?.postMessage('changed')
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    const version = ++generation.current
    authPending.current = true
    setPending(true)
    setError('')
    setNotice('')
    try {
      if (register) {
        await authRequest('register', {
          username: data.get('username'),
          password: data.get('password'),
          invitation: data.get('invitation'),
        })
        if (version !== generation.current) return
        setRegister(false)
        setNotice('注册成功，请登录')
      } else {
        const result = await authRequest<{ user: AccountUser }>('login', {
          username: data.get('username'),
          password: data.get('password'),
        })
        if (version !== generation.current) return
        setAccount(result.user)
        setUser(result.user)
        setState('ready')
        notifyTabs()
      }
    } catch (reason) {
      if (version === generation.current) setError(reason instanceof Error ? reason.message : '请求失败')
    } finally {
      if (version === generation.current) {
        authPending.current = false
        setPending(false)
      }
    }
  }
  async function logout() {
    const version = ++generation.current
    authPending.current = true
    setPending(true)
    setError('')
    try {
      await authRequest('logout', {})
      if (version !== generation.current) return
      signedOut()
      notifyTabs()
    } catch (reason) {
      if (version === generation.current) setError(reason instanceof Error ? reason.message : '退出失败')
    } finally {
      if (version === generation.current) {
        authPending.current = false
        setPending(false)
      }
    }
  }
  async function changePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    if (data.get('new') !== data.get('confirm')) {
      setError('两次输入的新密码不一致')
      return
    }
    const version = ++generation.current
    authPending.current = true
    setPending(true)
    setError('')
    try {
      await authRequest('password', { old_password: data.get('old'), new_password: data.get('new') })
      if (version !== generation.current) return
      signedOut()
      notifyTabs()
      setNotice('密码已更新，请重新登录')
    } catch (reason) {
      if (version === generation.current) setError(reason instanceof Error ? reason.message : '修改失败')
    } finally {
      if (version === generation.current) {
        authPending.current = false
        setPending(false)
      }
    }
  }
  if (state !== 'ready')
    return (
      <main className="account-page">
        <section className="account-form" aria-label="账号登录">
          <div className="account-brand">
            <span className="brand-mark">RV</span>
            <span>Triton-RISCV Agent</span>
          </div>
          <h1>
            {state === 'loading' ? '连接中' : state === 'error' ? '连接暂不可用' : register ? '创建账号' : '登录工作台'}
          </h1>
          {error && (
            <p role="alert" className="account-error">
              {error}
            </p>
          )}
          {notice && (
            <p role="status" className="account-notice">
              {notice}
            </p>
          )}
          {state === 'error' && (
            <button type="button" onClick={() => void check()}>
              重试
            </button>
          )}
          {state === 'login' && (
            <>
              <div className="account-tabs" role="tablist" aria-label="账号操作">
                <button
                  role="tab"
                  aria-selected={!register}
                  onClick={() => {
                    setRegister(false)
                    setError('')
                  }}
                >
                  登录
                </button>
                <button
                  role="tab"
                  aria-selected={register}
                  onClick={() => {
                    setRegister(true)
                    setError('')
                  }}
                >
                  注册
                </button>
              </div>
              <form key={String(register)} onSubmit={event => void submit(event)}>
                <label>
                  用户名
                  <input
                    name="username"
                    autoComplete="username"
                    required
                    minLength={3}
                    maxLength={32}
                    pattern="[a-zA-Z0-9][a-zA-Z0-9_-]{2,31}"
                  />
                </label>
                <label>
                  密码
                  <input
                    type="password"
                    name="password"
                    autoComplete={register ? 'new-password' : 'current-password'}
                    required
                    minLength={register ? 12 : 1}
                    maxLength={128}
                  />
                </label>
                {register && (
                  <label>
                    邀请码
                    <input name="invitation" autoComplete="off" required maxLength={128} />
                  </label>
                )}
                <button className="account-submit" type="submit" disabled={pending}>
                  {pending ? '处理中…' : register ? '注册账号' : '登录'}
                </button>
              </form>
            </>
          )}
        </section>
      </main>
    )
  return (
    <div className={user ? 'account-shell' : undefined}>
      {user && (
        <div className="account-bar">
          <span>
            <UserRound size={16} />
            {user.username}
          </span>
          {error && (
            <span role="alert" className="account-error">
              {error}
            </span>
          )}
          <button
            className="icon-button"
            title="修改密码"
            aria-label="修改密码"
            disabled={pending}
            onClick={() => {
              setPasswordDialog(true)
              setError('')
            }}
          >
            <KeyRound size={17} />
          </button>
          <button
            className="icon-button"
            title="退出登录"
            aria-label="退出登录"
            disabled={pending}
            onClick={() => void logout()}
          >
            <LogOut size={17} />
          </button>
        </div>
      )}
      <div key={user?.id || 'local'}>{children}</div>
      {passwordDialog && (
        <div className="account-overlay">
          <section className="account-form" role="dialog" aria-modal="true" aria-labelledby="password-title">
            <div className="account-dialog-title">
              <h2 id="password-title">修改密码</h2>
              <button
                className="icon-button"
                title="关闭"
                aria-label="关闭修改密码"
                onClick={() => setPasswordDialog(false)}
              >
                <X size={18} />
              </button>
            </div>
            <form onSubmit={event => void changePassword(event)}>
              <label>
                原密码
                <input name="old" type="password" required autoComplete="current-password" />
              </label>
              <label>
                新密码
                <input name="new" type="password" required minLength={12} maxLength={128} autoComplete="new-password" />
              </label>
              <label>
                确认新密码
                <input
                  name="confirm"
                  type="password"
                  required
                  minLength={12}
                  maxLength={128}
                  autoComplete="new-password"
                />
              </label>
              {error && (
                <p role="alert" className="account-error">
                  {error}
                </p>
              )}
              <button className="account-submit" disabled={pending}>
                {pending ? '处理中…' : '更新密码'}
              </button>
            </form>
          </section>
        </div>
      )}
    </div>
  )
}
