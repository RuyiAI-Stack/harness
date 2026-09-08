window.__ModuleLoader__.load({
  id: '@dangosys/dsh-env-sidebar',
  factory: require => {
    const module = { exports: {} }
    const React = require('react')
    const {
      IconPlusOutline16,
      IconProjectAddOutline16,
      IconRefreshOutline16,
      IconTrashOutline16,
      IconCloseFill14,
      Tooltip,
    } = require('@deepseek-ai/dsh-client-ui-primitives')
    const { useEffect, useState, useSyncExternalStore } = React
    const h = React.createElement
    const API = '/integrations/env-builder'
    const ROLE_REPOS = '/integrations/role/repos'
    const STYLE_ID = 'dsh-env-sidebar-style'
    const STATUSES = new Set(['ready', 'modified', 'installing'])
    const CSS = `
.dsh-env-root{--dsh-env-panel:var(--dsw-alias-bg-secondary,#f3f5f8);--dsh-env-surface:var(--dsw-alias-bg-primary,#fff);--dsh-env-surface-muted:var(--dsw-alias-bg-tertiary,#f8fafc);--dsh-env-border:var(--dsw-alias-border-l3,#dfe4eb);--dsh-env-text:var(--dsw-alias-label-primary,#1f2937);--dsh-env-muted:var(--dsw-alias-label-tertiary,#8993a3);display:flex;flex-direction:column;gap:10px;padding:10px;min-height:0;height:100%;box-sizing:border-box;background:var(--dsh-env-panel);font:13px system-ui;color:var(--dsh-env-text)}
@media (prefers-color-scheme:dark){.dsh-env-root{--dsh-env-panel:var(--dsw-alias-bg-secondary,#171b23);--dsh-env-surface:var(--dsw-alias-bg-primary,#1c212b);--dsh-env-surface-muted:var(--dsw-alias-bg-tertiary,#202733);--dsh-env-border:var(--dsw-alias-border-l3,#303846);--dsh-env-text:var(--dsw-alias-label-primary,#e5e7eb);--dsh-env-muted:var(--dsw-alias-label-tertiary,#8d98aa)}}
.dsh-env-header{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:30px}
.dsh-env-title{font-weight:650;font-size:15px;letter-spacing:-.01em;color:var(--dsh-env-text)}
.dsh-env-actions{display:flex;align-items:center;gap:5px}
.dsh-env-icon{position:relative;display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border:1px solid var(--dsh-env-border);border-radius:9px;padding:0;background:var(--dsh-env-surface);color:var(--dsw-alias-label-secondary,#687384);cursor:pointer}
.dsh-env-icon:hover{background:var(--dsh-env-surface-muted);border-color:#b8c3d1}
.dsh-env-icon:focus-visible,.dsh-env-session:focus-visible,.dsh-env-name:focus-visible,.dsh-env-tag:focus-visible{outline:2px solid #3b82f6;outline-offset:1px}
.dsh-env-icon:disabled{opacity:.4;cursor:default}
.dsh-env-icon.danger{color:#b42318;background:#fee2e2}
.dsh-env-list{display:flex;flex:1;flex-direction:column;gap:8px;min-height:0;overflow:auto}
.dsh-env-card{position:relative;display:flex;flex-direction:column;gap:8px;border:1px solid var(--dsh-env-border);border-radius:14px;padding:10px;background:var(--dsh-env-surface);box-shadow:0 1px 2px rgba(15,23,42,.04)}
.dsh-env-card.on{border-color:#6ea8ff;background:color-mix(in srgb,#3b82f6 6%,var(--dsh-env-surface));box-shadow:0 0 0 1px #3b82f6,0 2px 6px rgba(59,130,246,.12)}
.dsh-env-card-top{display:flex;align-items:center;gap:7px;min-width:0;min-height:28px}
.dsh-env-pin{width:9px;height:9px;flex:0 0 auto;border:2px solid #3b82f6;border-radius:50%;background:#93c5fd;box-sizing:border-box}
.dsh-env-name{border:0;background:transparent;padding:0;font:inherit;font-size:15px;font-weight:550;cursor:pointer;color:inherit;text-align:left;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dsh-env-live{display:inline-flex;align-items:center;gap:5px;height:18px;margin-left:5px;padding:0 8px;border-radius:999px;background:#10b981;color:#062c22;font-size:9px;letter-spacing:.04em;font-weight:750}
.dsh-env-live-dot{width:5px;height:5px;border-radius:50%;background:#d1fae5}
.dsh-env-tags{display:flex;flex-wrap:wrap;gap:6px}
.dsh-env-tag{position:relative;display:inline-flex;align-items:center;gap:5px;border:1px solid transparent;border-radius:6px;padding:3px 8px 3px 9px;font-size:11px;font-weight:500;cursor:pointer;background:var(--dsh-env-surface-muted);color:var(--dsh-env-text);white-space:nowrap}
.dsh-env-tag::before{width:3px;align-self:stretch;border-radius:2px;background:var(--dsh-env-tag-color);content:""}
.dsh-env-tag.ready{--dsh-env-tag-color:#10b981}
.dsh-env-tag.modified{--dsh-env-tag-color:#f59e0b}
.dsh-env-tag.installing{--dsh-env-tag-color:#94a3b8}
.dsh-env-tag.on{border-color:#3b82f6;box-shadow:inset 0 0 0 1px rgba(59,130,246,.18);background:color-mix(in srgb,#3b82f6 10%,var(--dsh-env-surface-muted))}
.dsh-env-tag-diamond{display:none}
.dsh-env-x{position:absolute;top:-6px;right:-6px;width:16px;height:16px;border:2px solid var(--dsh-env-surface);border-radius:50%;padding:0;background:#ef4444;color:#fff;font-size:10px;line-height:12px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;z-index:1}
.dsh-env-sessions{display:flex;flex-direction:column;gap:6px;margin-top:4px}
.dsh-env-sessions-head{display:flex;align-items:center;justify-content:space-between;padding:0 4px}
.dsh-env-sessions-label{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--dsh-env-muted)}
.dsh-env-session-tree{display:flex;flex-direction:column;gap:2px;border:1px solid var(--dsh-env-border);border-radius:10px;padding:4px;background:var(--dsh-env-surface-muted)}
.dsh-env-session{display:flex;align-items:center;gap:8px;min-height:30px;border:1px solid transparent;border-radius:7px;padding:4px 8px;background:transparent;font-size:12px;width:100%;text-align:left;cursor:pointer;color:inherit;font:inherit}
.dsh-env-session:hover{background:color-mix(in srgb,#3b82f6 6%,var(--dsh-env-surface))}
.dsh-env-session.on{background:color-mix(in srgb,#3b82f6 10%,var(--dsh-env-surface));border-color:#9ec5ff}
.dsh-env-mark{width:8px;height:8px;border:0;border-radius:50%;flex:0 0 auto;background:#3b82f6}
.dsh-env-mark.subagent{background:#8b5cf6}
.dsh-env-mark.ghost{border:1px solid #8b5cf6;background:transparent}
.dsh-env-session-activity{margin-left:auto;color:var(--dsh-env-muted);font-size:11px;white-space:nowrap}
.dsh-env-session-kids{display:flex;flex-direction:column;gap:2px;margin-left:14px;padding-left:9px;border-left:1px solid var(--dsh-env-border)}
.dsh-env-err{color:#b42318;white-space:pre-wrap;font-size:12px}
`

    async function api(method, path, body) {
      const res = await fetch(`${API}${path}`, {
        method,
        headers: body ? { 'content-type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      })
      const text = await res.text()
      if (!res.ok) throw new Error(text || res.statusText)
      if (!text) return null
      const type = res.headers.get('content-type') || ''
      if (type.includes('application/json')) return JSON.parse(text)
      return text
    }
    async function addRepoList(repo) {
      const res = await fetch(ROLE_REPOS, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ repo }),
      })
      const text = await res.text()
      if (!res.ok) throw new Error(text || res.statusText)
    }
    function assertEnv(env) {
      if (typeof env.running !== 'boolean') throw new Error(`env-sidebar: env ${env.id} missing boolean running`)
      if (!Array.isArray(env.components)) throw new Error(`env-sidebar: env ${env.id} missing components`)
      for (const c of env.components) {
        if (!STATUSES.has(c.status)) throw new Error(`env-sidebar: invalid component status ${c.status}`)
      }
      if (!Array.isArray(env.sessionIds)) throw new Error(`env-sidebar: env ${env.id} missing sessionIds`)
      for (const sid of env.sessionIds) {
        if (typeof sid !== 'string' || !sid) throw new Error(`env-sidebar: env ${env.id} invalid sessionId`)
      }
    }

    function sessionTitle(s) {
      const title = s.displayTitle || s.title
      if (typeof title !== 'string' || !title) throw new Error(`env-sidebar: session ${s.id} missing title`)
      return title
    }

    function buildSessionTree(sessionIds, listState) {
      if (!listState || typeof listState !== 'object') throw new Error('env-sidebar: sessions.list snapshot missing')
      const byId = listState.byId
      if (!byId || typeof byId !== 'object') throw new Error('env-sidebar: sessions.list missing byId')
      const set = new Set(sessionIds)
      const children = new Map()
      for (const id of sessionIds) children.set(id, [])
      const roots = []
      for (const id of sessionIds) {
        const s = byId[id]
        if (!s) throw new Error(`env-sidebar: session ${id} not in sessions.list`)
        const parentId = s.parentId
        if (parentId == null || !set.has(parentId)) {
          // parent outside env.sessionIds (e.g. singularity root) — env-panel tree root
          roots.push(id)
          continue
        }
        if (!byId[parentId]) {
          throw new Error(`env-sidebar: session parent ${parentId} not in sessions.list`)
        }
        children.get(parentId).push(id)
      }
      function node(id) {
        const s = byId[id]
        const activity = typeof s.activity === 'string' ? s.activity : ''
        return {
          id,
          title: sessionTitle(s),
          mark:
            s.origin === 'subagent' && ['idle', 'empty', 'wait', 'waiting'].includes(activity.toLowerCase())
              ? 'ghost'
              : s.origin === 'subagent'
                ? 'subagent'
                : 'session',
          activity,
          on: listState.current === id,
          children: children.get(id).map(node),
        }
      }
      return roots.map(node)
    }

    function EnvironmentsBrowser({ wide, openSession, sessionsList }) {
      const [envs, setEnvs] = useState([])
      const [sel, setSel] = useState(null)
      const [selRepo, setSelRepo] = useState(null)
      const [err, setErr] = useState('')
      const [busy, setBusy] = useState(false)
      const [deleteMode, setDeleteMode] = useState(false)
      const listState = useSyncExternalStore(
        cb => sessionsList.subscribe(cb),
        () => sessionsList.getSnapshot(),
      )
      async function reload() {
        const [list, selected] = await Promise.all([api('GET', '/environments'), api('GET', '/selected')])
        if (!Array.isArray(list)) throw new Error('env-sidebar: environments is not an array')
        for (const env of list) assertEnv(env)
        setEnvs(list)
        if (selected == null) {
          setSel(null)
          setSelRepo(null)
          return
        }
        if (typeof selected.running !== 'boolean') throw new Error('env-sidebar: selected missing boolean running')
        setSel(selected.id)
        setSelRepo(selected.repo ?? null)
      }

      useEffect(() => {
        reload().catch(e => setErr(String(e.message || e)))
      }, [])

      async function run(fn) {
        setBusy(true)
        setErr('')
        try {
          await fn()
          await reload()
        } catch (e) {
          setErr(String(e.message || e))
        } finally {
          setBusy(false)
        }
      }

      if (!wide) {
        return h('div', { className: 'dsh-env-root', title: 'Environments' }, 'E')
      }

      return h(
        'div',
        { className: 'dsh-env-root' },
        h(
          'div',
          { className: 'dsh-env-header' },
          h('span', { className: 'dsh-env-title' }, 'Environments'),
          h(
            'div',
            { className: 'dsh-env-actions' },
            deleteMode
              ? h(
                  Tooltip,
                  { label: 'Exit delete mode', side: 'bottom', delayMs: 500 },
                  h(
                    'button',
                    {
                      type: 'button',
                      className: 'dsh-env-icon danger',
                      'aria-label': 'Exit delete mode',
                      onClick: () => setDeleteMode(false),
                    },
                    h(IconCloseFill14),
                  ),
                )
              : h(
                  Tooltip,
                  { label: 'New environment', side: 'bottom', delayMs: 500 },
                  h(
                    'button',
                    {
                      type: 'button',
                      className: 'dsh-env-icon',
                      'aria-label': 'New environment',
                      disabled: busy,
                      onClick: () => run(() => api('POST', '/environments')),
                    },
                    h(IconProjectAddOutline16),
                  ),
                ),
            !deleteMode &&
              h(
                'button',
                {
                  type: 'button',
                  className: 'dsh-env-icon',
                  'aria-label': 'Enter delete mode',
                  disabled: busy,
                  onClick: () => setDeleteMode(true),
                },
                h(IconTrashOutline16),
              ),
          ),
        ),
        err && h('div', { className: 'dsh-env-err' }, err),
        h(
          'div',
          { className: 'dsh-env-list' },
          envs.map(e => {
            const on = sel === e.id
            return h(
              'div',
              { key: e.id, className: `dsh-env-card${on ? ' on' : ''}` },
              h(
                'div',
                { className: 'dsh-env-card-top' },
                on && h('span', { className: 'dsh-env-pin', 'aria-hidden': true }),
                h(
                  'button',
                  {
                    type: 'button',
                    className: 'dsh-env-name',
                    'aria-label': `Select or double-click to rename ${e.id}`,
                    onClick: () => void run(() => api('POST', `/environments/${encodeURIComponent(e.id)}/select`)),
                    onDoubleClick: ev => {
                      ev.preventDefault()
                      const id = window.prompt('Rename environment', e.id)
                      if (!id || id === e.id) return
                      void run(() => api('POST', `/environments/${encodeURIComponent(e.id)}/rename`, { id }))
                    },
                  },
                  e.id,
                ),
                e.running &&
                  h('span', { className: 'dsh-env-live' }, h('span', { className: 'dsh-env-live-dot' }), 'Live'),
                deleteMode &&
                  h(
                    'button',
                    {
                      type: 'button',
                      className: 'dsh-env-x',
                      style: { position: 'relative', top: 0, right: 0 },
                      'aria-label': `Delete ${e.id}`,
                      disabled: busy,
                      onClick: () =>
                        void run(async () => {
                          await api('DELETE', `/environments/${encodeURIComponent(e.id)}`)
                          setDeleteMode(false)
                        }),
                    },
                    '×',
                  ),
                h(
                  'div',
                  { className: 'dsh-env-actions', style: { marginLeft: 'auto' } },
                  deleteMode
                    ? null
                    : [
                        h(
                          Tooltip,
                          { key: 'add', label: 'Add component', side: 'bottom', delayMs: 500 },
                          h(
                            'button',
                            {
                              type: 'button',
                              className: 'dsh-env-icon',
                              'aria-label': `Add component to ${e.id}`,
                              disabled: busy,
                              onClick: () => {
                                const repo = window.prompt('GitHub owner/repo or URL')
                                if (!repo) return
                                void run(() =>
                                  api('POST', `/environments/${encodeURIComponent(e.id)}/components`, { repo }),
                                )
                              },
                            },
                            h(IconPlusOutline16),
                          ),
                        ),
                        h(
                          Tooltip,
                          { key: 'reset', label: 'Reset environment', side: 'bottom', delayMs: 500 },
                          h(
                            'button',
                            {
                              type: 'button',
                              className: 'dsh-env-icon',
                              'aria-label': `Reset ${e.id}`,
                              disabled: busy,
                              onClick: () =>
                                void run(() => api('POST', `/environments/${encodeURIComponent(e.id)}/reset`)),
                            },
                            h(IconRefreshOutline16),
                          ),
                        ),
                        on &&
                          selRepo &&
                          h(
                            Tooltip,
                            { key: 'wl', label: 'Add selected to repoList', side: 'bottom', delayMs: 500 },
                            h(
                              'button',
                              {
                                type: 'button',
                                className: 'dsh-env-icon',
                                'aria-label': 'Add selected to repoList',
                                disabled: busy,
                                onClick: () => void run(() => addRepoList(selRepo)),
                              },
                              h(IconProjectAddOutline16),
                            ),
                          ),
                      ],
                ),
              ),
              h(
                'div',
                { className: 'dsh-env-tags' },
                e.components.map(c => {
                  const ref = `${c.owner}/${c.repo}`
                  return h(
                    'button',
                    {
                      key: ref,
                      type: 'button',
                      className: `dsh-env-tag ${c.status}${selRepo === ref ? ' on' : ''}`,
                      onClick: () => {
                        if (deleteMode) return
                        void run(() => api('POST', '/selected-repo', { repo: ref }))
                      },
                    },
                    h('span', { className: 'dsh-env-tag-diamond' }, '◆'),
                    ref,
                    deleteMode &&
                      h(
                        'span',
                        {
                          role: 'button',
                          className: 'dsh-env-x',
                          'aria-label': `Delete ${ref}`,
                          onClick: ev => {
                            ev.stopPropagation()
                            void run(() =>
                              api('DELETE', `/environments/${encodeURIComponent(e.id)}/components`, { repo: ref }),
                            )
                          },
                        },
                        '×',
                      ),
                  )
                }),
              ),
              on &&
                (() => {
                  let tree
                  try {
                    tree = buildSessionTree(e.sessionIds, listState)
                  } catch (ex) {
                    return h('div', { className: 'dsh-env-err' }, String(ex.message || ex))
                  }
                  function renderNode(node) {
                    return h(
                      'div',
                      { key: node.id },
                      h(
                        'button',
                        {
                          type: 'button',
                          className: `dsh-env-session${node.on ? ' on' : ''}`,
                          onClick: () => openSession(node.id),
                        },
                        h('span', { className: `dsh-env-mark ${node.mark}` }),
                        node.title,
                        node.activity && h('span', { className: 'dsh-env-session-activity' }, node.activity),
                      ),
                      node.children.length > 0 &&
                        h('div', { className: 'dsh-env-session-kids' }, node.children.map(renderNode)),
                    )
                  }
                  return h(
                    'div',
                    { className: 'dsh-env-sessions' },
                    h('div', { className: 'dsh-env-sessions-label' }, 'SESSIONS'),
                    h('div', { className: 'dsh-env-session-tree' }, tree.map(renderNode)),
                  )
                })(),
            )
          }),
        ),
      )
    }

    function apply(ctx) {
      ctx.effect(() => {
        const style = document.createElement('style')
        style.id = STYLE_ID
        style.textContent = CSS
        document.head.appendChild(style)
        return () => style.remove()
      }, 'env-sidebar')
      function Browser(props) {
        return h(EnvironmentsBrowser, {
          ...props,
          sessionsList: ctx.sessions.list,
          openSession: sid => {
            ctx.sessions.open(sid)
          },
        })
      }
      ctx.slots.inject('sidebar.workspaces', () =>
        ctx.slots.register({ name: 'sidebar.workspaces', priority: -10 }, Browser),
      )
    }

    module.exports.apply = apply
    module.exports.inject = ['slots', 'sessions']
    return module.exports
  },
})
