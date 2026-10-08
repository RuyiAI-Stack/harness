window.__ModuleLoader__.load({
  id: 'dsh-triton-riscv',
  factory: require => {
    const module = { exports: {} }
    const { createElement: h, useEffect, useRef, useState, useSyncExternalStore } = require('react')
    const { Modal, Tooltip, IconCodeOutline16, IconCloseOutline16, IconRefreshOutline16 } = require(
      '@deepseek-ai/dsh-client-ui-primitives',
    )

    const inject = ['slots', 'locale']
    const NS = 'tritonWorkbench'
    const dictionaries = {
      zh: { open: '算子工作台', title: 'Triton-RISCV 算子工作台', close: '返回 Harness', refresh: '刷新工作台' },
      en: {
        open: 'Operator workbench',
        title: 'Triton-RISCV workbench',
        close: 'Return to Harness',
        refresh: 'Reload workbench',
      },
    }
    const CSS = `
.dsh-triton-entry{display:flex;align-items:center;gap:8px;min-width:0;width:100%;height:36px;padding:0 8px;border:0;border-radius:8px;background:transparent;color:var(--dsw-alias-label-primary);font:inherit;cursor:pointer;text-align:left;letter-spacing:0}
.dsh-triton-entry svg{flex:none}
.dsh-triton-entry span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dsh-triton-entry[data-wide="false"]{width:36px;padding:0;justify-content:center}
.dsh-triton-entry:hover,.dsh-triton-action:hover{background:var(--dsw-alias-interactive-bg-hover)}
.dsh-triton-entry:focus-visible,.dsh-triton-action:focus-visible{outline:2px solid currentColor;outline-offset:-2px}
.dsh-triton-dialog.dsh-triton-dialog{width:calc(100vw - 48px);height:calc(100dvh - 48px);max-width:1600px;gap:0;padding:0;border-radius:8px}
.dsh-triton-header{display:flex;align-items:center;gap:8px;flex:none;min-height:48px;padding:6px 12px;box-sizing:border-box;border-bottom:1px solid var(--dsw-alias-border-l3);color:var(--dsw-alias-label-primary)}
.dsh-triton-header h2{flex:1;min-width:0;margin:0;font-size:14px;line-height:20px;font-weight:600;letter-spacing:0;overflow-wrap:anywhere}
.dsh-triton-action{display:inline-flex;align-items:center;justify-content:center;flex:none;width:32px;height:32px;padding:0;border:0;border-radius:6px;background:transparent;color:inherit;cursor:pointer}
.dsh-triton-frame{display:block;flex:1;min-height:0;width:100%;border:0;background:#fff}
`

    function workbenchUrl() {
      const fallback = 'http://127.0.0.1:8765'
      const configured = new URLSearchParams(globalThis.location?.search ?? '').get('tritonWorkbenchUrl') ?? fallback
      try {
        const parsed = new URL(configured)
        return ['http:', 'https:'].includes(parsed.protocol) && !parsed.username && !parsed.password
          ? parsed.toString()
          : fallback
      } catch {
        return fallback
      }
    }

    function apply(ctx) {
      const params = new URLSearchParams(globalThis.location?.search ?? '')
      let open = params.get('nativeHarness') !== '1' && params.get('tritonWorkbench') === '1'
      let returnFocus
      const listeners = new Set()
      const snapshot = () => open
      const subscribe = listener => {
        listeners.add(listener)
        return () => listeners.delete(listener)
      }
      const setOpen = value => {
        open = value
        for (const listener of listeners) listener()
      }
      const close = () => {
        setOpen(false)
        // Clear the legacy deep link without reloading the native conversation.
        const url = new URL(globalThis.location.href)
        if (url.searchParams.has('tritonWorkbench')) {
          url.searchParams.delete('tritonWorkbench')
          globalThis.history.replaceState(globalThis.history.state, '', url.pathname + url.search + url.hash)
        }
        if (returnFocus?.isConnected) returnFocus.focus()
      }

      function WorkbenchEntry({ wide, t }) {
        return h(
          Tooltip,
          { label: t('open'), disabled: wide },
          h(
            'button',
            {
              type: 'button',
              className: 'dsh-triton-entry',
              'data-wide': String(wide),
              'aria-label': t('open'),
              'aria-haspopup': 'dialog',
              onClick: event => {
                returnFocus = event.currentTarget
                setOpen(true)
              },
            },
            h(IconCodeOutline16, { size: 18 }),
            wide ? h('span', null, t('open')) : null,
          ),
        )
      }

      function WorkbenchPanel({ t }) {
        const visible = useSyncExternalStore(subscribe, snapshot, snapshot)
        const [reload, setReload] = useState(0)
        const closeButton = useRef(null)
        useEffect(() => {
          if (visible) closeButton.current?.focus()
        }, [visible])
        const action = (label, icon, onClick, ref) =>
          h(
            Tooltip,
            { label, side: 'bottom' },
            h('button', { type: 'button', className: 'dsh-triton-action', 'aria-label': label, onClick, ref }, h(icon)),
          )
        return h(
          Modal,
          { open: visible, onClose: close, title: t('title'), headless: true, className: 'dsh-triton-dialog' },
          h(
            'header',
            { className: 'dsh-triton-header' },
            h(IconCodeOutline16),
            h('h2', null, t('title')),
            action(t('refresh'), IconRefreshOutline16, () => setReload(value => value + 1)),
            action(t('close'), IconCloseOutline16, close, closeButton),
          ),
          visible
            ? h('iframe', {
                key: reload,
                src: workbenchUrl(),
                title: 'Triton-RISCV Agent',
                className: 'dsh-triton-frame',
                allow: 'clipboard-read; clipboard-write',
              })
            : null,
        )
      }

      ctx.effect(() => ctx.locale.register(NS, dictionaries), 'triton-riscv.workbench-locale')
      ctx.effect(() => {
        const style = document.createElement('style')
        style.textContent = CSS
        document.head.append(style)
        return () => style.remove()
      }, 'triton-riscv.workbench-style')
      ctx.effect(() => () => listeners.clear(), 'triton-riscv.workbench-state')
      // Wait for the owning plugins to declare their slots; never replace root.
      ctx.effect(
        () =>
          ctx.slots.inject('sidebar.footer.action', () =>
            ctx.slots.register(
              { name: 'sidebar.footer.action', id: 'triton-workbench-entry', locale: NS },
              WorkbenchEntry,
            ),
          ),
        'triton-riscv.workbench-entry',
      )
      ctx.effect(
        () =>
          ctx.slots.inject('shell.overlay', () =>
            ctx.slots.register({ name: 'shell.overlay', id: 'triton-workbench-panel', locale: NS }, WorkbenchPanel),
          ),
        'triton-riscv.workbench-panel',
      )
    }

    module.exports = { apply, inject }
    return module.exports
  },
})
