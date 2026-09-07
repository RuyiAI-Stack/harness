window.__ModuleLoader__.load({
  id: 'dsh-triton-riscv',
  factory: (require) => {
    const module = { exports: {} }
    const { createElement } = require('react')

    const inject = ['slots']

    function workbenchUrl() {
      const fallback = 'http://127.0.0.1:8765'
      const configured = new URLSearchParams(globalThis.location?.search ?? '')
        .get('tritonWorkbenchUrl') ?? fallback
      try {
        const parsed = new URL(configured)
        return ['http:', 'https:'].includes(parsed.protocol) ? parsed.toString() : fallback
      } catch {
        return fallback
      }
    }

    function WorkbenchRoot() {
      return createElement('iframe', {
        src: workbenchUrl(),
        title: 'Triton-RISCV Agent',
        allow: 'clipboard-read; clipboard-write',
        style: {
          width: '100vw',
          height: '100vh',
          display: 'block',
          border: 0,
          background: '#fff',
        },
      })
    }

    function apply(ctx) {
      const params = new URLSearchParams(globalThis.location?.search ?? '')
      if (params.get('nativeHarness') === '1') return
      ctx.effect(
        () => ctx.slots.register({
          name: 'root',
          priority: -100,
        }, WorkbenchRoot),
        'triton-riscv.workbench-root',
      )
    }

    module.exports = { apply, inject }
    return module.exports
  },
})
