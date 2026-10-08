import { readFile } from 'node:fs/promises'
import vm from 'node:vm'
import { expect, it, vi } from 'vitest'

const source = await readFile(new URL('../../lib/client.js', import.meta.url), 'utf8')

function setup(search = '') {
  let definition
  let dictionaries
  const entries = new Map()
  const styles = new Set()
  const disposers = []
  const history = { state: { session: 'kept' }, replaceState: vi.fn() }
  const sandbox = {
    URL,
    URLSearchParams,
    location: new URL('http://localhost:8782/' + search + '#session-kept'),
    history,
    document: {
      createElement: () => {
        const style = { textContent: '', remove: () => styles.delete(style) }
        return style
      },
      head: { append: style => styles.add(style) },
    },
    window: { __ModuleLoader__: { load: value => (definition = value) } },
  }
  vm.runInNewContext(source, sandbox)
  const primitives = Object.fromEntries(
    ['Modal', 'Tooltip', 'IconCodeOutline16', 'IconCloseOutline16', 'IconRefreshOutline16'].map(k => [k, k]),
  )
  const client = definition.factory(name => {
    if (name === '@deepseek-ai/dsh-client-ui-primitives') return primitives
    expect(name).toBe('react')
    return {
      createElement: (type, props, ...children) => ({ type, props: props || {}, children }),
      useSyncExternalStore: (_subscribe, snapshot) => snapshot(),
      useState: value => [value, vi.fn()],
      useRef: () => ({ current: null }),
      useEffect: () => {},
    }
  })
  const inject = vi.fn((_name, fn) => fn())
  client.apply({
    effect: fn => disposers.push(fn()),
    locale: {
      register: (_name, data) => {
        dictionaries = data
        return () => {}
      },
    },
    slots: {
      inject,
      register: (options, component) => {
        entries.set(options.name, { options, component })
        return () => entries.delete(options.name)
      },
    },
  })
  const t = key => dictionaries.zh[key]
  const entry = (wide = true) => entries.get('sidebar.footer.action').component({ wide, t })
  const panel = () => entries.get('shell.overlay').component({ t })
  return {
    definition,
    client,
    entries,
    styles,
    history,
    inject,
    entry,
    panel,
    dispose: () => disposers.reverse().forEach(fn => fn?.()),
  }
}

it('uses declared sidebar/overlay slots and leaves the native root in control', () => {
  const h = setup()
  expect(h.definition.id).toBe('dsh-triton-riscv')
  expect(Array.from(h.client.inject)).toEqual(['slots', 'locale'])
  expect([...h.entries.keys()]).toEqual(['sidebar.footer.action', 'shell.overlay'])
  expect(h.inject.mock.calls.map(call => call[0])).toEqual(['sidebar.footer.action', 'shell.overlay'])
  expect(h.panel().props.open).toBe(false)
  expect(h.panel().children[1]).toBeNull()
  h.dispose()
  expect(h.entries.size).toBe(0)
  expect(h.styles.size).toBe(0)
})

it('opens on click and closes without navigation, preserving the native session', () => {
  const h = setup()
  const target = { isConnected: true, focus: vi.fn() }
  h.entry().children[0].props.onClick({ currentTarget: target })
  expect(h.panel().props.open).toBe(true)
  expect(h.panel().children[1].props.src).toBe('http://127.0.0.1:8765/')
  h.panel().props.onClose()
  expect(h.panel().props.open).toBe(false)
  expect(h.history.replaceState).not.toHaveBeenCalled()
  expect(target.focus).toHaveBeenCalledOnce()
})

it('keeps an accessible icon entry when the sidebar is collapsed', () => {
  const h = setup()
  const tooltip = h.entry(false)
  const button = tooltip.children[0]
  expect(tooltip.props.disabled).toBe(false)
  expect(button.props['aria-label']).toBe('算子工作台')
  expect(button.props['aria-haspopup']).toBe('dialog')
  expect(button.props['data-wide']).toBe('false')
  expect(button.children[1]).toBeNull()
})

it('retains deep links as an overlay and clears only its own flag on close', () => {
  const h = setup('?tritonWorkbench=1&keep=yes')
  expect(h.panel().props.open).toBe(true)
  expect(h.entries.has('root')).toBe(false)
  h.panel().props.onClose()
  expect(h.history.replaceState).toHaveBeenCalledWith(h.history.state, '', '/?keep=yes#session-kept')
  expect(h.panel().props.open).toBe(false)
})

it('honors nativeHarness at startup without disabling the visual entry', () => {
  const h = setup('?tritonWorkbench=1&nativeHarness=1')
  expect(h.panel().props.open).toBe(false)
  h.entry().children[0].props.onClick({ currentTarget: null })
  expect(h.panel().props.open).toBe(true)
})

it.each([
  ['http://127.0.0.1:8776/custom', 'http://127.0.0.1:8776/custom'],
  ['https://example.test/workbench', 'https://example.test/workbench'],
  ['javascript:alert(1)', 'http://127.0.0.1:8765'],
  ['data:text/html,test', 'http://127.0.0.1:8765'],
  ['not-a-url', 'http://127.0.0.1:8765'],
  ['http://user:secret@example.test', 'http://127.0.0.1:8765'],
])('validates the iframe address %s', (input, expected) => {
  const h = setup('?tritonWorkbench=1&tritonWorkbenchUrl=' + encodeURIComponent(input))
  expect(h.panel().children[1].props.src).toBe(expected)
})
