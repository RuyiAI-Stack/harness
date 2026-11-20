import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MessageText, shortDate } from './format'
import { setAccount } from './auth'

afterEach(() => {
  cleanup()
  setAccount(null)
})

describe('shortDate', () => {
  it('formats API timestamps for the session list', () => {
    expect(shortDate('2026-08-26T11:17:47')).toBe('08-26 11:17')
  })
})

describe('MessageText', () => {
  it('renders supported message markup without injecting HTML', () => {
    render(<MessageText value={'### 计划\n- 检查 `relu`\n**确认** <script>'} />)
    expect(screen.getByRole('heading', { name: '计划' })).toBeInTheDocument()
    expect(screen.getByText('relu')).toBeInTheDocument()
    expect(screen.getByText('<script>', { exact: false })).toBeInTheDocument()
    expect(document.querySelector('script')).toBeNull()
  })

  it('opens source, tests, receipt and log links through the account-scoped reader', () => {
    setAccount({ id: 'alice', username: 'alice', csrf_token: 'private-csrf' })
    const paths = [
      'python/examples/flaggems/demo_masked_softmax.py',
      'python/examples/flaggems/test_demo_masked_softmax.py',
      'agent-results/operator-lifecycle/receipts/run-example.json',
      '/Users/example/workspace/agent-results/operator-lifecycle/validation/logs/run-example.log',
      'agent-results/operator-development/patches/development-example.diff',
    ]
    render(<MessageText value={paths.map((path, i) => `- [file-${i}](${path})`).join('\n')} />)
    const links = screen.getAllByRole('link')
    expect(links).toHaveLength(5)
    links.forEach((link, i) => {
      const url = new URL(link.getAttribute('href')!, window.location.origin)
      expect(url.pathname).toBe('/api/workspace/files')
      expect(url.searchParams.get('path')).toBe(paths[i])
      expect(url.searchParams.get('account_id')).toBe('alice')
      expect(link).toHaveAttribute('target', '_blank')
      expect(link).toHaveAttribute('rel', 'noopener noreferrer')
      expect(link.getAttribute('href')).not.toContain('private-csrf')
    })
  })

  it('also links paths in old messages that used inline code or plain text', () => {
    render(
      <MessageText
        value={
          '实现：`python/examples/flaggems/relu.py`\n测试：python/examples/flaggems/test_relu.py\n**[需求](tasks/operators/relu.md)**'
        }
      />,
    )
    expect(screen.getAllByRole('link')).toHaveLength(3)
    expect(screen.getByRole('link', { name: '需求' })).toHaveAttribute(
      'href',
      '/api/workspace/files?path=tasks%2Foperators%2Frelu.md',
    )
  })

  it('preserves Unicode, spaces and escaped underscores in Markdown file paths', () => {
    render(
      <MessageText
        value={'[日志](</Users/测试 账号/agent-results/operator-lifecycle/validation/logs/demo\\_run.log>)'}
      />,
    )
    const url = new URL(screen.getByRole('link').getAttribute('href')!, window.location.origin)
    expect(url.searchParams.get('path')).toBe(
      '/Users/测试 账号/agent-results/operator-lifecycle/validation/logs/demo_run.log',
    )
  })

  it('never turns code blocks into clickable execution or file controls', () => {
    render(
      <MessageText
        value={
          '```python\npath = "python/examples/flaggems/demo.py"\n[not a link](https://example.com)\n```\n`python -m pytest test_demo.py`'
        }
      />,
    )
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(document.querySelector('pre')).toHaveTextContent('path = "python/examples/flaggems/demo.py"')
  })

  it('allows web links but never active content, local file URLs or arbitrary API requests', () => {
    render(
      <MessageText
        value={
          '[docs](https://example.com/docs) [unsafe](javascript:alert) [data](data:text/html,boom) [local](file:///etc/passwd) [api](/api/auth/logout) [network](//example.com/file.py)'
        }
      />,
    )
    expect(screen.getAllByRole('link')).toHaveLength(1)
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://example.com/docs')
  })

  it('handles incomplete streamed links and code blocks without losing content', () => {
    const { rerender } = render(<MessageText value={'源码：[file](python/examples/flaggems/demo'} />)
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    rerender(<MessageText value={'```python\nprint("still streaming")'} />)
    expect(document.querySelector('pre')).toHaveTextContent('still streaming')
  })
})
