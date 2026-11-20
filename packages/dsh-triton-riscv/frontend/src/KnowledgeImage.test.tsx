import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { authFetch } from './auth'
import { KnowledgeImage } from './KnowledgeImage'
import { validateSourceFile } from './admin-api'

vi.mock('./auth', () => ({ authFetch: vi.fn() }))
const fetch = vi.mocked(authFetch)
const png = () => new Response(new Blob(['image'], { type: 'image/png' }), { headers: { 'Content-Type': 'image/png' } })
beforeEach(() => {
  fetch.mockReset().mockResolvedValue(png())
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:preview') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

it.each(['png', 'PNG', 'jpg', 'JPEG'])('accepts .%s without uploading on selection', extension => {
  expect(validateSourceFile(new File(['bytes'], `screenshot.${extension}`))).toBe('')
  expect(fetch).not.toHaveBeenCalled()
})
it('rejects unsupported and empty images', () => {
  expect(validateSourceFile(new File(['x'], 'image.svg'))).not.toBe('')
  expect(validateSourceFile(new File([], 'image.png'))).toContain('不能为空')
})
it('loads a versioned authenticated preview, highlights the OCR region and revokes it on unmount', async () => {
  const view = render(<KnowledgeImage src="/api/admin/knowledge/a/preview?version=2" filename="report.png"
    section={{ id: 's1', text: 'recognized', locator: 'bbox', edited: false, bbox: [10, 20, 30, 40], image_size: [100, 200] }} onForbidden={vi.fn()} />)
  expect(await screen.findByRole('img')).toHaveAttribute('src', 'blob:preview')
  expect(fetch).toHaveBeenCalledWith('/api/admin/knowledge/a/preview?version=2', expect.objectContaining({ signal: expect.any(AbortSignal) }))
  expect(screen.getByLabelText('当前 OCR 文字区域')).toHaveStyle({ left: '10%', top: '10%', width: '30%', height: '20%' })
  expect(screen.getByRole('link')).toHaveAttribute('rel', 'noopener noreferrer')
  view.unmount()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:preview')
})
it('drops a late response when a different source or version is selected', async () => {
  let resolve!: (response: Response) => void
  fetch.mockImplementationOnce(() => new Promise(done => { resolve = done }))
  const view = render(<KnowledgeImage src="/api/admin/old/preview" filename="old.png" onForbidden={vi.fn()} />)
  view.rerender(<KnowledgeImage src="/api/admin/new/preview" filename="new.png" onForbidden={vi.fn()} />)
  expect(await screen.findByRole('img')).toHaveAttribute('alt', '原图：new.png')
  await act(async () => resolve(png()))
  expect(URL.createObjectURL).toHaveBeenCalledTimes(1)
})
it('does not render non-image responses and reports lost administrator permission', async () => {
  const denied = vi.fn()
  fetch.mockResolvedValueOnce(new Response('', { status: 403 }))
  render(<KnowledgeImage src="/api/admin/private/preview" filename="private.png" onForbidden={denied} />)
  await waitFor(() => expect(denied).toHaveBeenCalledOnce())
  expect(screen.queryByRole('img')).not.toBeInTheDocument()
  cleanup()
  fetch.mockResolvedValueOnce(new Response('<script>no</script>', { headers: { 'Content-Type': 'text/html' } }))
  render(<KnowledgeImage src="/api/admin/bad/preview" filename="bad.png" onForbidden={denied} />)
  expect(await screen.findByText('图片预览格式无效')).toBeInTheDocument()
  expect(URL.createObjectURL).not.toHaveBeenCalled()
})
