import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { RagVectors } from './RagVectors'
import { ragApi } from './rag-api'

afterEach(() => { cleanup(); vi.restoreAllMocks() })
const response = (state: string, count: number | null) => ({ version: '1', complete: true,
  vector_scan_performed: count !== null, checked_at: '2026-11-18T12:00:00Z', checks: [
    { generation: 'test', expected_count: 10, actual_count: count, dimensions: 384, state },
  ] })
it('distinguishes actual vectors from persisted counts without editing anything', async () => {
  const request = vi.spyOn(ragApi, 'vectors').mockResolvedValue(response('missing', 0))
  render(<RagVectors itemId="document" version="1" sourceId="current" onForbidden={vi.fn()} />)
  await screen.findByText('未找到该版本的向量')
  expect(screen.getByText('0')).toBeInTheDocument()
  expect(screen.getByText('10')).toBeInTheDocument()
  expect(request).toHaveBeenCalledWith('document', '1', expect.any(AbortSignal), 'current')
})
it('does not show a timeout as a missing vector and supports a fresh check', async () => {
  vi.spyOn(ragApi, 'vectors').mockResolvedValueOnce(response('unavailable', null)).mockResolvedValue(response('present', 10))
  render(<RagVectors itemId="document" version="1" sourceId="current" onForbidden={vi.fn()} />)
  await screen.findByText('向量服务暂不可核对')
  expect(screen.getByText('未知')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '重新核对向量' }))
  await screen.findByText('实际向量数量与记录一致')
})
it('rejects mismatched versions rather than showing stale evidence', async () => {
  vi.spyOn(ragApi, 'vectors').mockResolvedValue({ ...response('present', 10), version: '2' })
  render(<RagVectors itemId="document" version="1" sourceId="current" onForbidden={vi.fn()} />)
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('向量核对结果与当前版本不一致'))
  expect(screen.queryByText('实际向量数量与记录一致')).not.toBeInTheDocument()
})
