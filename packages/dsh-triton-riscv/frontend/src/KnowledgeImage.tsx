import { ExternalLink } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { authFetch } from './auth'
import { KnowledgeSection } from './admin-api'

export function KnowledgeImage({ src, filename, section, onForbidden }: {
  src: string; filename: string; section?: KnowledgeSection; onForbidden: () => void
}) {
  const [image, setImage] = useState<{ src: string; url: string } | null>(null)
  const [error, setError] = useState('')
  const forbidden = useRef(onForbidden)
  forbidden.current = onForbidden
  useEffect(() => {
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 15000)
    let url = '', disposed = false
    setError('')
    void (async () => {
      try {
        const response = await authFetch(src, { signal: controller.signal })
        if (response.status === 403) forbidden.current()
        if (!response.ok) throw new Error(`图片预览不可用 (${response.status})`)
        if (response.headers.get('content-type')?.split(';')[0] !== 'image/png') throw new Error('图片预览格式无效')
        const blob = await response.blob()
        if (blob.size > 8 * 1024 * 1024) throw new Error('图片预览超过大小限制')
        if (disposed || controller.signal.aborted) return
        url = URL.createObjectURL(blob)
        setImage({ src, url })
      } catch (reason) {
        if (!disposed) setError(controller.signal.aborted ? '图片预览超时，请重新读取。' : reason instanceof Error ? reason.message : '图片预览失败')
      } finally { window.clearTimeout(timeout) }
    })()
    return () => { disposed = true; controller.abort(); window.clearTimeout(timeout); if (url) URL.revokeObjectURL(url) }
  }, [src])
  const visible = image?.src === src ? image : null
  const box = section?.bbox, size = section?.image_size
  const region = box && size && size[0] > 0 && size[1] > 0 ? {
    left: `${100 * box[0] / size[0]}%`, top: `${100 * box[1] / size[1]}%`,
    width: `${100 * box[2] / size[0]}%`, height: `${100 * box[3] / size[1]}%`,
  } : undefined
  return <figure className="knowledge-image" aria-label="源图片预览">
    <figcaption><span>{filename}</span>{visible && <a className="icon-button" href={visible.url} target="_blank" rel="noopener noreferrer" aria-label="打开完整图片预览" title="打开完整图片预览"><ExternalLink size={16} /></a>}</figcaption>
    {error && <p className="admin-warning" role="status">{error}</p>}
    {!visible && !error && <p role="status">正在读取图片…</p>}
    {visible && <div className="knowledge-image-scroll"><div className="knowledge-image-canvas">
      <img src={visible.url} alt={`原图：${filename}`} onError={() => setError('图片预览无法显示，请重新读取。')} />
      {region && <div className="knowledge-image-region" style={region} aria-label="当前 OCR 文字区域" />}
    </div></div>}
  </figure>
}
