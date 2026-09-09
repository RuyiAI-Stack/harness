import { closeSync, openSync, readSync, statSync } from 'node:fs'

/** Default number of bytes read from the end of the log file. */
export const TAIL_BYTES = 256 * 1024

/** Return the last `limit` characters of `text`. */
export function tailText(text: string, limit = 8000): string {
  if (text.length <= limit) return text
  return text.slice(-limit)
}

/** Read the last `byteLimit` bytes of a file as UTF-8 text; empty when missing. */
export function readTailFile(file: string, byteLimit = TAIL_BYTES): string {
  let fd: number
  try {
    fd = openSync(file, 'r')
  } catch {
    return ''
  }
  try {
    const { size } = statSync(file)
    if (size === 0) return ''
    const start = Math.max(0, size - byteLimit)
    const buffer = Buffer.allocUnsafe(size - start)
    let offset = 0
    while (offset < buffer.length) {
      const read = readSync(fd, buffer, offset, buffer.length - offset, start + offset)
      if (read <= 0) break
      offset += read
    }
    const text = buffer.toString('utf8')
    // A partial multibyte sequence at the head decodes to one replacement char.
    return start === 0 ? text : text.replace(/^\uFFFD+/, '')
  } finally {
    closeSync(fd)
  }
}
