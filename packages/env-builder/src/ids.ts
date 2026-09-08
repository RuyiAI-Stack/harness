export function nextProjectId(existingIds: string[]): string {
  const used = new Set<number>()
  for (const id of existingIds) {
    const m = /^project(\d+)$/.exec(id)
    if (!m) continue
    used.add(Number(m[1]))
  }
  let n = 1
  while (used.has(n)) n += 1
  return `project${n}`
}
