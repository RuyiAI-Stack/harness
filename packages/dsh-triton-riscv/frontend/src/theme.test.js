import { readFileSync } from 'node:fs'
import { afterAll, describe, expect, it } from 'vitest'

const style = document.createElement('style')
style.textContent = readFileSync('src/styles.css', 'utf8')
document.head.appendChild(style)
const rules = Array.from(style.sheet.cssRules)
const palette = rules.find(rule => rule.selectorText === ':root').style
afterAll(() => style.remove())

function luminance(name) {
  const hex = palette.getPropertyValue(name).trim()
  expect(hex).toMatch(/^#[\da-f]{6}$/i)
  const linear = [1, 3, 5].map(offset => {
    const value = parseInt(hex.slice(offset, offset + 2), 16) / 255
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  })
  return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722
}

describe('dark theme readability', () => {
  it('uses native dark controls and respects reduced motion', () => {
    expect(palette.getPropertyValue('color-scheme')).toBe('dark')
    const reduced = rules.find(rule => rule.conditionText === '(prefers-reduced-motion: reduce)')
    expect(reduced?.cssText).toContain('animation: none !important')
  })

  it.each([
    ['--ink', '--canvas'],
    ['--muted', '--surface-alt'],
    ['--muted', '--surface-raised'],
    ['--green', '--green-soft'],
    ['--blue', '--blue-soft'],
    ['--red', '--red-soft'],
    ['--amber', '--amber-soft'],
    ['--code', '--surface-raised'],
    ['--on-accent', '--green'],
  ])('keeps %s on %s above 4.5:1 contrast', (foreground, background) => {
    const a = luminance(foreground)
    const b = luminance(background)
    expect((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)).toBeGreaterThanOrEqual(4.5)
  })
})
