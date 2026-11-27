import { describe, expect, it } from 'vitest'
import { BASE_CPS, nextRevealLength } from './smoothReveal'

const text = 'Functions can run for at most 900 seconds [S1]. Memory goes up to 3 GiB [S1, S2].'

describe('nextRevealLength', () => {
  it('reveals about as fast as the old Streamlit typing, ending on a word boundary', () => {
    const n = nextRevealLength(0, text, 20, false) // ~9 χαρακτήρες σε 20 ms
    expect(n).toBeGreaterThanOrEqual(Math.round((BASE_CPS * 20) / 1000))
    expect(n).toBeLessThan(text.length)
    expect(text[n] === ' ' || n === text.length).toBe(true)
  })

  it('never shows half a citation: it jumps to the closing bracket', () => {
    const open = text.indexOf('[S1, S2]')
    const n = nextRevealLength(open, text, 5, false)
    expect(text.slice(0, n).endsWith('[S1, S2]')).toBe(true)
  })

  it('waits at an open citation whose closing bracket has not arrived yet', () => {
    const partial = 'Memory goes up to 3 GiB [S1, S'
    const at = partial.indexOf('[')
    expect(nextRevealLength(at, partial, 100, false)).toBe(at)
  })

  it('keeps up with a large backlog, and finishes quickly once the stream ends', () => {
    const long = 'word '.repeat(2000) // 10.000 χαρακτήρες
    expect(nextRevealLength(0, long, 1000, false)).toBeGreaterThanOrEqual(long.length / 2.5 - 5)
    expect(nextRevealLength(0, long, 600, true)).toBe(long.length)
  })

  it('never goes backwards and stops at the end', () => {
    expect(nextRevealLength(text.length, text, 50, false)).toBe(text.length)
    expect(nextRevealLength(10, text, 0, false)).toBeGreaterThanOrEqual(10)
  })
})
