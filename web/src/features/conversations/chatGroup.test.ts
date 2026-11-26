import { describe, expect, it } from 'vitest'
import { chatGroup } from './chatGroup'

describe('chatGroup', () => {
  const now = new Date(2026, 10, 26, 15, 0) // 26/11/2026, τοπική ώρα

  it('puts conversations into the same buckets as the Streamlit sidebar', () => {
    expect(chatGroup(new Date(2026, 10, 26, 9, 0).toISOString(), now)).toBe('Today')
    expect(chatGroup(new Date(2026, 10, 25, 23, 0).toISOString(), now)).toBe('Yesterday')
    expect(chatGroup(new Date(2026, 10, 21, 12, 0).toISOString(), now)).toBe('Previous 7 days')
    expect(chatGroup(new Date(2026, 9, 1).toISOString(), now)).toBe('Earlier')
  })

  it('does not invent a date for conversations without one', () => {
    expect(chatGroup(null, now)).toBe('Earlier')
    expect(chatGroup('not a date', now)).toBe('Earlier')
  })
})
