import { describe, expect, it } from 'vitest'
import { MessageSchema } from '@/api/types'
import { applyEvent, finishLive, isRunning, startLive, titleFrom } from './liveAnswer'

describe('live answer', () => {
  it('moves through searching → composing → writing → done', () => {
    let s = startLive('Q?', 7, 10, 1000)
    expect(s.phase).toBe('searching')
    s = applyEvent(s, { type: 'sources', data: [{ label: 'S1', file: 'a.pdf', page: 2, doc_id: 1 }] })
    expect(s.phase).toBe('composing')
    s = applyEvent(s, { type: 'text', data: 'Hello ' })
    s = applyEvent(s, { type: 'text', data: 'world [S1].' })
    expect(s.phase).toBe('writing')
    expect(s.text).toBe('Hello world [S1].')
    s = applyEvent(s, { type: 'metrics', data: { retrieval_s: 0.5, generation_s: 2, pages: 1 } })
    expect(isRunning(s)).toBe(true)
    s = finishLive(s, 'done', 4000)
    expect(s.totalS).toBe(3)
    expect(isRunning(s)).toBe(false)
    expect(s.metrics?.pages).toBe(1)
  })

  it('keeps empty sources as a refusal signal and records an incomplete status', () => {
    let s = startLive('Q?', null, 0, 0)
    s = applyEvent(s, { type: 'sources', data: [] })
    expect(s.sources).toEqual([])
    s = applyEvent(s, { type: 'status', data: 'incomplete', reason: 'MAX_TOKENS' })
    expect(s.incomplete).toBe('MAX_TOKENS')
  })

  it('records a stop', () => {
    const s = finishLive(applyEvent(startLive('Q?', 1, 0, 0), { type: 'text', data: 'Part' }), 'stopped', 500)
    expect(s.phase).toBe('stopped')
    expect(s.text).toBe('Part')
  })

  it('makes a title, cutting only when needed', () => {
    expect(titleFrom('  short   question ')).toBe('short question')
    expect(titleFrom('x'.repeat(50))).toBe(`${'x'.repeat(40)}…`)
  })
})

describe('MessageSchema', () => {
  it('parses sources and passages stored as JSON text', () => {
    const m = MessageSchema.parse({
      id: 3,
      conversation_id: 1,
      role: 'assistant',
      content: 'A [S1].',
      sources: JSON.stringify([{ label: 'S1', file: 'a.pdf', page: 4, doc_id: 2 }]),
      passages: JSON.stringify({ status: 'done', items: [] }),
    })
    expect(m.sources[0]?.page).toBe(4)
    expect(m.passages?.status).toBe('done')
  })

  it('survives broken or very old data instead of breaking the screen', () => {
    const m = MessageSchema.parse({ id: 4, role: 'assistant', content: null, sources: '{oops', passages: 'nope' })
    expect(m.content).toBe('')
    expect(m.sources).toEqual([])
    expect(m.passages).toBeNull()
    const old = MessageSchema.parse({ id: 5, role: 'assistant', content: 'x', sources: '["old.pdf"]', passages: null })
    expect(old.sources).toEqual([{ file: 'old.pdf' }])
  })
})
