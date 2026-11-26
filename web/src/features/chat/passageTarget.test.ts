import { describe, expect, it } from 'vitest'
import type { PassageItem } from '@/api/types'
import { numberCitations } from '@/lib/citations'
import { citationTarget, findSentence, normalizeSentence, passageTarget } from './passageTarget'

// Κείμενο γραμμένο για το test, όχι από άρθρο.
const answer =
  'Functions can run for at most 900 seconds [S1]. Memory goes up to 3 GiB [S1, S2].\n' +
  '- **Storage** is limited to 0.5 GiB [S1].'

const item = (i: number, sentence: string, labels: string[], extra: Partial<PassageItem> = {}): PassageItem => ({
  i,
  sentence,
  labels,
  status: 'found',
  quote: null,
  label: null,
  file: 'paper.pdf',
  page: 5,
  ...extra,
})

// Οι προτάσεις όπως τις στέλνει ο backend: χωρίς παραπομπές και χωρίς **.
const items: PassageItem[] = [
  item(1, 'Functions can run for at most 900 seconds.', ['S1'], { quote: 'Maximum Run Time 900 seconds', label: 'S1' }),
  item(2, 'Memory goes up to 3 GiB.', ['S1', 'S2'], { quote: 'up to 3 GiB of memory', label: 'S2', page: 3 }),
  item(3, 'Storage is limited to 0.5 GiB.', ['S1'], { quote: 'Maximum Local Storage 0.5 GiB', label: 'S1' }),
]

describe('findSentence', () => {
  const { contexts } = numberCitations(answer)

  it('finds the sentence of each citation, even when the same source is cited by several sentences', () => {
    expect(findSentence(items, 'S1', contexts[0] ?? '')?.i).toBe(1)
    expect(findSentence(items, 'S1', contexts[1] ?? '')?.i).toBe(2)
    expect(findSentence(items, 'S2', contexts[1] ?? '')?.i).toBe(2)
    expect(findSentence(items, 'S1', contexts[2] ?? '')?.i).toBe(3)
  })

  it('returns null rather than guessing when the context does not match', () => {
    expect(findSentence(items, 'S1', 'Something unrelated entirely ')).toBeNull()
  })

  it('returns null for a label no sentence cites', () => {
    expect(findSentence(items, 'S9', contexts[0] ?? '')).toBeNull()
  })

  it('normalizes markdown, bullets and citations the same way on both sides', () => {
    expect(normalizeSentence('- **Storage** is limited to 0.5 GiB [S1] .')).toBe('storage is limited to 0.5 gib.')
  })
})

describe('citationTarget', () => {
  it('highlights the quote when it comes from the clicked source', () => {
    expect(citationTarget(items[0] ?? null, 'S1')).toEqual({ label: 'S1', quote: 'Maximum Run Time 900 seconds', note: null, alternate: null })
  })

  it('does not highlight another source’s quote on the clicked page', () => {
    const t = citationTarget(items[1] ?? null, 'S1')
    expect(t.label).toBe('S1')
    expect(t.quote).toBeNull()
    expect(t.note).toBe('other_source')
    expect(t.alternate).toEqual({ label: 'S2', quote: 'up to 3 GiB of memory' })
  })

  it('opens the page where the passage was found when it differs from the cited page', () => {
    const elsewhere = item(4, 'X is Y.', ['S1'], { status: 'found_elsewhere', quote: 'X is Y', label: 'S4' })
    expect(citationTarget(elsewhere, 'S1')).toEqual({
      label: 'S4',
      quote: 'X is Y',
      note: 'elsewhere',
      alternate: { label: 'S1', quote: null },
    })
  })

  it('never highlights when unsure or not found', () => {
    expect(citationTarget(null, 'S1')).toMatchObject({ quote: null, note: 'unsure' })
    expect(citationTarget(item(5, 'Z.', ['S1'], { status: 'not_found' }), 'S1')).toMatchObject({ quote: null, note: 'not_found' })
  })
})

describe('passageTarget', () => {
  it('opens the cited page without highlight when no passage was found', () => {
    expect(passageTarget(item(6, 'Z.', ['S2'], { status: 'not_found' }))).toEqual({
      label: 'S2',
      quote: null,
      note: 'not_found',
      alternate: null,
    })
  })

  it('returns null for a sentence without citation', () => {
    expect(passageTarget(item(7, 'Z.', [], { status: 'no_citation' }))).toBeNull()
  })
})
