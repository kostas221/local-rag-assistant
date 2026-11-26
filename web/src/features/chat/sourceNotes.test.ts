import { describe, expect, it } from 'vitest'
import type { PassageItem, Source } from '@/api/types'
import { numberCitations } from '@/lib/citations'
import { answerForCopy, compareCandidates, evidenceCounts, sourceNotes } from './sourceNotes'

const src = (label: string, doc: number, file: string, page: number, extra: Partial<Source> = {}): Source => ({
  label,
  doc_id: doc,
  file,
  page,
  ...extra,
})

describe('compareCandidates', () => {
  const sources = [src('S1', 1, 'a.pdf', 2), src('S2', 2, 'b.pdf', 4), src('S3', 3, 'c.pdf', 1)]

  it('offers the cited papers when 2–3 different papers are cited through search', () => {
    expect(compareCandidates('X [S1]. Y [S2].', sources)).toEqual([
      { id: 1, file: 'a.pdf' },
      { id: 2, file: 'b.pdf' },
    ])
  })

  it('offers nothing for one paper, for uncited pages, or when the full text was already read', () => {
    expect(compareCandidates('X [S1]. Y [S1].', sources)).toEqual([])
    expect(compareCandidates('X [S1].', sources)).toEqual([])
    const full = sources.map((x) => ({ ...x, full_text: true }))
    expect(compareCandidates('X [S1]. Y [S2].', full)).toEqual([])
  })
})

describe('answerForCopy', () => {
  it('keeps the on-screen numbers and lists the cited sources', () => {
    const text = answerForCopy('Run time is 900 s [S3]. Storage [S1, S3].', [src('S1', 1, 'a.pdf', 2), src('S3', 1, 'a.pdf', 5)])
    expect(text).toBe('Run time is 900 s [1]. Storage [2][1].\n\nSources:\n[1] a.pdf, p. 5\n[2] a.pdf, p. 2')
  })

  it('leaves an answer without citations as it is', () => {
    expect(answerForCopy('No citations here.', [])).toBe('No citations here.')
  })
})

describe('evidenceCounts and sourceNotes', () => {
  const it_ = (status: PassageItem['status'], labels: string[] = ['S1']): PassageItem => ({ i: 0, sentence: 'x', labels, status })

  it('counts where passages were found, keeping uncited sentences apart', () => {
    expect(
      evidenceCounts([it_('found'), it_('found'), it_('found_elsewhere'), it_('not_found'), it_('absence'), it_('no_citation', [])]),
    ).toEqual({ cited: 5, found: 2, elsewhere: 1, missing: 1, absence: 1, uncited: 1 })
  })

  it('notes a second search and a full-text read', () => {
    expect(sourceNotes([src('S1', 1, 'a.pdf', 1, { second_search: true })])).toHaveLength(1)
    expect(sourceNotes([src('S1', 1, 'a.pdf', 1, { full_text: true }), src('S2', 2, 'b.pdf', 1, { full_text: true })])[0]).toMatch(
      /2 selected documents/,
    )
  })
})

describe('numberCitations refs', () => {
  it('lists every citation, also inside groups, in reading order', () => {
    expect(numberCitations('A [S2]. B [S1, S2].').refs).toEqual([
      { label: 'S2', occurrence: 0 },
      { label: 'S1', occurrence: 1 },
      { label: 'S2', occurrence: 1 },
    ])
  })
})
