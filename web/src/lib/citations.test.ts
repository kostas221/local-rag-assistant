import { describe, expect, it } from 'vitest'
import { citationFromHref, citedLabels, numberCitations, stripCitations } from './citations'

describe('numberCitations', () => {
  it('numbers labels in order of first appearance and tags each occurrence', () => {
    const { markdown, order } = numberCitations('Run time is 900 s [S3]. Storage is 0.5 GiB [S1]. Again [S3].')
    expect(order).toEqual(['S3', 'S1'])
    expect(markdown).toBe(
      'Run time is 900 s [1](#cite-S3-0). Storage is 0.5 GiB [2](#cite-S1-1). Again [1](#cite-S3-2).',
    )
  })

  it('splits grouped and adjacent labels, with commas or semicolons', () => {
    const { markdown, order } = numberCitations('Both [S1, S3] and [S2][S4; S5].')
    expect(order).toEqual(['S1', 'S3', 'S2', 'S4', 'S5'])
    expect(markdown).toBe(
      'Both [1](#cite-S1-0)[2](#cite-S3-0) and [3](#cite-S2-1)[4](#cite-S4-2)[5](#cite-S5-2).',
    )
  })

  it('keeps the text of the line before each citation', () => {
    const { contexts } = numberCitations('Intro line.\n- Memory is 3 GiB [S1]. Time is 900 s [S2].')
    expect(contexts).toEqual(['- Memory is 3 GiB ', '- Memory is 3 GiB [S1]. Time is 900 s '])
  })

  it('leaves other brackets alone', () => {
    expect(numberCitations('Use [x] and a list [1].').markdown).toBe('Use [x] and a list [1].')
  })
})

describe('citedLabels, citationFromHref, stripCitations', () => {
  it('collects every cited label', () => {
    expect([...citedLabels('a [S1] b [S2, S5] c [S1]')]).toEqual(['S1', 'S2', 'S5'])
  })

  it('reads the label and occurrence back from a citation link only', () => {
    expect(citationFromHref('#cite-S12-3')).toEqual({ label: 'S12', occurrence: 3 })
    expect(citationFromHref('#cite-S12')).toBeNull()
    expect(citationFromHref('https://example.com')).toBeNull()
  })

  it('removes citation groups', () => {
    expect(stripCitations('A [S1, S2] b [S3].')).toBe('A  b .')
  })
})
