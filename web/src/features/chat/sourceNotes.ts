import type { PassageItem, Source } from '@/api/types'
import { strings } from '@/i18n/en'
import { citedLabels, numberCitations } from '@/lib/citations'

const s = strings.app

/** Οι ουδέτερες σημειώσεις για τη διαδρομή της απάντησης (όπως στο Streamlit, κύκλος 2). */
export function sourceNotes(sources: Source[]): string[] {
  const notes: string[] = []
  if (sources.some((x) => x.second_search)) notes.push(s.secondSearch)
  if (sources.some((x) => x.full_text)) {
    const docs = new Set(sources.map((x) => `${x.doc_id ?? ''}|${x.file}`)).size
    notes.push(s.fullText(docs, sources.length))
  }
  return notes
}

export interface CompareDoc {
  id: number
  file: string
}

/**
 * Τα άρθρα για το «Compare only these»: ΜΟΝΟ όταν η απάντηση παρέθεσε 2–3 διαφορετικά άρθρα μέσω αναζήτησης (όχι όλο
 * το κείμενο). Εκεί η επιλογή μόνο αυτών μετρήθηκε καλύτερη (κλειστό σετ: δύο άρθρων με επιλογή 12/12, χωρίς 9/12).
 */
export function compareCandidates(answer: string, sources: Source[]): CompareDoc[] {
  if (sources.some((x) => x.full_text)) return []
  const cited = citedLabels(answer)
  const docs = new Map<number, string>()
  for (const x of sources)
    if (x.label && cited.has(x.label) && typeof x.doc_id === 'number' && x.doc_id >= 0) docs.set(x.doc_id, x.file)
  return docs.size >= 2 && docs.size <= 3 ? [...docs].map(([id, file]) => ({ id, file })) : []
}

/** Η απάντηση για αντιγραφή: οι αριθμοί όπως στην οθόνη, και στο τέλος οι πηγές που παρατέθηκαν. */
export function answerForCopy(answer: string, sources: Source[]): string {
  const { markdown, order } = numberCitations(answer)
  const body = markdown.replace(/\[(\d+)\]\(#cite-S\d+-\d+\)/g, '[$1]')
  if (order.length === 0) return body
  const byLabel = new Map(sources.map((x) => [x.label, x]))
  const list = order.map((label, i) => {
    const src = byLabel.get(label)
    return `[${i + 1}] ${src?.file ?? label}${src?.page ? `, p. ${src.page}` : ''}`
  })
  return `${body}\n\n${s.sources}:\n${list.join('\n')}`
}

export interface EvidenceCounts {
  cited: number
  found: number
  elsewhere: number
  missing: number
  absence: number
  uncited: number
}

/** Η σύνοψη της προβολής τεκμηρίων, από τα χωρία ανά πρόταση. */
export function evidenceCounts(items: PassageItem[]): EvidenceCounts {
  const c: EvidenceCounts = { cited: 0, found: 0, elsewhere: 0, missing: 0, absence: 0, uncited: 0 }
  for (const it of items) {
    if (it.status === 'no_citation') {
      c.uncited++
      continue
    }
    c.cited++
    if (it.status === 'found') c.found++
    else if (it.status === 'found_elsewhere') c.elsewhere++
    else if (it.status === 'absence') c.absence++
    else c.missing++
  }
  return c
}
