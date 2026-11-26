import type { Source } from '@/api/types'
import { strings } from '@/i18n/en'
import { citedLabels } from '@/lib/citations'

const s = strings.app

/** Οι ουδέτερες σημειώσεις για τη διαδρομή της απάντησης (όπως στο Streamlit, κύκλος 2). */
export function sourceNotes(answer: string, sources: Source[]): string[] {
  const notes: string[] = []
  if (sources.some((x) => x.second_search)) notes.push(s.secondSearch)
  const full = sources.some((x) => x.full_text)
  if (full) {
    const docs = new Set(sources.map((x) => `${x.doc_id ?? ''}|${x.file}`)).size
    notes.push(s.fullText(docs, sources.length))
  }
  // Σημείωση για συγκρίσεις: ΜΟΝΟ όταν η απάντηση παρέθεσε 2+ διαφορετικά άρθρα μέσω αναζήτησης.
  const cited = citedLabels(answer)
  const citedFiles = new Set(sources.filter((x) => x.label && cited.has(x.label)).map((x) => x.file))
  if (!full && citedFiles.size >= 2) notes.push(s.compareHint)
  return notes
}
