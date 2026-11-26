import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { PassageItem, Source } from '@/api/types'
import { citationFromHref, numberCitations } from '@/lib/citations'
import { CitationChip, type EvidenceStatus } from './CitationChip'
import { citationTarget, findSentence } from './passageTarget'

/** Μία παραπομπή της απάντησης: ετικέτα, κείμενο πριν από αυτήν (για την πρότασή της) και μοναδικό κλειδί. */
export interface CiteTarget {
  label: string
  context: string
  key: string
}

interface Props {
  answer: string
  sources: Source[]
  /** Τα χωρία, ΜΟΝΟ αν έχει τρέξει ο έλεγχος· αλλιώς η κάρτα δεν δείχνει χωρίο. */
  passages?: PassageItem[]
  /** Πρόθεμα για τα κλειδιά των παραπομπών αυτής της απάντησης (π.χ. το id του μηνύματος). */
  citeKey: string
  activeKey?: string | null
  /** Προβολή τεκμηρίων: κάθε κουμπί δείχνει πού βρέθηκε το χωρίο της πρότασής του. */
  evidence?: boolean
  /** Η παραπομπή που πατήθηκε, και όλες της απάντησης με τη σειρά (για το `[` / `]`). */
  onCitation: (target: CiteTarget, all: CiteTarget[]) => void
}

/**
 * Η απάντηση ως markdown, με τις παραπομπές [S#] ως κουμπιά.
 *
 * ΑΣΦΑΛΕΙΑ: το react-markdown ΔΕΝ αποδίδει HTML (χωρίς rehype-raw). Ένα PDF με κρυμμένη επίθεση δεν μπορεί να βάλει
 * script ή εικόνα μέσω της απάντησης: ό,τι μοιάζει με HTML εμφανίζεται ως κείμενο. Και οι εικόνες markdown δεν
 * φορτώνονται (θα μπορούσαν να στείλουν δεδομένα σε τρίτο server).
 */
export function Answer({ answer, sources, passages, citeKey, activeKey, evidence, onCitation }: Props) {
  const { markdown, order, contexts, refs } = useMemo(() => numberCitations(answer), [answer])
  const byLabel = useMemo(() => new Map(sources.map((s) => [s.label, s])), [sources])
  const all = useMemo<CiteTarget[]>(
    () => refs.map((r) => ({ label: r.label, context: contexts[r.occurrence] ?? '', key: `${citeKey}:${r.occurrence}:${r.label}` })),
    [refs, contexts, citeKey],
  )

  return (
    <div className="prose-answer space-y-3 leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ alt }) => <span className="text-muted-foreground">[{alt || 'image'}]</span>,
          a: ({ href, children }) => {
            const ref = citationFromHref(href)
            if (!ref)
              return (
                <a href={href} target="_blank" rel="noreferrer noopener" className="underline underline-offset-2">
                  {children}
                </a>
              )
            const context = contexts[ref.occurrence] ?? ''
            const key = `${citeKey}:${ref.occurrence}:${ref.label}`
            // Το χωρίο στην κάρτα: μόνο αν είναι σίγουρα της πρότασης ΑΥΤΟΥ του κουμπιού, και από αυτή την πηγή.
            const sentence = passages ? findSentence(passages, ref.label, context) : null
            const target = passages ? citationTarget(sentence, ref.label) : null
            const quote = target && target.label === ref.label ? target.quote : null
            const status: EvidenceStatus | null = evidence && passages ? (sentence?.status ?? 'unsure') : null
            return (
              <CitationChip
                number={order.indexOf(ref.label) + 1}
                source={byLabel.get(ref.label)}
                quote={quote}
                active={key === activeKey}
                evidence={status}
                onOpen={() => onCitation({ label: ref.label, context, key }, all)}
              />
            )
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}
