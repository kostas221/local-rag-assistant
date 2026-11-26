import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { PassageItem, Source } from '@/api/types'
import { citationFromHref, numberCitations } from '@/lib/citations'
import { CitationChip } from './CitationChip'
import { citationTarget, findSentence } from './passageTarget'

interface Props {
  answer: string
  sources: Source[]
  /** Τα χωρία, ΜΟΝΟ αν έχει τρέξει ο έλεγχος· αλλιώς η κάρτα δεν δείχνει χωρίο. */
  passages?: PassageItem[]
  /** Πρόθεμα για τα κλειδιά των παραπομπών αυτής της απάντησης (π.χ. το id του μηνύματος). */
  citeKey: string
  activeKey?: string | null
  onCitation: (label: string, context: string, activeKey: string) => void
}

/**
 * Η απάντηση ως markdown, με τις παραπομπές [S#] ως κουμπιά.
 *
 * ΑΣΦΑΛΕΙΑ: το react-markdown ΔΕΝ αποδίδει HTML (χωρίς rehype-raw). Ένα PDF με κρυμμένη επίθεση δεν μπορεί να βάλει
 * script ή εικόνα μέσω της απάντησης: ό,τι μοιάζει με HTML εμφανίζεται ως κείμενο. Και οι εικόνες markdown δεν
 * φορτώνονται (θα μπορούσαν να στείλουν δεδομένα σε τρίτο server).
 */
export function Answer({ answer, sources, passages, citeKey, activeKey, onCitation }: Props) {
  const { markdown, order, contexts } = useMemo(() => numberCitations(answer), [answer])
  const byLabel = useMemo(() => new Map(sources.map((s) => [s.label, s])), [sources])

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
            const target = passages ? citationTarget(findSentence(passages, ref.label, context), ref.label) : null
            const quote = target && target.label === ref.label ? target.quote : null
            return (
              <CitationChip
                number={order.indexOf(ref.label) + 1}
                source={byLabel.get(ref.label)}
                quote={quote}
                active={key === activeKey}
                onOpen={() => onCitation(ref.label, context, key)}
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
