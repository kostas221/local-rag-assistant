import { CornerDownRight, FileText } from 'lucide-react'
import type { PassageItem, Source } from '@/api/types'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { strings } from '@/i18n/en'
import { cn } from '@/lib/utils'

/** Η κατάσταση της πρότασης της παραπομπής, στην προβολή τεκμηρίων· «unsure» όταν δεν ξέρουμε με σιγουριά ποια είναι. */
export type EvidenceStatus = PassageItem['status'] | 'unsure'

interface Props {
  number: number
  source: Source | undefined
  /** Το χωρίο της πρότασης, ΜΟΝΟ αν έχει τρέξει ο έλεγχος χωρίων. */
  quote: string | null
  active: boolean
  /** Μόνο στην προβολή τεκμηρίων. */
  evidence?: EvidenceStatus | null
  onOpen: () => void
}

// Τα χρώματα λένε ΠΟΥ βρέθηκε χωρίο, όχι ότι η πρόταση είναι σωστή: πουθενά «verified».
const EVIDENCE_STYLE: Record<EvidenceStatus, string> = {
  found: 'border-highlight bg-highlight/35 text-foreground',
  found_elsewhere: 'border-amber-500 bg-amber-500/15 text-foreground',
  not_found: 'border-dashed border-muted-foreground/70 bg-transparent text-muted-foreground',
  absence: 'border-dotted border-muted-foreground/70 bg-transparent text-muted-foreground',
  no_citation: '',
  unsure: '',
}

/**
 * Παραπομπή μέσα στο κείμενο: με το ποντίκι από πάνω δείχνει αρχείο, σελίδα και το χωρίο· με κλικ ανοίγει την πηγή.
 *
 * ΕΙΛΙΚΡΙΝΕΙΑ: αν δεν έχει βρεθεί το χωρίο, ΔΕΝ δείχνουμε το απόσπασμα της αναζήτησης ως χωρίο. Εκείνο είναι το
 * κομμάτι που έφερε τη σελίδα, όχι αυτό που στηρίζει την πρόταση, και θα έμοιαζε απόδειξη.
 */
export function CitationChip({ number, source, quote, active, evidence, onOpen }: Props) {
  const label = source ? `${source.file}, ${strings.app.page.toLowerCase()} ${source.page ?? '?'}` : `${number}`
  const evidenceText = evidence ? strings.app.evidence.status[evidence] : null
  return (
    <HoverCard openDelay={150} closeDelay={80}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          onClick={onOpen}
          aria-label={`Source ${number}: ${label}${evidenceText ? `. ${evidenceText}` : ''}`}
          className={cn(
            'mx-0.5 inline-flex h-5 min-w-5 items-center justify-center gap-0.5 rounded-md border px-1 align-[0.1em]',
            'text-[0.7rem] font-semibold tabular-nums transition-colors',
            active
              ? 'border-brand bg-brand text-brand-foreground'
              : evidence && EVIDENCE_STYLE[evidence]
                ? EVIDENCE_STYLE[evidence]
                : 'border-border bg-muted text-muted-foreground hover:border-brand hover:text-foreground',
          )}
        >
          {number}
          {evidence === 'found_elsewhere' && !active && <CornerDownRight className="size-2.5" />}
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-80 space-y-2 text-sm" side="top" align="start">
        <div className="flex items-start gap-2">
          <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          <div className="min-w-0">
            <p className="truncate font-medium">{source?.file ?? '—'}</p>
            <p className="text-xs text-muted-foreground">
              {strings.app.page} {source?.page ?? '?'}
            </p>
          </div>
        </div>
        {evidenceText && <p className="text-xs font-medium">{evidenceText}</p>}
        {quote ? (
          <>
            <blockquote className="border-l-2 border-highlight pl-3 text-muted-foreground italic">“{quote}”</blockquote>
            <p className="text-xs text-muted-foreground">{`Click to ${strings.app.openPage.toLowerCase()}.`}</p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">{strings.app.hoverNoPassage}</p>
        )}
      </HoverCardContent>
    </HoverCard>
  )
}
