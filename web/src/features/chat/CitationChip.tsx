import { FileText } from 'lucide-react'
import type { Source } from '@/api/types'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { strings } from '@/i18n/en'
import { cn } from '@/lib/utils'

interface Props {
  number: number
  source: Source | undefined
  /** Το χωρίο της πρότασης, ΜΟΝΟ αν έχει τρέξει ο έλεγχος χωρίων. */
  quote: string | null
  active: boolean
  onOpen: () => void
}

/**
 * Παραπομπή μέσα στο κείμενο: με το ποντίκι από πάνω δείχνει αρχείο, σελίδα και το χωρίο· με κλικ ανοίγει την πηγή.
 *
 * ΕΙΛΙΚΡΙΝΕΙΑ: αν δεν έχει βρεθεί το χωρίο, ΔΕΝ δείχνουμε το απόσπασμα της αναζήτησης ως χωρίο. Εκείνο είναι το
 * κομμάτι που έφερε τη σελίδα, όχι αυτό που στηρίζει την πρόταση, και θα έμοιαζε απόδειξη.
 */
export function CitationChip({ number, source, quote, active, onOpen }: Props) {
  const label = source ? `${source.file}, ${strings.app.page.toLowerCase()} ${source.page ?? '?'}` : `${number}`
  return (
    <HoverCard openDelay={150} closeDelay={80}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          onClick={onOpen}
          aria-label={`Source ${number}: ${label}`}
          className={cn(
            'mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md border px-1 align-[0.1em]',
            'text-[0.7rem] font-semibold tabular-nums transition-colors',
            active
              ? 'border-brand bg-brand text-brand-foreground'
              : 'border-border bg-muted text-muted-foreground hover:border-brand hover:text-foreground',
          )}
        >
          {number}
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
