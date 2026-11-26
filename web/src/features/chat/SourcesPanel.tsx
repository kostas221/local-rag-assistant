import { FileText } from 'lucide-react'
import type { Source } from '@/api/types'
import { Button } from '@/components/ui/button'
import { useViewer } from '@/features/viewer/viewer'
import { strings } from '@/i18n/en'
import { citedLabels, numberCitations } from '@/lib/citations'

const s = strings.app

function SourceRow({ source, number }: { source: Source; number: number | null }) {
  const { openSource } = useViewer()
  const hasPage = source.doc_id !== null && source.doc_id !== undefined && source.doc_id !== -1 && Boolean(source.page)
  return (
    <li className="space-y-1.5 rounded-lg border bg-card/50 p-2.5">
      <div className="flex items-start gap-2">
        {number !== null ? (
          <span className="mt-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-muted px-1 text-[0.7rem] font-semibold tabular-nums">
            {number}
          </span>
        ) : (
          <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium" title={source.file}>
            {source.file}
          </p>
          <p className="text-xs text-muted-foreground">
            {s.page} {source.page ?? '?'}
          </p>
        </div>
        {hasPage && (
          <Button variant="outline" size="xs" onClick={() => openSource(source)}>
            {s.openPage}
          </Button>
        )}
      </div>
      {source.preview && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer select-none">{s.searchMatch}</summary>
          <p className="mt-1 border-l-2 pl-2">{source.preview}</p>
        </details>
      )}
    </li>
  )
}

/**
 * Οι σελίδες της απάντησης. Πρώτα όσες ΠΑΡΕΘΕΣΕ η απάντηση, με τον αριθμό που έχουν στο κείμενο· μετά όσες δόθηκαν
 * στο μοντέλο χωρίς να παρατεθούν (ήταν εύρημα της αξιολόγησης 31/10: «δόθηκαν» ≠ «παρατέθηκαν»).
 */
export function SourcesPanel({ answer, sources }: { answer: string; sources: Source[] }) {
  const cited = citedLabels(answer)
  const { order } = numberCitations(answer)
  const full = sources.some((x) => x.full_text)
  const used = sources
    .filter((x) => x.label && cited.has(x.label))
    .sort((a, b) => order.indexOf(a.label ?? '') - order.indexOf(b.label ?? ''))
  const rest = sources.filter((x) => !x.label || !cited.has(x.label))
  return (
    <div className="space-y-3">
      {used.length > 0 && (
        <section className="space-y-1.5">
          <h3 className="text-xs font-medium text-muted-foreground">
            {s.citedInAnswer} ({used.length})
          </h3>
          <ul className="space-y-1.5">
            {used.map((x) => (
              <SourceRow key={`${x.label}-${x.file}-${x.page}`} source={x} number={order.indexOf(x.label ?? '') + 1} />
            ))}
          </ul>
        </section>
      )}
      {rest.length > 0 && (
        <section className="space-y-1.5">
          <h3 className="text-xs font-medium text-muted-foreground">
            {used.length ? s.otherPages(full) : s.pagesNoneCited(full)} ({rest.length})
          </h3>
          <ul className="space-y-1.5">
            {rest.map((x, i) => (
              <SourceRow key={`${x.label ?? i}-${x.file}-${x.page}`} source={x} number={null} />
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
