import type { PassageItem, Source } from '@/api/types'
import { Button } from '@/components/ui/button'
import { useViewer } from '@/features/viewer/viewer'
import { strings } from '@/i18n/en'

const s = strings.app

/**
 * Το χωρίο κάθε πρότασης (βήμα 6). ΠΟΤΕ «επαληθευμένο»: ένα χωρίο δείχνει πού να κοιτάξεις, δεν αποδεικνύει ότι η
 * σελίδα στηρίζει κάθε μέρος της πρότασης.
 */
export function PassagesPanel({ items, sources }: { items: PassageItem[]; sources: Source[] }) {
  const { openPassage } = useViewer()
  const byLabel = new Map(sources.map((x) => [x.label, x]))
  return (
    <section className="space-y-3 rounded-lg border bg-muted/20 p-3">
      <div>
        <h3 className="text-sm font-medium">{s.passagesTitle(items.length)}</h3>
        <p className="text-xs text-muted-foreground">{s.passagesIntro}</p>
      </div>
      <ol className="space-y-3">
        {items.map((it, n) => {
          const found = (it.status === 'found' || it.status === 'found_elsewhere') && it.quote
          const cited = [...new Set(it.labels)].map((l) => byLabel.get(l)).filter((x): x is Source => Boolean(x))
          return (
            <li key={it.i} className="space-y-1.5 text-sm">
              <p>
                <span className="mr-1 font-medium tabular-nums">{n + 1}.</span>
                {it.sentence}
              </p>
              {found ? (
                <div className="space-y-1.5 pl-4">
                  <p className="text-xs text-muted-foreground">
                    {it.file} · {s.page.toLowerCase()} {it.page ?? '?'}
                  </p>
                  <blockquote className="border-l-2 border-highlight pl-3 text-muted-foreground italic">“{it.quote}”</blockquote>
                  {it.status === 'found_elsewhere' && cited.length > 0 && (
                    <p className="text-xs text-muted-foreground">
                      {`This sentence cites ${cited.map((c) => `${c.file} p. ${c.page ?? '?'}`).join(', ')}; the passage was found on ${it.file} p. ${it.page ?? '?'}.`}
                    </p>
                  )}
                  <Button variant="outline" size="xs" onClick={() => openPassage(it, sources)}>
                    {s.seePassage(it.page)}
                  </Button>
                </div>
              ) : (
                <div className="space-y-1.5 pl-4">
                  {it.status !== 'found' && it.status !== 'found_elsewhere' && (
                    <p className="text-xs text-muted-foreground">{s.passageNote[it.status]}</p>
                  )}
                  {it.status === 'not_found' && cited.length > 0 && (
                    <Button variant="outline" size="xs" onClick={() => openPassage(it, sources)}>
                      {s.openCitedPage(cited[0]?.page)}
                    </Button>
                  )}
                </div>
              )}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
