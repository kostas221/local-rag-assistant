import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight, Loader2, X, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useState } from 'react'
import { ApiError } from '@/api/client'
import { getPageImage } from '@/api/endpoints'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useLibrary } from '@/features/library/library'
import { strings } from '@/i18n/en'
import { useViewer, type ViewerState } from './viewer'

const s = strings.app

/** Η εικόνα της σελίδας ως object URL, που ελευθερώνεται όταν αλλάξει ή κλείσει το πάνελ. */
function useObjectUrl(blob: Blob | undefined): string | null {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!blob) return
    const u = URL.createObjectURL(blob)
    // Σκόπιμα σε effect: το URL είναι πόρος του browser που πρέπει να ελευθερωθεί. Με useMemo, το StrictMode θα το
    // ελευθέρωνε στο πρώτο cleanup και η εικόνα θα έσπαγε.
    // oxlint-disable-next-line react/set-state-in-effect
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [blob])
  return url
}

function QuoteBlock({ quote }: { quote: string }) {
  return <blockquote className="border-l-2 border-highlight pl-3 text-sm text-muted-foreground italic">“{quote}”</blockquote>
}

/**
 * Το πάνελ της πηγής: η ΠΡΑΓΜΑΤΙΚΗ σελίδα του άρθρου (page_view.py), δίπλα στην απάντηση. Ολόκληρη η σελίδα, με
 * κίτρινο μόνο όταν το σημείο του χωρίου είναι μοναδικό. Κλείνει με Esc ή με ×.
 */
export function SourceViewer({ state }: { state: ViewerState }) {
  const { setPage, close, openSource } = useViewer()
  const { docs } = useLibrary()
  const [zoom, setZoom] = useState(1)
  const { source, page, finding, note, alternate } = state
  const docId = source.doc_id ?? null
  const citedPage = source.page ?? null
  const quote = page === citedPage ? state.quote : null
  const pagesTotal = docs.find((d) => d.id === docId)?.pages_total ?? null

  const image = useQuery({
    queryKey: ['page-image', docId, page, quote],
    queryFn: ({ signal }) => getPageImage(docId ?? -1, page, quote, signal),
    enabled: docId !== null && docId !== -1,
    placeholderData: keepPreviousData, // η παλιά εικόνα μένει μέχρι να έρθει η νέα (π.χ. με το κίτρινο)
    staleTime: Infinity,
    gcTime: 2 * 60_000,
    retry: false,
  })
  const url = useObjectUrl(image.data?.blob)
  const highlight = image.isPlaceholderData ? null : image.data?.highlight
  const gone = image.error instanceof ApiError && image.error.status === 404
  const noView = docId === null || docId === -1

  return (
    <section aria-label={s.sourceViewer} className="flex h-full min-h-0 flex-col bg-card">
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium" title={source.file}>
            {source.file}
          </p>
          <p className="text-xs text-muted-foreground">
            {s.page} {page}
            {pagesTotal ? ` / ${pagesTotal}` : ''}
            {citedPage !== null && citedPage !== page && ` · ${s.citedPageNote(citedPage)}`}
          </p>
        </div>
        <Button variant="ghost" size="icon-sm" aria-label={s.zoomOut} onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}>
          <ZoomOut />
        </Button>
        <Button variant="ghost" size="icon-sm" aria-label={s.zoomIn} onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}>
          <ZoomIn />
        </Button>
        <Button variant="ghost" size="icon-sm" aria-label={s.closeViewer} onClick={close}>
          <X />
        </Button>
      </header>

      {(note || alternate) && page === citedPage && (
        <div className="space-y-2 border-b bg-muted/40 px-3 py-2 text-xs">
          {note && <p className="text-muted-foreground">{s.viewerNote[note]}</p>}
          {alternate && (
            <Button
              variant="outline"
              size="xs"
              onClick={() => openSource(alternate.source, { quote: alternate.quote, activeKey: state.activeKey })}
            >
              {alternate.quote ? s.seePassage(alternate.source.page) : s.openCitedPage(alternate.source.page)}
            </Button>
          )}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto">
        <div className="space-y-3 p-4">
          <div style={{ width: `${zoom * 100}%` }} className="mx-auto transition-[width]">
            {noView ? (
              <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">{s.noPageView}</p>
            ) : image.isError ? (
              <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                {gone ? s.pageGone : image.error instanceof ApiError && image.error.status === 429 ? strings.errors.rateLimited : s.pageFailed}
              </p>
            ) : url ? (
              <img
                src={url}
                alt={`${source.file}, ${s.page.toLowerCase()} ${page}`}
                className="w-full rounded-md bg-white shadow-sm ring-1 ring-black/10"
              />
            ) : (
              <Skeleton className="aspect-[3/4] w-full" aria-label={s.loadingPage} />
            )}
          </div>
          {quote && highlight && highlight !== 'highlighted' && <QuoteBlock quote={quote} />}
        </div>
      </div>

      <footer className="flex items-center justify-between gap-2 border-t px-3 py-2">
        <Button
          variant="ghost"
          size="sm"
          disabled={page <= 1}
          onClick={() => setPage(page - 1)}
          aria-label={s.prevPage}
        >
          <ChevronLeft /> {page > 1 ? page - 1 : ''}
        </Button>
        <p className="flex min-w-0 items-center gap-1.5 text-center text-xs text-muted-foreground" aria-live="polite">
          {finding && page === citedPage && (
            <>
              <Loader2 className="size-3.5 shrink-0 animate-spin" /> {s.findingPassage}
            </>
          )}
          {!finding && quote && image.isFetching && <Loader2 className="size-3.5 shrink-0 animate-spin" />}
          {!finding && quote && highlight && s.highlight[highlight === 'none' ? 'not_found' : highlight]}
        </p>
        <Button
          variant="ghost"
          size="sm"
          disabled={pagesTotal !== null && page >= pagesTotal}
          onClick={() => setPage(page + 1)}
          aria-label={s.nextPage}
        >
          {pagesTotal === null || page < pagesTotal ? page + 1 : ''} <ChevronRight />
        </Button>
      </footer>
    </section>
  )
}
