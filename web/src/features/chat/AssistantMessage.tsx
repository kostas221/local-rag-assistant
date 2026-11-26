import { useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, BookOpen, Info, Loader2, Search, Square } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import type { Metrics, Passages, Source } from '@/api/types'
import { ZLogo } from '@/components/brand'
import { Button } from '@/components/ui/button'
import { useViewer } from '@/features/viewer/viewer'
import { strings } from '@/i18n/en'
import { citedLabels } from '@/lib/citations'
import { Answer } from './Answer'
import type { MessageMeta } from './chat'
import { Feedback } from './Feedback'
import type { LivePhase } from './liveAnswer'
import { ensurePassages, usePassages } from './passages'
import { PassagesPanel } from './PassagesPanel'
import { RefusalCard } from './RefusalCard'
import { sourceNotes } from './sourceNotes'
import { SourcesPanel } from './SourcesPanel'

const s = strings.app

export function AssistantShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 shrink-0">
        <ZLogo size={22} />
      </div>
      <div className="min-w-0 flex-1 space-y-3 text-[0.95rem]">{children}</div>
    </div>
  )
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(performance.now()), 100)
    return () => window.clearInterval(t)
  }, [])
  return <span className="tabular-nums">{((now - since) / 1000).toFixed(1)} s</span>
}

function Details({ sources, metrics, totalS }: { sources: Source[]; metrics: Metrics | null; totalS: number | null }) {
  const route = sources.some((x) => x.full_text) ? s.routeFull : sources.some((x) => x.second_search) ? s.routeSecond : s.routeSearch
  const cell = (label: string, value: ReactNode) => (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="tabular-nums">{value}</dd>
    </div>
  )
  return (
    <div className="space-y-2 rounded-lg border bg-muted/30 p-3 text-xs">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
        {cell(s.route, route)}
        {cell(s.pagesGiven, metrics?.pages ?? sources.length)}
        {metrics && cell(s.retrieval, `${(metrics.retrieval_s ?? 0).toFixed(2)} s`)}
        {metrics && cell(s.generation, `${(metrics.generation_s ?? 0).toFixed(2)} s`)}
        {totalS !== null && cell(s.total, `${totalS.toFixed(1)} s`)}
        {metrics?.prompt_tokens ? cell(s.tokens, `${metrics.prompt_tokens} → ${metrics.completion_tokens ?? '?'}`) : null}
      </dl>
      {!metrics && <p className="text-muted-foreground">{s.timingsSessionOnly}</p>}
    </div>
  )
}

interface Props {
  content: string
  /** null: δεν έχουν έρθει ακόμα (απάντηση που γράφεται). */
  sources: Source[] | null
  /** null: απάντηση που γράφεται, χωρίς id ακόμα. */
  messageId: number | null
  savedPassages: Passages | null
  meta?: MessageMeta
  /** Μόνο για την απάντηση που γράφεται τώρα. */
  live?: { phase: LivePhase; startedAt: number; incomplete: string | null }
}

export function AssistantMessage({ content, sources, messageId, savedPassages, meta, live }: Props) {
  const qc = useQueryClient()
  const viewer = useViewer()
  const [panel, setPanel] = useState<'sources' | 'passages' | 'details' | null>(null)
  const passages = usePassages(messageId === null ? null : { id: messageId, passages: savedPassages })
  const writing = Boolean(live) && live?.phase !== 'done'
  const list = sources ?? []
  const items = passages.data?.status === 'done' ? passages.data.items : undefined
  const citeKey = messageId === null ? 'live' : String(messageId)

  if (live && (live.phase === 'searching' || live.phase === 'composing') && !content)
    return (
      <AssistantShell>
        <p className="flex items-center gap-2 text-sm text-muted-foreground italic">
          <Loader2 className="size-4 animate-spin" />
          {live.phase === 'searching' ? s.searching : s.composing}
          <Elapsed since={live.startedAt} />
        </p>
      </AssistantShell>
    )

  if (sources !== null && sources.length === 0)
    return (
      <AssistantShell>
        <RefusalCard text={content} />
      </AssistantShell>
    )

  const hasCitations = citedLabels(content).size > 0
  const findPassages = () => {
    setPanel('passages')
    if (messageId !== null) void ensurePassages(qc, messageId, savedPassages).catch(() => undefined)
  }
  const notes = writing ? [] : sourceNotes(content, list)
  const incomplete = live?.incomplete ?? meta?.incomplete ?? null

  return (
    <AssistantShell>
      <Answer
        answer={content}
        sources={list}
        passages={items}
        citeKey={citeKey}
        activeKey={viewer.state?.activeKey}
        onCitation={(label, context, activeKey) =>
          viewer.openCitation({
            message: { id: messageId, passages: savedPassages },
            sources: list,
            label,
            context,
            activeKey,
          })
        }
      />
      {writing && <span className="inline-block h-4 w-1.5 animate-pulse rounded-sm bg-brand align-middle" />}

      {(meta?.stopped || incomplete) && (
        <p className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-400">
          {meta?.stopped ? <Square className="mt-0.5 size-3 shrink-0" /> : <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />}
          {meta?.stopped ? s.stopped : s.incomplete(incomplete ?? '')}
        </p>
      )}
      {notes.map((n) => (
        <p key={n} className="text-xs text-muted-foreground">
          {n}
        </p>
      ))}

      {!writing && list.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant={panel === 'sources' ? 'secondary' : 'outline'} size="sm" onClick={() => setPanel(panel === 'sources' ? null : 'sources')}>
            <BookOpen /> {s.sources} · {list.length}
          </Button>
          {messageId !== null && hasCitations && (
            <Button
              variant={panel === 'passages' ? 'secondary' : 'outline'}
              size="sm"
              title={s.passageCost}
              onClick={() => (panel === 'passages' ? setPanel(null) : findPassages())}
            >
              {passages.isFetching ? <Loader2 className="animate-spin" /> : <Search />} {s.findPassages}
            </Button>
          )}
          <Button variant={panel === 'details' ? 'secondary' : 'outline'} size="sm" onClick={() => setPanel(panel === 'details' ? null : 'details')}>
            <Info /> {s.details}
          </Button>
          {messageId !== null && (
            <div className="ml-auto">
              <Feedback messageId={messageId} />
            </div>
          )}
        </div>
      )}

      {panel === 'sources' && <SourcesPanel answer={content} sources={list} />}
      {panel === 'details' && <Details sources={list} metrics={meta?.metrics ?? null} totalS={meta?.totalS ?? null} />}
      {panel === 'passages' && (
        <>
          {passages.isFetching && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" /> {s.passagesFinding}
            </p>
          )}
          {passages.isError && !passages.isFetching && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              {s.passagesFailed}
              <Button variant="outline" size="xs" onClick={findPassages}>
                {s.tryAgain}
              </Button>
            </p>
          )}
          {passages.data?.status === 'no_pages' && <p className="text-xs text-muted-foreground">{s.passagesNoPages}</p>}
          {items && <PassagesPanel items={items} sources={list} />}
        </>
      )}
    </AssistantShell>
  )
}
