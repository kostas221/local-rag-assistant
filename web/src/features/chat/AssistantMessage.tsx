import { useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, BookOpen, Copy, Eye, EyeOff, GitCompareArrows, Info, Loader2, Search, Square } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import type { Metrics, PassageItem, Passages, Source } from '@/api/types'
import { ZLogo } from '@/components/brand'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { Button } from '@/components/ui/button'
import { useLibrary } from '@/features/library/library'
import { useViewer } from '@/features/viewer/viewer'
import { strings } from '@/i18n/en'
import { citedLabels } from '@/lib/citations'
import { Answer, type CiteTarget } from './Answer'
import { useChat, type MessageMeta } from './chat'
import { Feedback } from './Feedback'
import type { LivePhase } from './liveAnswer'
import { ensurePassages, usePassages } from './passages'
import { PassagesPanel } from './PassagesPanel'
import { RefusalCard } from './RefusalCard'
import { answerForCopy, compareCandidates, evidenceCounts, sourceNotes } from './sourceNotes'
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

/** Η σύνοψη της προβολής τεκμηρίων. Λέει ΠΟΥ βρέθηκαν χωρία, ποτέ ότι η απάντηση «επαληθεύτηκε». */
function EvidenceSummary({ items }: { items: PassageItem[] }) {
  const e = s.evidence
  const c = evidenceCounts(items)
  const parts = [e.summary(c.found, c.cited)]
  if (c.elsewhere) parts.push(e.elsewhere(c.elsewhere))
  if (c.missing) parts.push(e.missing(c.missing))
  if (c.absence) parts.push(e.absence(c.absence))
  if (c.uncited) parts.push(e.uncited(c.uncited))
  const swatch = (cls: string, text: string) => (
    <span className="inline-flex items-center gap-1">
      <span className={`inline-block size-3 rounded-sm border ${cls}`} />
      {text}
    </span>
  )
  return (
    <div className="space-y-1.5 rounded-lg border bg-muted/30 p-3 text-xs">
      <p className="font-medium">{parts.join(' · ')}</p>
      <p className="flex flex-wrap gap-x-4 gap-y-1 text-muted-foreground">
        {swatch('border-highlight bg-highlight/35', e.legendFound)}
        {swatch('border-amber-500 bg-amber-500/15', e.legendElsewhere)}
        {swatch('border-dashed border-muted-foreground/70', e.legendMissing)}
      </p>
      <p className="text-muted-foreground">{e.caveat}</p>
    </div>
  )
}

function PassagesStatus({ fetching, failed, onRetry }: { fetching: boolean; failed: boolean; onRetry: () => void }) {
  if (fetching)
    return (
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" /> {s.passagesFinding}
      </p>
    )
  if (failed)
    return (
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        {s.passagesFailed}
        <Button variant="outline" size="xs" onClick={onRetry}>
          {s.tryAgain}
        </Button>
      </p>
    )
  return null
}

interface Props {
  content: string
  /** null: δεν έχουν έρθει ακόμα (απάντηση που γράφεται). */
  sources: Source[] | null
  /** null: απάντηση που γράφεται, χωρίς id ακόμα. */
  messageId: number | null
  savedPassages: Passages | null
  meta?: MessageMeta
  /** Η ερώτηση αυτής της απάντησης και η συνομιλία της (για το «Compare only these»). */
  question?: string | null
  convId?: number | null
  /** Μόνο για την απάντηση που γράφεται τώρα. */
  live?: { phase: LivePhase; startedAt: number; incomplete: string | null }
}

export function AssistantMessage({ content, sources, messageId, savedPassages, meta, question, convId, live }: Props) {
  const qc = useQueryClient()
  const viewer = useViewer()
  const chat = useChat()
  const library = useLibrary()
  const [panel, setPanel] = useState<'sources' | 'passages' | 'details' | null>(null)
  const [evidence, setEvidence] = useState(false)
  const [confirmCompare, setConfirmCompare] = useState(false)
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

  const canCheck = messageId !== null && citedLabels(content).size > 0
  const loadPassages = () => {
    if (messageId !== null) void ensurePassages(qc, messageId, savedPassages).catch(() => undefined)
  }
  const findPassages = () => {
    setPanel('passages')
    loadPassages()
  }
  const toggleEvidence = () => {
    if (!evidence) loadPassages()
    setEvidence(!evidence)
  }
  const passagesFailed = passages.isError && !passages.isFetching
  const notes = writing ? [] : sourceNotes(list)
  const incomplete = live?.incomplete ?? meta?.incomplete ?? null
  // «Compare only these»: μόνο για άρθρα που υπάρχουν ακόμα και είναι έτοιμα, και μόνο αν ξέρουμε την ερώτηση.
  const compare = writing || !question ? [] : compareCandidates(content, list)
  const compareDocs = library.docs.filter((d) => d.status === 'ready' && compare.some((c) => c.id === d.id))

  const onCitation = (target: CiteTarget, all: CiteTarget[]) => {
    const toClick = (t: CiteTarget) => ({
      message: { id: messageId, passages: savedPassages },
      sources: list,
      label: t.label,
      context: t.context,
      activeKey: t.key,
    })
    viewer.openCitation({ ...toClick(target), siblings: all.map(toClick) })
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(answerForCopy(content, list))
      toast.success(s.copied)
    } catch {
      toast.error(s.copyFailed)
    }
  }

  return (
    <AssistantShell>
      {evidence && (
        <>
          <PassagesStatus fetching={passages.isFetching} failed={passagesFailed} onRetry={loadPassages} />
          {items && <EvidenceSummary items={items} />}
        </>
      )}
      <Answer
        answer={content}
        sources={list}
        passages={items}
        citeKey={citeKey}
        activeKey={viewer.state?.activeKey}
        evidence={evidence}
        onCitation={onCitation}
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
      {compareDocs.length >= 2 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-dashed p-2.5 text-xs text-muted-foreground">
          <span className="min-w-0 flex-1">{s.compareHint}</span>
          <Button variant="outline" size="xs" disabled={chat.busy} onClick={() => setConfirmCompare(true)}>
            <GitCompareArrows /> {s.compareOnly}
          </Button>
        </div>
      )}

      {!writing && list.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant={panel === 'sources' ? 'secondary' : 'outline'} size="sm" onClick={() => setPanel(panel === 'sources' ? null : 'sources')}>
            <BookOpen /> {s.sources} · {list.length}
          </Button>
          {canCheck && (
            <Button
              variant={panel === 'passages' ? 'secondary' : 'outline'}
              size="sm"
              title={s.passageCost}
              onClick={() => (panel === 'passages' ? setPanel(null) : findPassages())}
            >
              <Search /> {s.findPassages}
            </Button>
          )}
          {canCheck && (
            <Button variant={evidence ? 'secondary' : 'outline'} size="sm" title={s.evidence.cost} onClick={toggleEvidence}>
              {evidence ? <EyeOff /> : <Eye />} {evidence ? s.evidence.hide : s.evidence.show}
            </Button>
          )}
          <Button variant={panel === 'details' ? 'secondary' : 'outline'} size="sm" onClick={() => setPanel(panel === 'details' ? null : 'details')}>
            <Info /> {s.details}
          </Button>
          <div className="ml-auto flex items-center gap-0.5">
            <Button variant="ghost" size="icon-sm" aria-label={s.copy} title={s.copy} onClick={() => void copy()}>
              <Copy />
            </Button>
            {messageId !== null && <Feedback messageId={messageId} />}
          </div>
        </div>
      )}

      {panel === 'sources' && <SourcesPanel answer={content} sources={list} />}
      {panel === 'details' && <Details sources={list} metrics={meta?.metrics ?? null} totalS={meta?.totalS ?? null} />}
      {panel === 'passages' && (
        <>
          {!evidence && <PassagesStatus fetching={passages.isFetching} failed={passagesFailed} onRetry={findPassages} />}
          {passages.data?.status === 'no_pages' && <p className="text-xs text-muted-foreground">{s.passagesNoPages}</p>}
          {items && <PassagesPanel items={items} sources={list} />}
        </>
      )}

      <ConfirmDialog
        open={confirmCompare}
        onOpenChange={setConfirmCompare}
        title={s.compareTitle}
        body={s.compareBody(compareDocs.map((d) => d.file_name))}
        confirm={s.askAgain}
        destructive={false}
        onConfirm={() => {
          if (!question) return
          library.selectOnly(compareDocs.map((d) => d.id))
          chat.send(question, convId ?? null, compareDocs)
        }}
      />
    </AssistantShell>
  )
}
