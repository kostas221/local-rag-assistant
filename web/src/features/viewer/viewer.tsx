/**
 * Ποια σελίδα δείχνει το πάνελ της πηγής, και με ποιο χωρίο.
 *
 * Κλικ σε παραπομπή: η σελίδα ανοίγει ΑΜΕΣΩΣ, και το χωρίο ψάχνεται μόνο του (μία κλήση ανά απάντηση, που σώζεται
 * στο μήνυμα). Το κίτρινο έρχεται όταν βρεθεί. Απόφαση μετά το πρωτότυπο (26/11/2026): κανένα κουμπί δεν χρειάζεται.
 */
import { useQueryClient } from '@tanstack/react-query'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { Message, PassageItem, Source } from '@/api/types'
import { ensurePassages } from '@/features/chat/passages'
import {
  citationTarget,
  findSentence,
  passageTarget,
  type CitationTarget,
  type ViewerNoteKind,
} from '@/features/chat/passageTarget'

export interface ViewerAlternate {
  source: Source
  quote: string | null
}

export interface ViewerState {
  source: Source
  /** Η σελίδα που φαίνεται τώρα (ο χρήστης μπορεί να πάει μπρος-πίσω). */
  page: number
  /** Το χωρίο, για κίτρινο ΜΟΝΟ πάνω στη σελίδα της πηγής. */
  quote: string | null
  finding: boolean
  note: ViewerNoteKind | null
  alternate: ViewerAlternate | null
  /** Ποιο κουμπί παραπομπής άνοιξε το πάνελ (για να φαίνεται ενεργό). */
  activeKey: string | null
}

export interface CitationClick {
  message: Pick<Message, 'passages'> & { id: number | null }
  sources: Source[]
  label: string
  /** Το κείμενο της γραμμής πριν από την παραπομπή (βλ. lib/citations). */
  context: string
  activeKey: string
  /** Όλες οι παραπομπές της ίδιας απάντησης, με τη σειρά: με `[` / `]` το πάνελ πάει στην προηγούμενη / επόμενη. */
  siblings?: CitationClick[]
}

/** Τα πλήκτρα του πάνελ δεν ενεργούν όσο ο χρήστης γράφει. */
function typing(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  return Boolean(el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)))
}

interface ViewerApi {
  state: ViewerState | null
  openSource: (source: Source, extra?: Partial<Omit<ViewerState, 'source' | 'page'>>) => void
  openCitation: (click: CitationClick) => void
  openPassage: (item: PassageItem, sources: Source[]) => void
  setPage: (page: number) => void
  close: () => void
}

const ViewerContext = createContext<ViewerApi | null>(null)

function stateFor(source: Source, extra?: Partial<Omit<ViewerState, 'source' | 'page'>>): ViewerState {
  return {
    source,
    page: source.page ?? 1,
    quote: null,
    finding: false,
    note: null,
    alternate: null,
    activeKey: null,
    ...extra,
  }
}

function apply(target: CitationTarget, sources: Source[], fallback: Source, activeKey: string | null): ViewerState {
  const source = sources.find((s) => s.label === target.label) ?? fallback
  const alt = target.alternate ? sources.find((s) => s.label === target.alternate?.label) : undefined
  return stateFor(source, {
    quote: source === fallback && target.label !== fallback.label ? null : target.quote,
    note: target.note,
    alternate: alt ? { source: alt, quote: target.alternate?.quote ?? null } : null,
    activeKey,
  })
}

export function ViewerProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const [state, setState] = useState<ViewerState | null>(null)
  // Κάθε άνοιγμα παίρνει αριθμό: αποτέλεσμα που αργεί για παλιό κλικ δεν αλλάζει το πάνελ ενός νεότερου.
  const ticket = useRef(0)
  const siblings = useRef<CitationClick[]>([])

  const openSource = useCallback<ViewerApi['openSource']>((source, extra) => {
    ticket.current++
    setState(stateFor(source, extra))
  }, [])

  const openCitation = useCallback(
    (click: CitationClick) => {
      const { message, sources, label, context, activeKey } = click
      const source = sources.find((s) => s.label === label)
      if (!source) return
      if (click.siblings) siblings.current = click.siblings
      const mine = ++ticket.current
      // Απάντηση που γράφεται ακόμα: δεν έχει id, άρα ούτε χωρία. Ανοίγει μόνο η σελίδα.
      if (message.id === null) {
        setState(stateFor(source, { activeKey }))
        return
      }
      setState(stateFor(source, { finding: true, activeKey }))
      ensurePassages(qc, message.id, message.passages)
        .then((passages) => {
          if (mine !== ticket.current) return
          if (passages.status !== 'done') {
            setState(stateFor(source, { note: 'no_pages', activeKey }))
            return
          }
          const item = findSentence(passages.items, label, context)
          setState(apply(citationTarget(item, label), sources, source, activeKey))
        })
        .catch(() => {
          if (mine === ticket.current) setState(stateFor(source, { note: 'check_failed', activeKey }))
        })
    },
    [qc],
  )

  const openPassage = useCallback((item: PassageItem, sources: Source[]) => {
    const target = passageTarget(item)
    const first = target && sources.find((s) => s.label === target.label)
    if (!target || !first) return
    ticket.current++
    setState(apply(target, sources, first, null))
  }, [])

  // `[` / `]`: προηγούμενη / επόμενη παραπομπή της απάντησης, με το χωρίο της (όχι απλώς άλλη πηγή).
  const activeKey = state?.activeKey ?? null
  useEffect(() => {
    if (!activeKey) return
    const onKey = (e: KeyboardEvent) => {
      if ((e.key !== '[' && e.key !== ']') || e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return
      const list = siblings.current
      const i = list.findIndex((c) => c.activeKey === activeKey)
      const next = i < 0 ? undefined : list[e.key === ']' ? i + 1 : i - 1]
      if (!next) return
      e.preventDefault()
      openCitation({ ...next, siblings: list })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeKey, openCitation])

  const setPage = useCallback((page: number) => setState((s) => (s ? { ...s, page } : s)), [])
  const close = useCallback(() => {
    ticket.current++
    setState(null)
  }, [])

  const value = useMemo(
    () => ({ state, openSource, openCitation, openPassage, setPage, close }),
    [state, openSource, openCitation, openPassage, setPage, close],
  )
  return <ViewerContext.Provider value={value}>{children}</ViewerContext.Provider>
}

export function useViewer(): ViewerApi {
  const ctx = useContext(ViewerContext)
  if (!ctx) throw new Error('useViewer must be used inside <ViewerProvider>')
  return ctx
}
