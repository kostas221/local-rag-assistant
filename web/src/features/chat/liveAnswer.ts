/**
 * Η απάντηση που γράφεται τώρα, ως καθαρή συνάρτηση γεγονότων (δοκιμάζεται χωρίς browser και χωρίς backend).
 */
import type { Metrics, Source, StreamEvent } from '@/api/types'

/** searching: πριν από τις πηγές· composing: πηγές ήρθαν, κείμενο όχι ακόμα· writing: έρχεται κείμενο. */
export type LivePhase = 'searching' | 'composing' | 'writing' | 'done' | 'stopped' | 'error'

export interface LiveAnswer {
  convId: number | null
  question: string
  text: string
  /** null μέχρι να έρθουν· [] = ο φύλακας δεν έδωσε πηγές (άρνηση ή διευκρίνιση). */
  sources: Source[] | null
  metrics: Metrics | null
  /** Λόγος διακοπής από το μοντέλο (π.χ. MAX_TOKENS), όταν ο backend στέλνει status «incomplete». */
  incomplete: string | null
  phase: LivePhase
  error: string | null
  /** Τα μηνύματα της συνομιλίας μέχρι αυτό το id υπήρχαν πριν από την ερώτηση. */
  previousMaxId: number
  startedAt: number
  totalS: number | null
}

export function startLive(question: string, convId: number | null, previousMaxId: number, now: number): LiveAnswer {
  return {
    convId,
    question,
    text: '',
    sources: null,
    metrics: null,
    incomplete: null,
    phase: 'searching',
    error: null,
    previousMaxId,
    startedAt: now,
    totalS: null,
  }
}

export function applyEvent(live: LiveAnswer, event: StreamEvent): LiveAnswer {
  switch (event.type) {
    case 'sources':
      return { ...live, sources: event.data, phase: live.text ? live.phase : 'composing' }
    case 'text':
      return { ...live, text: live.text + event.data, phase: 'writing' }
    case 'metrics':
      return { ...live, metrics: event.data }
    case 'status':
      return event.data === 'incomplete' ? { ...live, incomplete: event.reason ?? 'unknown' } : live
  }
}

export function finishLive(live: LiveAnswer, phase: 'done' | 'stopped' | 'error', now: number, error?: string): LiveAnswer {
  return { ...live, phase, error: error ?? null, totalS: (now - live.startedAt) / 1000 }
}

export function isRunning(live: LiveAnswer | null): boolean {
  return live !== null && (live.phase === 'searching' || live.phase === 'composing' || live.phase === 'writing')
}

/** Τίτλος νέας συνομιλίας από την ερώτηση (όπως το Streamlit, αλλά «…» μόνο όταν όντως κόβεται). */
export function titleFrom(question: string, max = 40): string {
  const q = question.trim().replace(/\s+/g, ' ')
  return q.length > max ? `${q.slice(0, max).trimEnd()}…` : q
}
