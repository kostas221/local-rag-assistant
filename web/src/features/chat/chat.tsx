/**
 * Η αποστολή ερώτησης και η απάντηση που γράφεται, σε ένα σημείο πάνω από τις οθόνες: η απάντηση συνεχίζει και αν
 * ο χρήστης αλλάξει συνομιλία, και τα κλικ αλλού (βιβλιοθήκη, πηγές) δεν την κόβουν, σε αντίθεση με το Streamlit.
 *
 * Μετά το τέλος, το ιστορικό ξαναφορτώνεται από τον server, τη μόνη πηγή αλήθειας. Μόνο έτσι έρχεται και το `id` του
 * μηνύματος, που χρειάζονται τα χωρία και το 👍/👎: το streaming δεν το στέλνει.
 */
import { useQueryClient } from '@tanstack/react-query'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { ApiError } from '@/api/client'
import { createConversation, getMessages, streamChat } from '@/api/endpoints'
import { PERSONAS, type Message, type Metrics, type Persona } from '@/api/types'
import { useLibrary } from '@/features/library/library'
import { strings } from '@/i18n/en'
import { applyEvent, finishLive, isRunning, startLive, titleFrom, type LiveAnswer } from './liveAnswer'

/** Ό,τι ξέρουμε για μια απάντηση ΜΟΝΟ σε αυτή τη συνεδρία: ο backend δεν σώζει χρόνους ούτε το Stop. */
export interface MessageMeta {
  metrics: Metrics | null
  totalS: number | null
  stopped: boolean
  incomplete: string | null
}

export interface ChatFailure {
  convId: number | null
  question: string
  error: string
}

interface ChatState {
  live: LiveAnswer | null
  /** Από την αποστολή μέχρι να φορτωθεί το σωμένο μήνυμα: όσο ισχύει, δεν στέλνεται νέα ερώτηση. */
  busy: boolean
  /** Μόνο όσο έρχεται η απάντηση: τότε φαίνεται το Stop. */
  running: boolean
  meta: ReadonlyMap<number, MessageMeta>
  failure: ChatFailure | null
  persona: Persona
  setPersona: (p: Persona) => void
  send: (question: string, convId: number | null) => void
  stop: () => void
}

const PERSONA_KEY = 'zai.persona'

function loadPersona(): Persona {
  try {
    const v = localStorage.getItem(PERSONA_KEY)
    return (PERSONAS as readonly string[]).includes(v ?? '') ? (v as Persona) : 'Researcher'
  } catch {
    return 'Researcher'
  }
}

export const messagesKey = (convId: number) => ['messages', convId] as const

const maxId = (msgs: Message[] | undefined) => (msgs ?? []).reduce((m, x) => Math.max(m, x.id), 0)
const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms))

const ChatContext = createContext<ChatState | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const { selected } = useLibrary()
  const [live, setLive] = useState<LiveAnswer | null>(null)
  const [meta, setMeta] = useState<Map<number, MessageMeta>>(() => new Map())
  const [failure, setFailure] = useState<ChatFailure | null>(null)
  const [persona, setPersonaState] = useState<Persona>(loadPersona)
  const abortRef = useRef<AbortController | null>(null)
  const busyRef = useRef(false)
  const selectedRef = useRef(selected)
  const personaRef = useRef(persona)
  useEffect(() => {
    selectedRef.current = selected
    personaRef.current = persona
  }, [selected, persona])

  const setPersona = useCallback((p: Persona) => {
    setPersonaState(p)
    try {
      localStorage.setItem(PERSONA_KEY, p)
    } catch {
      /* απλώς δεν θυμόμαστε */
    }
  }, [])

  const fetchMessages = useCallback(
    (convId: number) => qc.fetchQuery({ queryKey: messagesKey(convId), queryFn: () => getMessages(convId) }),
    [qc],
  )

  /** Μετά το τέλος: φέρνει το σωμένο μήνυμα (και το id του). Μετά από Stop ο backend σώζει λίγο αργότερα. */
  const settle = useCallback(
    async (convId: number, final: LiveAnswer) => {
      const attempts = final.text ? 4 : 1
      for (let attempt = 0; attempt < attempts; attempt++) {
        const msgs = await fetchMessages(convId).catch(() => null)
        const saved = msgs?.filter((m) => m.role === 'assistant' && m.id > final.previousMaxId).at(-1)
        if (saved) {
          setMeta((prev) =>
            new Map(prev).set(saved.id, {
              metrics: final.metrics,
              totalS: final.totalS,
              stopped: final.phase !== 'done',
              incomplete: final.incomplete,
            }),
          )
          return
        }
        if (attempt + 1 < attempts) await sleep(400 * (attempt + 1))
      }
    },
    [fetchMessages],
  )

  const send = useCallback(
    (question: string, convId: number | null) => {
      const docs = selectedRef.current
      if (busyRef.current || !question.trim() || docs.length === 0) return
      busyRef.current = true
      const controller = new AbortController()
      abortRef.current = controller
      setFailure(null)
      let state = startLive(question.trim(), convId, 0, performance.now())
      const push = (next: LiveAnswer) => {
        state = next
        setLive(next)
      }
      push(state)

      void (async () => {
        let cid = convId
        try {
          if (cid === null) {
            const conv = await createConversation(titleFrom(question))
            cid = conv.id
            void qc.invalidateQueries({ queryKey: ['conversations'] })
            push({ ...state, convId: cid })
            navigate(`/c/${cid}`)
          } else {
            push({ ...state, previousMaxId: maxId(await fetchMessages(cid)) })
          }
          const events = streamChat(
            {
              conversation_id: cid,
              question: state.question,
              target_doc_ids: docs.map((d) => d.id),
              target_filenames: docs.map((d) => d.file_name),
              persona: personaRef.current,
            },
            controller.signal,
          )
          for await (const event of events) push(applyEvent(state, event))
          push(finishLive(state, 'done', performance.now()))
        } catch (err) {
          if (controller.signal.aborted) push(finishLive(state, 'stopped', performance.now()))
          else push(finishLive(state, 'error', performance.now(), err instanceof ApiError ? err.message : strings.errors.generic))
        } finally {
          abortRef.current = null
          if (cid !== null) await settle(cid, state)
          if (state.phase === 'error') setFailure({ convId: cid, question: state.question, error: state.error ?? '' })
          setLive(null)
          busyRef.current = false
        }
      })()
    },
    [qc, navigate, fetchMessages, settle],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const value = useMemo<ChatState>(
    () => ({ live, busy: live !== null, running: isRunning(live), meta, failure, persona, setPersona, send, stop }),
    [live, meta, failure, persona, setPersona, send, stop],
  )
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChat(): ChatState {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used inside <ChatProvider>')
  return ctx
}
