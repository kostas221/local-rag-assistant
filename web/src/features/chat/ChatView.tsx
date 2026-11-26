import { useQuery } from '@tanstack/react-query'
import { Menu } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router'
import { ApiError } from '@/api/client'
import { getConversations, getMessages } from '@/api/endpoints'
import type { Message } from '@/api/types'
import { ZLogo } from '@/components/brand'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useDocuments, useLibrary } from '@/features/library/library'
import { UploadDropzone } from '@/features/library/UploadDropzone'
import { useShell } from '@/features/shell/shell'
import { strings } from '@/i18n/en'
import { AssistantMessage } from './AssistantMessage'
import { messagesKey, useChat } from './chat'
import { Composer } from './Composer'

const s = strings.app

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] rounded-2xl rounded-br-md bg-muted px-4 py-2.5 text-sm whitespace-pre-wrap">{text}</p>
    </div>
  )
}

/**
 * Ερώτηση χωρίς απάντηση στο ιστορικό. Ο backend σώζει την ερώτηση ΠΡΙΝ απαντήσει, και την απάντηση μόνο αν γράφτηκε
 * έστω μία λέξη. Ένα Stop στην αναζήτηση, ή ένα σφάλμα πριν από το κείμενο, αφήνει λοιπόν μόνη την ερώτηση· χωρίς
 * σημείωση θα έμοιαζε ότι κάτι ακόμα περιμένει.
 */
function UnansweredNote({ onAskAgain }: { onAskAgain: () => void }) {
  return (
    <p className="flex flex-wrap items-center justify-end gap-2 text-xs text-muted-foreground">
      {s.unanswered}
      <Button variant="outline" size="xs" onClick={onAskAgain}>
        {s.askAgain}
      </Button>
    </p>
  )
}

function EmptyState({ onExample }: { onExample: (q: string) => void }) {
  const { docs } = useLibrary()
  const { isPending } = useDocuments()
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center gap-4 px-4 pt-[12vh] text-center">
      <ZLogo size={44} />
      <h2 className="text-2xl font-semibold">{s.emptyTitle}</h2>
      <p className="text-sm text-muted-foreground">{s.emptyBody}</p>
      <p className="text-xs text-muted-foreground">{strings.scope}</p>
      {!isPending && docs.length === 0 ? (
        <UploadDropzone className="w-full max-w-sm" />
      ) : (
        <div className="flex flex-wrap justify-center gap-2">
          {s.examples.map((q) => (
            <Button key={q} variant="outline" size="sm" onClick={() => onExample(q)}>
              {q}
            </Button>
          ))}
        </div>
      )}
    </div>
  )
}

export function ChatView() {
  const params = useParams()
  const convId = params.convId && /^\d+$/.test(params.convId) ? Number(params.convId) : null
  const chat = useChat()
  const shell = useShell()
  const [draft, setDraft] = useState('')
  const scroller = useRef<HTMLDivElement>(null)
  const nearBottom = useRef(true)

  const messages = useQuery({
    queryKey: messagesKey(convId ?? -1),
    queryFn: () => getMessages(convId ?? -1),
    enabled: convId !== null,
  })
  const conversations = useQuery({ queryKey: ['conversations'], queryFn: getConversations })

  const live = chat.live && chat.live.convId === convId ? chat.live : null
  // Όσο γράφεται απάντηση, τα μηνύματα μετά την ερώτηση τα δείχνει η ζωντανή απάντηση, όχι ο server (αλλιώς η ερώτηση
  // θα φαινόταν δύο φορές, αφού ο backend τη σώζει πριν απαντήσει).
  const shown: Message[] = (messages.data ?? []).filter((m) => !live || m.id <= live.previousMaxId)
  const failure = chat.failure && chat.failure.convId === convId ? chat.failure : null
  const title =
    conversations.data?.find((c) => c.id === convId)?.title ?? (live ? live.question.slice(0, 60) : convId === null ? s.newChat : '')
  const gone = messages.error instanceof ApiError && messages.error.status === 404

  // Κύλιση στο τέλος όταν έρχεται κείμενο, ΜΟΝΟ αν ο χρήστης ήταν ήδη κάτω (αλλιώς διαβάζει κάτι πιο πάνω).
  useLayoutEffect(() => {
    const el = scroller.current
    if (el && nearBottom.current) el.scrollTop = el.scrollHeight
  }, [shown.length, live?.text, live?.phase, failure])

  useEffect(() => {
    nearBottom.current = true
  }, [convId])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-center gap-2 border-b px-4 py-2.5">
        <Button variant="ghost" size="icon-sm" className="md:hidden" aria-label={s.menu} onClick={shell.openMenu}>
          <Menu />
        </Button>
        <h1 className="truncate text-sm font-medium">{title}</h1>
      </header>

      <div
        ref={scroller}
        className="min-h-0 flex-1 overflow-y-auto"
        onScroll={(e) => {
          const el = e.currentTarget
          nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
        }}
      >
        {convId === null && !live ? (
          <EmptyState onExample={setDraft} />
        ) : (
          <div className="mx-auto max-w-3xl space-y-8 px-4 py-8">
            {messages.isPending && convId !== null && !live && (
              <div className="space-y-4">
                <Skeleton className="ml-auto h-10 w-2/3" />
                <Skeleton className="h-24 w-full" />
              </div>
            )}
            {gone && (
              <p className="text-sm text-muted-foreground">
                {strings.errors.notFound}{' '}
                <Link to="/" className="underline">
                  {s.newChat}
                </Link>
              </p>
            )}
            {messages.isError && !gone && <p className="text-sm text-muted-foreground">{strings.errors.network}</p>}
            {shown.map((m, i) =>
              m.role === 'user' ? (
                <div key={m.id} className="space-y-2">
                  <UserBubble text={m.content} />
                  {/* Χωρίς απάντηση μετά: η επόμενη είναι πάλι ερώτηση, ή είναι η τελευταία (εκτός αν το σφάλμα φαίνεται ήδη). */}
                  {shown[i + 1]?.role !== 'assistant' && !(i === shown.length - 1 && !live && failure) && (
                    <UnansweredNote onAskAgain={() => setDraft(m.content)} />
                  )}
                </div>
              ) : (
                <AssistantMessage
                  key={m.id}
                  content={m.content}
                  sources={m.sources}
                  messageId={m.id}
                  savedPassages={m.passages}
                  meta={chat.meta.get(m.id)}
                  question={shown[i - 1]?.role === 'user' ? shown[i - 1]?.content : null}
                  convId={convId}
                />
              ),
            )}
            {live && (
              <>
                <UserBubble text={live.question} />
                <AssistantMessage
                  content={live.text}
                  sources={live.sources}
                  messageId={null}
                  savedPassages={null}
                  live={{ phase: live.phase, startedAt: live.startedAt, incomplete: live.incomplete }}
                />
              </>
            )}
            {failure && (
              <Alert variant="destructive">
                <AlertDescription className="flex flex-wrap items-center justify-between gap-2">
                  <span>
                    {s.errorTitle} {failure.error}
                  </span>
                  <Button variant="outline" size="xs" disabled={chat.busy} onClick={() => chat.send(failure.question, convId)}>
                    {s.retry}
                  </Button>
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}
      </div>

      <Composer convId={convId} value={draft} onChange={setDraft} />
    </div>
  )
}
