import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { MessageSquare, MoreHorizontal, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { NavLink, useMatch, useNavigate } from 'react-router'
import { toast } from 'sonner'
import { ApiError } from '@/api/client'
import { deleteConversation, getConversations } from '@/api/endpoints'
import type { Conversation } from '@/api/types'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Skeleton } from '@/components/ui/skeleton'
import { useChat } from '@/features/chat/chat'
import { strings } from '@/i18n/en'
import { cn } from '@/lib/utils'
import { chatGroup } from './chatGroup'

const s = strings.app

function groupConversations(list: Conversation[]): [string, Conversation[]][] {
  const groups = new Map<string, Conversation[]>()
  for (const c of list) {
    const g = chatGroup(c.created_at)
    groups.set(g, [...(groups.get(g) ?? []), c])
  }
  return [...groups.entries()]
}

export function ConversationList({ onNavigate }: { onNavigate?: () => void }) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  // Η λίστα είναι στο πλαίσιο (layout), άρα δεν βλέπει τις παραμέτρους της οθόνης: το id από το URL.
  const convId = useMatch('/c/:convId')?.params.convId
  const chat = useChat()
  const [pending, setPending] = useState<Conversation | null>(null)
  const { data, isPending, isError } = useQuery({ queryKey: ['conversations'], queryFn: getConversations })

  const remove = useMutation({
    mutationFn: (id: number) => deleteConversation(id),
    onSuccess: (_r, id) => {
      qc.removeQueries({ queryKey: ['messages', id] })
      void qc.invalidateQueries({ queryKey: ['conversations'] })
      if (String(id) === convId) navigate('/')
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : strings.errors.generic),
  })

  if (isPending)
    return (
      <div className="space-y-1.5 px-1">
        <Skeleton className="h-7 w-full" />
        <Skeleton className="h-7 w-4/5" />
      </div>
    )
  if (isError) return <p className="px-2 text-xs text-muted-foreground">{strings.errors.network}</p>
  if (data.length === 0) return <p className="px-2 text-xs text-muted-foreground">{s.noConversations}</p>

  return (
    <>
      <div className="space-y-3">
        {groupConversations(data).map(([group, items]) => (
          <div key={group}>
            <p className="px-2 pb-1 text-[0.7rem] text-muted-foreground">{group}</p>
            <ul className="space-y-0.5">
              {items.map((c) => (
                <li key={c.id} className="group/conv relative">
                  <NavLink
                    to={`/c/${c.id}`}
                    onClick={onNavigate}
                    title={c.title ?? ''}
                    className={({ isActive }) =>
                      cn(
                        'flex w-full items-center gap-2 rounded-md py-1.5 pr-8 pl-2 text-left text-sm',
                        isActive ? 'bg-muted font-medium' : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
                      )
                    }
                  >
                    <MessageSquare className="size-4 shrink-0" />
                    <span className="truncate">{c.title || s.newChat}</span>
                  </NavLink>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        aria-label={`${s.deleteConversation}: ${c.title ?? ''}`}
                        className="absolute top-1/2 right-1 -translate-y-1/2 opacity-60 group-hover/conv:opacity-100 focus-visible:opacity-100"
                      >
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem variant="destructive" onSelect={() => setPending(c)}>
                        <Trash2 /> {s.delete}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <ConfirmDialog
        open={pending !== null}
        onOpenChange={(o) => !o && setPending(null)}
        title={s.deleteConversation}
        body={s.deleteConversationBody}
        confirm={s.delete}
        onConfirm={() => {
          if (!pending) return
          if (chat.live?.convId === pending.id) chat.stop()
          remove.mutate(pending.id)
          setPending(null)
        }}
      />
    </>
  )
}
