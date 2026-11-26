import { Send, Square, X } from 'lucide-react'
import { useEffect, useRef, type KeyboardEvent } from 'react'
import { PERSONAS, type Persona } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useLibrary } from '@/features/library/library'
import { strings } from '@/i18n/en'
import { useChat } from './chat'

const s = strings.app

interface Props {
  convId: number | null
  value: string
  onChange: (v: string) => void
}

/** Το πεδίο ερώτησης: Enter στέλνει, Shift+Enter αλλάζει γραμμή. Όσο γράφεται η απάντηση, το κουμπί γίνεται Stop. */
export function Composer({ convId, value, onChange }: Props) {
  const chat = useChat()
  const { selected, readyCount, docs, selectAll } = useLibrary()
  const area = useRef<HTMLTextAreaElement>(null)
  const noneSelected = docs.length > 0 && selected.length === 0
  const canSend = !chat.busy && value.trim().length > 0 && selected.length > 0
  // Σύγκριση: 2–3 τσεκαρισμένα από περισσότερα. Δεν υποσχόμαστε «όλο το κείμενο»: αν χωράνε το ξέρει ο backend.
  const comparing = selected.length >= 2 && selected.length <= 3 && selected.length < readyCount

  useEffect(() => {
    const el = area.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  const submit = () => {
    if (!canSend) return
    chat.send(value, convId)
    onChange('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t bg-background px-4 py-3">
      <div className="mx-auto max-w-3xl space-y-1.5">
        {noneSelected && <p className="px-1 text-xs text-amber-600 dark:text-amber-400">{s.selectAtLeastOne}</p>}
        <div className="rounded-2xl border bg-card p-2 shadow-sm focus-within:ring-2 focus-within:ring-brand/40">
          {comparing && (
            <div className="flex min-w-0 items-center gap-1.5 px-2 pt-1 pb-2">
              <Badge variant="secondary" className="max-w-full shrink justify-start gap-1">
                <span className="min-w-0 truncate" title={selected.map((d) => d.file_name).join(' · ')}>
                  {s.comparing}: {selected.map((d) => d.file_name.replace(/\.pdf$/i, '')).join(' · ')}
                </span>
                <button type="button" className="shrink-0" aria-label={s.searchAll} title={s.searchAll} onClick={selectAll}>
                  <X className="size-3" />
                </button>
              </Badge>
            </div>
          )}
          <Textarea
            ref={area}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={s.askPlaceholder}
            aria-label={s.askPlaceholder}
            className="max-h-[200px] min-h-12 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
            rows={2}
          />
          <div className="flex items-center justify-between gap-2 px-1 pt-1">
            <Select value={chat.persona} onValueChange={(v) => chat.setPersona(v as Persona)} disabled={chat.busy}>
              <SelectTrigger size="sm" className="w-36" aria-label={s.style}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PERSONAS.map((p) => (
                  <SelectItem key={p} value={p}>
                    {p}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {chat.running ? (
              <Button size="sm" variant="outline" onClick={chat.stop}>
                <Square className="fill-current" /> {s.stop}
              </Button>
            ) : (
              <Button size="sm" className="bg-brand text-brand-foreground hover:bg-brand/90" disabled={!canSend} onClick={submit}>
                <Send /> {s.send}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
