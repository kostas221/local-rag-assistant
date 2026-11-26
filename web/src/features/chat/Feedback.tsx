import { ThumbsDown, ThumbsUp } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { sendFeedback } from '@/api/endpoints'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Textarea } from '@/components/ui/textarea'
import { strings } from '@/i18n/en'
import { cn } from '@/lib/utils'

const s = strings.app

/**
 * 👍/👎 με προαιρετικό σχόλιο στο 👎. Ο backend κρατά ΕΝΑ feedback ανά (χρήστη, μήνυμα) και το ενημερώνει.
 * Δεν το επιστρέφει με το ιστορικό, οπότε μετά το ξαναφόρτωμα φαίνεται κενό (όπως και στο Streamlit· Φάση 4).
 */
export function Feedback({ messageId }: { messageId: number }) {
  const [vote, setVote] = useState<boolean | null>(null)
  const [commentOpen, setCommentOpen] = useState(false)
  const [comment, setComment] = useState('')
  const [commentSent, setCommentSent] = useState(false)
  const [busy, setBusy] = useState(false)

  async function submit(positive: boolean, text?: string) {
    setBusy(true)
    try {
      await sendFeedback({ message_id: messageId, is_positive: positive, comment: text ?? null })
      setVote(positive)
      if (text) {
        setCommentSent(true)
        setCommentOpen(false)
        toast.success(s.commentThanks)
      } else {
        toast.success(s.feedbackThanks)
        if (!positive && !commentSent) setCommentOpen(true)
      }
    } catch {
      toast.error(s.feedbackFailed)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex items-center gap-0.5">
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={s.helpful}
        aria-pressed={vote === true}
        disabled={busy}
        className={cn(vote === true && 'text-brand')}
        onClick={() => void submit(true)}
      >
        <ThumbsUp />
      </Button>
      <Popover open={commentOpen} onOpenChange={setCommentOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={s.notHelpful}
            aria-pressed={vote === false}
            disabled={busy}
            className={cn(vote === false && 'text-brand')}
            onClick={(e) => {
              e.preventDefault()
              void submit(false)
            }}
          >
            <ThumbsDown />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80 space-y-2" align="start">
          <label htmlFor={`fb-${messageId}`} className="text-sm font-medium">
            {s.whatWentWrong}
          </label>
          <Textarea
            id={`fb-${messageId}`}
            value={comment}
            maxLength={1000}
            placeholder={s.whatWentWrongPlaceholder}
            onChange={(e) => setComment(e.target.value)}
            rows={3}
          />
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={() => setCommentOpen(false)}>
              {s.cancel}
            </Button>
            <Button size="sm" disabled={busy || !comment.trim()} onClick={() => void submit(false, comment.trim())}>
              {s.send}
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
