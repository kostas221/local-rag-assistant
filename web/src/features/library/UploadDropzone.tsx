import { useQueryClient } from '@tanstack/react-query'
import { Upload } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'
import { toast } from 'sonner'
import { ApiError } from '@/api/client'
import { MAX_UPLOAD_BYTES, uploadDocument } from '@/api/endpoints'
import { Progress } from '@/components/ui/progress'
import { strings } from '@/i18n/en'
import { cn } from '@/lib/utils'

const s = strings.app

/** Το μήνυμα του διπλότυπου (409) έχει το όνομα του αρχείου σε «»· το κρατάμε, το υπόλοιπο είναι ελληνικά του server. */
function uploadError(err: unknown): string {
  if (err instanceof ApiError && err.status === 409) {
    const name = /«([^»]+)»/.exec(err.detail ?? '')?.[1]
    return name ? s.duplicateAs(name) : strings.errors.duplicate
  }
  if (err instanceof ApiError && err.status === 400) return s.onlyPdf
  return err instanceof ApiError ? err.message : strings.errors.generic
}

/**
 * Ανέβασμα PDF με drag & drop ή κλικ. Πολλά αρχεία ανεβαίνουν ένα-ένα. Οι έλεγχοι τύπου και μεγέθους γίνονται και
 * εδώ, για να μη στέλνονται 50 MB για να απορριφθούν· ο backend τους ξανακάνει, και αυτός αποφασίζει.
 */
export function UploadDropzone({ className }: { className?: string }) {
  const qc = useQueryClient()
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [current, setCurrent] = useState<{ name: string; progress: number } | null>(null)

  async function uploadAll(files: File[]) {
    for (const file of files) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        toast.error(`${file.name}: ${s.onlyPdf}`)
        continue
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        toast.error(`${file.name}: ${strings.errors.tooLarge}`)
        continue
      }
      setCurrent({ name: file.name, progress: 0 })
      try {
        await uploadDocument(file, (p) => setCurrent({ name: file.name, progress: p }))
        toast.success(`${file.name}: ${s.uploaded}`)
        await qc.invalidateQueries({ queryKey: ['documents'] })
      } catch (err) {
        toast.error(`${file.name}: ${uploadError(err)}`)
      }
    }
    setCurrent(null)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    if (!current) void uploadAll([...e.dataTransfer.files])
  }

  return (
    <div className={className}>
      <button
        type="button"
        disabled={current !== null}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          'flex w-full flex-col items-center justify-center gap-1 rounded-lg border border-dashed px-3 py-3 text-sm text-muted-foreground transition-colors',
          'hover:border-brand hover:text-foreground disabled:cursor-default disabled:hover:border-border',
          dragging && 'border-brand bg-brand/5 text-foreground',
        )}
      >
        {current ? (
          <span className="w-full space-y-1.5">
            <span className="block truncate text-xs">
              {s.uploading} {current.name}… {Math.round(current.progress * 100)}%
            </span>
            <Progress value={current.progress * 100} />
          </span>
        ) : (
          <>
            <span className="flex items-center gap-2">
              <Upload className="size-4" /> {s.upload}
            </span>
            <span className="text-[0.7rem]">{s.dropHere}</span>
          </>
        )}
      </button>
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(e) => {
          const files = [...(e.target.files ?? [])]
          e.target.value = ''
          void uploadAll(files)
        }}
      />
    </div>
  )
}
