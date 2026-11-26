import { useMutation, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, Loader2, MoreHorizontal, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { ApiError } from '@/api/client'
import { deleteDocument } from '@/api/endpoints'
import type { LibraryDocument } from '@/api/types'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Skeleton } from '@/components/ui/skeleton'
import { strings } from '@/i18n/en'
import { cn } from '@/lib/utils'
import { useDocuments, useLibrary } from './library'
import { UploadDropzone } from './UploadDropzone'

const s = strings.app

function statusLine(d: LibraryDocument): { text: string; warn: boolean; busy: boolean } {
  if (d.status === 'processing') return { text: s.statusProcessing, warn: false, busy: true }
  if (d.status === 'empty') return { text: s.statusEmpty, warn: true, busy: false }
  if (d.status === 'failed') return { text: s.statusFailed, warn: true, busy: false }
  const owner = d.is_mine ? s.you : (d.uploader_name ?? 'Unknown')
  const partial = Boolean(d.pages_total) && d.pages_text !== d.pages_total
  // Μερική εξαγωγή: λέμε ποιες σελίδες δεν έχουν κείμενο (π.χ. σκαναρισμένες), γιατί εκεί η αναζήτηση δεν βλέπει.
  if (partial && d.extraction) return { text: `${owner} · ${d.extraction}`, warn: true, busy: false }
  return { text: d.pages_total ? `${owner} · ${s.pages(d.pages_total)}` : owner, warn: false, busy: false }
}

function DocumentRow({ doc, onDelete }: { doc: LibraryDocument; onDelete: (d: LibraryDocument) => void }) {
  const { isChecked, toggle } = useLibrary()
  const ready = doc.status === 'ready'
  const line = statusLine(doc)
  return (
    <li className="group/doc relative">
      <label
        className={cn(
          'flex items-start gap-2 rounded-md py-1.5 pr-8 pl-2',
          ready ? 'cursor-pointer hover:bg-muted/60' : 'cursor-default opacity-75',
        )}
        title={doc.file_name}
      >
        <Checkbox className="mt-0.5" checked={ready && isChecked(doc.id)} disabled={!ready} onCheckedChange={() => toggle(doc.id)} />
        <span className="min-w-0 text-sm">
          <span className="block truncate">{doc.file_name}</span>
          <span className={cn('flex items-center gap-1 text-xs', line.warn ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground')}>
            {line.busy && <Loader2 className="size-3 shrink-0 animate-spin" />}
            {line.warn && <AlertTriangle className="size-3 shrink-0" />}
            <span className="truncate" title={line.text}>
              {line.text}
            </span>
          </span>
        </span>
      </label>
      {doc.is_mine && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={`${s.deleteDocument}: ${doc.file_name}`}
              className="absolute top-1.5 right-1 opacity-60 group-hover/doc:opacity-100 focus-visible:opacity-100"
            >
              <MoreHorizontal />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem variant="destructive" onSelect={() => onDelete(doc)}>
              <Trash2 /> {s.delete}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </li>
  )
}

export function LibraryPanel() {
  const qc = useQueryClient()
  const { isPending, isError } = useDocuments()
  const { docs, selectAll, clearAll } = useLibrary()
  const [pending, setPending] = useState<LibraryDocument | null>(null)

  const remove = useMutation({
    mutationFn: (id: number) => deleteDocument(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['documents'] }),
    onError: (err) => toast.error(err instanceof ApiError ? err.message : strings.errors.generic),
  })

  return (
    <section>
      <div className="flex items-center justify-between px-1 pb-1">
        <h2 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{s.library}</h2>
        {docs.length > 0 && (
          <div className="flex gap-1">
            <Button variant="ghost" size="xs" onClick={selectAll}>
              {s.selectAll}
            </Button>
            <Button variant="ghost" size="xs" onClick={clearAll}>
              {s.clear}
            </Button>
          </div>
        )}
      </div>
      {isPending ? (
        <div className="space-y-1.5 px-1">
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
        </div>
      ) : isError ? (
        <p className="px-2 text-xs text-muted-foreground">{strings.errors.network}</p>
      ) : docs.length === 0 ? (
        <p className="px-2 text-xs text-muted-foreground">{s.noDocuments}</p>
      ) : (
        <ul className="space-y-0.5">
          {docs.map((d) => (
            <DocumentRow key={d.id} doc={d} onDelete={setPending} />
          ))}
        </ul>
      )}
      <UploadDropzone className="mt-2" />
      {docs.length > 1 && <p className="mt-2 px-1 text-xs text-muted-foreground">💡 {s.compareTip}</p>}
      <ConfirmDialog
        open={pending !== null}
        onOpenChange={(o) => !o && setPending(null)}
        title={s.deleteDocument}
        body={pending ? s.deleteDocumentBody(pending.file_name) : ''}
        confirm={s.delete}
        onConfirm={() => {
          if (pending) remove.mutate(pending.id)
          setPending(null)
        }}
      />
    </section>
  )
}
