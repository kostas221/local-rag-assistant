import type { ReactNode } from 'react'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { strings } from '@/i18n/en'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  body: ReactNode
  confirm: string
  onConfirm: () => void
  /** false για επιβεβαίωση που δεν σβήνει τίποτα (π.χ. νέα ερώτηση με κόστος). */
  destructive?: boolean
}

/** Επιβεβαίωση πριν από κάτι που δεν αναιρείται (διαγραφή) ή που κοστίζει (νέα κλήση μοντέλου). */
export function ConfirmDialog({ open, onOpenChange, title, body, confirm, onConfirm, destructive = true }: Props) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{body}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{strings.app.cancel}</AlertDialogCancel>
          <AlertDialogAction variant={destructive ? 'destructive' : 'default'} onClick={onConfirm}>
            {confirm}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
