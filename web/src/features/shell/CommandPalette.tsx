import { useQuery } from '@tanstack/react-query'
import { FileText, MessageSquare, Moon, Plus } from 'lucide-react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { getConversations } from '@/api/endpoints'
import { toggleTheme } from '@/components/theme'
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from '@/components/ui/command'
import { useLibrary } from '@/features/library/library'
import { strings } from '@/i18n/en'

const p = strings.app.palette

/**
 * Ctrl+K (ή ⌘K): αναζήτηση σε συνομιλίες και έγγραφα, και γρήγορες ενέργειες. Ένα έγγραφο εδώ σημαίνει «ρώτα μόνο
 * αυτό»: τσεκάρεται μόνο του στη βιβλιοθήκη, ώστε η επόμενη ερώτηση να πάει μόνο εκεί.
 */
export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const navigate = useNavigate()
  const { docs, selectOnly } = useLibrary()
  const conversations = useQuery({ queryKey: ['conversations'], queryFn: getConversations })
  const run = (action: () => void) => {
    onOpenChange(false)
    action()
  }
  const ready = docs.filter((d) => d.status === 'ready')

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title={p.title} description={p.description}>
      {/* Η νέα έκδοση του CommandDialog του shadcn ΔΕΝ βάζει μόνη της το <Command>: χωρίς αυτό το cmdk σπάει. */}
      <Command>
        <CommandInput placeholder={p.placeholder} />
        <CommandList>
          <CommandEmpty>{p.empty}</CommandEmpty>
          <CommandGroup heading={p.actions}>
            <CommandItem onSelect={() => run(() => navigate('/'))}>
              <Plus /> {strings.app.newChat}
            </CommandItem>
            <CommandItem onSelect={() => run(toggleTheme)}>
              <Moon /> {strings.app.theme}
            </CommandItem>
          </CommandGroup>
          {(conversations.data?.length ?? 0) > 0 && (
            <CommandGroup heading={strings.app.conversations}>
              {conversations.data?.map((c) => (
                <CommandItem
                  key={c.id}
                  value={`conversation ${c.id} ${c.title ?? ''}`}
                  onSelect={() => run(() => navigate(`/c/${c.id}`))}
                >
                  <MessageSquare /> <span className="truncate">{c.title || strings.app.newChat}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          )}
          {ready.length > 0 && (
            <CommandGroup heading={p.askOnly}>
              {ready.map((d) => (
                <CommandItem
                  key={d.id}
                  value={`document ${d.id} ${d.file_name}`}
                  onSelect={() =>
                    run(() => {
                      selectOnly([d.id])
                      toast.success(p.onlyChecked(d.file_name))
                    })
                  }
                >
                  <FileText /> <span className="truncate">{d.file_name}</span>
                  <CommandShortcut>{p.only}</CommandShortcut>
                </CommandItem>
              ))}
            </CommandGroup>
          )}
        </CommandList>
      </Command>
    </CommandDialog>
  )
}
