import { Moon, Sun } from 'lucide-react'
import { useEffect, useSyncExternalStore } from 'react'
import { Button } from '@/components/ui/button'
import { strings } from '@/i18n/en'

export type Theme = 'dark' | 'light'
const KEY = 'zai.theme'

function stored(): Theme {
  try {
    return localStorage.getItem(KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

// Μικρό κοινό store: το θέμα το διαβάζουν και άλλα κομμάτια (π.χ. οι ειδοποιήσεις), όχι μόνο το κουμπί.
let current: Theme = stored()
const listeners = new Set<() => void>()

function setTheme(next: Theme) {
  current = next
  document.documentElement.classList.toggle('dark', next === 'dark')
  try {
    localStorage.setItem(KEY, next)
  } catch {
    /* ιδιωτική περιήγηση: απλώς δεν θυμόμαστε */
  }
  for (const l of listeners) l()
}

export function toggleTheme(): void {
  setTheme(current === 'dark' ? 'light' : 'dark')
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    () => current,
  )
}

/** Σκούρο από προεπιλογή (όπως η σημερινή εφαρμογή). Η επιλογή μένει στον browser· δεν είναι δεδομένο του χρήστη. */
export function ThemeToggle() {
  const theme = useTheme()
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={strings.app.theme}
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
    >
      {theme === 'dark' ? <Sun /> : <Moon />}
    </Button>
  )
}
