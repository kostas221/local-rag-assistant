import { LogOut, Plus } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Outlet, useNavigate } from 'react-router'
import { ZLogo } from '@/components/brand'
import { ThemeToggle } from '@/components/theme'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { useAuth } from '@/features/auth/auth'
import { ChatProvider, useChat } from '@/features/chat/chat'
import { ConversationList } from '@/features/conversations/ConversationList'
import { LibraryPanel } from '@/features/library/LibraryPanel'
import { LibraryProvider } from '@/features/library/library'
import { SourceViewer } from '@/features/viewer/SourceViewer'
import { useViewer, ViewerProvider } from '@/features/viewer/viewer'
import { strings } from '@/i18n/en'
import { useMediaQuery } from '@/lib/useMediaQuery'
import { ShellContext } from './shell'

const s = strings.app

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { username, signOut } = useAuth()
  const chat = useChat()
  const navigate = useNavigate()
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 px-4 py-3">
        <ZLogo size={26} />
        <span className="font-semibold tracking-wide">{strings.brand}</span>
      </div>
      <div className="px-3">
        <Button
          className="w-full justify-start"
          variant="outline"
          onClick={() => {
            navigate('/')
            onNavigate?.()
          }}
        >
          <Plus /> {s.newChat}
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-6 px-3 py-4">
          <section>
            <h2 className="px-1 pb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">{s.conversations}</h2>
            <ConversationList onNavigate={onNavigate} />
          </section>
          <LibraryPanel />
        </div>
      </ScrollArea>
      <Separator />
      <div className="flex items-center justify-between gap-2 px-4 py-3 text-xs">
        <span className="truncate text-muted-foreground">
          {strings.auth.signedInAs} <span className="font-medium text-foreground">{username}</span>
        </span>
        <div className="flex items-center">
          <ThemeToggle />
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={strings.auth.signOut}
            onClick={() => {
              chat.stop()
              signOut()
            }}
          >
            <LogOut />
          </Button>
        </div>
      </div>
    </div>
  )
}

/** Το πάνελ της πηγής: στήλη δίπλα στην απάντηση σε φαρδιά οθόνη, συρτάρι πάνω από αυτήν σε στενή. Esc κλείνει. */
function ViewerPane() {
  const wide = useMediaQuery('(min-width: 1024px)')
  const { state, close } = useViewer()
  useEffect(() => {
    if (!wide || !state) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented) close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [wide, state, close])

  if (wide) return state ? <aside className="w-[min(520px,42vw)] shrink-0 border-l">{<SourceViewer state={state} />}</aside> : null
  return (
    <Sheet open={state !== null} onOpenChange={(o) => !o && close()}>
      <SheetContent
        side="right"
        className="gap-0 p-0 data-[side=right]:w-full data-[side=right]:sm:max-w-full"
        showCloseButton={false}
      >
        <SheetTitle className="sr-only">{s.sourceViewer}</SheetTitle>
        {state && <SourceViewer state={state} />}
      </SheetContent>
    </Sheet>
  )
}

function Layout() {
  const [menuOpen, setMenuOpen] = useState(false)
  const shell = useMemo(() => ({ openMenu: () => setMenuOpen(true) }), [])
  return (
    <ShellContext.Provider value={shell}>
      <div className="flex h-svh overflow-hidden bg-background">
        <aside className="hidden w-72 shrink-0 border-r bg-sidebar md:block">
          <Sidebar />
        </aside>
        <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
          <SheetContent side="left" className="gap-0 p-0 data-[side=left]:w-80 data-[side=left]:max-w-[85vw]">
            <SheetTitle className="sr-only">{s.menu}</SheetTitle>
            <Sidebar onNavigate={() => setMenuOpen(false)} />
          </SheetContent>
        </Sheet>
        <main className="flex min-w-0 flex-1 flex-col">
          <Outlet />
        </main>
        <ViewerPane />
      </div>
    </ShellContext.Provider>
  )
}

/** Η εφαρμογή μετά τη σύνδεση: βιβλιοθήκη, συνομιλία και πάνελ πηγής ζουν εδώ, πάνω από τις οθόνες. */
export function AppShell() {
  return (
    <LibraryProvider>
      <ChatProvider>
        <ViewerProvider>
          <Layout />
        </ViewerProvider>
      </ChatProvider>
    </LibraryProvider>
  )
}
