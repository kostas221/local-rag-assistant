/**
 * Η βιβλιοθήκη: τα έγγραφα του χρήστη (και τα δημόσια) και ποια είναι τσεκαρισμένα για αναζήτηση.
 *
 * Κρατάμε τα ΑΤΣΕΚΑΡΙΣΤΑ, όχι τα τσεκαρισμένα: ένα νέο έγγραφο μπαίνει τσεκαρισμένο, όπως στο Streamlit. Η επιλογή
 * ζει στο sessionStorage, όπως και το token: μένει μετά από ανανέωση, σβήνει όταν κλείσει η καρτέλα.
 */
import { useQuery } from '@tanstack/react-query'
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { getDocuments } from '@/api/endpoints'
import type { LibraryDocument } from '@/api/types'

const KEY = 'zai.unchecked'
const POLL_MS = 4000 // όσο κάτι επεξεργάζεται, όπως το Streamlit

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: getDocuments,
    refetchInterval: (query) => (query.state.data?.some((d) => d.status === 'processing') ? POLL_MS : false),
  })
}

function load(): Set<number> {
  try {
    const raw = sessionStorage.getItem(KEY)
    const ids = raw ? (JSON.parse(raw) as unknown) : []
    return new Set(Array.isArray(ids) ? ids.filter((x): x is number => typeof x === 'number') : [])
  } catch {
    return new Set()
  }
}

function save(ids: Set<number>) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify([...ids]))
  } catch {
    /* χωρίς αποθήκευση: η επιλογή απλώς δεν θα μείνει μετά από ανανέωση */
  }
}

interface LibraryState {
  docs: LibraryDocument[]
  isChecked: (id: number) => boolean
  toggle: (id: number) => void
  selectAll: () => void
  clearAll: () => void
  /** Τσεκάρει ΜΟΝΟ αυτά (π.χ. «Compare only these»). */
  selectOnly: (ids: number[]) => void
  /** Τα έτοιμα και τσεκαρισμένα: αυτά πάνε στην ερώτηση. */
  selected: LibraryDocument[]
  readyCount: number
}

const LibraryContext = createContext<LibraryState | null>(null)

export function LibraryProvider({ children }: { children: ReactNode }) {
  const { data } = useDocuments()
  const [unchecked, setUnchecked] = useState<Set<number>>(load)

  const update = useCallback((next: Set<number>) => {
    setUnchecked(next)
    save(next)
  }, [])

  const value = useMemo<LibraryState>(() => {
    const docs = data ?? []
    const ready = docs.filter((d) => d.status === 'ready')
    return {
      docs,
      isChecked: (id) => !unchecked.has(id),
      toggle: (id) => {
        const next = new Set(unchecked)
        if (next.has(id)) next.delete(id)
        else next.add(id)
        update(next)
      },
      selectAll: () => update(new Set()),
      clearAll: () => update(new Set(docs.map((d) => d.id))),
      selectOnly: (ids) => update(new Set(docs.filter((d) => !ids.includes(d.id)).map((d) => d.id))),
      selected: ready.filter((d) => !unchecked.has(d.id)),
      readyCount: ready.length,
    }
  }, [data, unchecked, update])

  return <LibraryContext.Provider value={value}>{children}</LibraryContext.Provider>
}

export function useLibrary(): LibraryState {
  const ctx = useContext(LibraryContext)
  if (!ctx) throw new Error('useLibrary must be used inside <LibraryProvider>')
  return ctx
}
