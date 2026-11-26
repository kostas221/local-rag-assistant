/**
 * Κατάσταση σύνδεσης. Το token ζει στο sessionStorage (βλ. api/client.ts), άρα μια ανανέωση της σελίδας ΔΕΝ
 * αποσυνδέει, σε αντίθεση με το Streamlit. Όταν ο server απαντήσει 401 (έληξε το token), ο client στέλνει
 * AUTH_EXPIRED_EVENT και εδώ γυρνάμε στη σύνδεση.
 */
import { useQueryClient } from '@tanstack/react-query'
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { AUTH_EXPIRED_EVENT, login as apiLogin, tokenStore, usernameFromToken } from '@/api/client'

/** Ό,τι κρατά ο browser για τον χρήστη πέρα από το token: σβήνεται στην αποσύνδεση, για να μη το δει ο επόμενος. */
const SESSION_KEYS = ['zai.unchecked']

interface AuthState {
  token: string | null
  username: string | null
  signIn: (username: string, password: string) => Promise<void>
  signOut: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const [token, setToken] = useState<string | null>(() => tokenStore.get())

  const forget = useCallback(() => {
    qc.clear()
    try {
      for (const k of SESSION_KEYS) sessionStorage.removeItem(k)
    } catch {
      /* τίποτα να σβηστεί */
    }
  }, [qc])

  useEffect(() => {
    const onExpired = () => {
      forget()
      setToken(null)
    }
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired)
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired)
  }, [forget])

  const signIn = useCallback(async (username: string, password: string) => {
    setToken(await apiLogin(username, password))
  }, [])

  const signOut = useCallback(() => {
    tokenStore.clear()
    forget()
    setToken(null)
  }, [forget])

  const value = useMemo(
    () => ({ token, username: usernameFromToken(token), signIn, signOut }),
    [token, signIn, signOut],
  )
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
