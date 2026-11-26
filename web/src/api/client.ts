/**
 * Ένα σημείο για κάθε κλήση στον backend: token, σφάλματα, έλεγχος σχήματος.
 *
 * Token: στο sessionStorage (απόφαση του πλάνου, φάση 1). Μένει μετά από ανανέωση και σβήνει όταν κλείσει η καρτέλα.
 * Το cookie httpOnly μπαίνει πριν από τη δημόσια φιλοξενία. Το κείμενο του μοντέλου δεν αποδίδεται ποτέ ως HTML,
 * άρα ένα PDF με επίθεση δεν έχει τρόπο να τρέξει κώδικα που θα διάβαζε το token.
 */
import type { z } from 'zod'
import { strings } from '@/i18n/en'
import { LoginResponseSchema } from './types'

const TOKEN_KEY = 'zai.token'
export const API_BASE = '/api'

function storage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.sessionStorage
  } catch {
    return null // π.χ. ιδιωτική περιήγηση με μπλοκαρισμένη αποθήκευση
  }
}

export const tokenStore = {
  get: (): string | null => storage()?.getItem(TOKEN_KEY) ?? null,
  set: (token: string): void => storage()?.setItem(TOKEN_KEY, token),
  clear: (): void => storage()?.removeItem(TOKEN_KEY),
}

/** Το ίδιο το όνομα χρήστη από το JWT (πεδίο `sub`), μόνο για εμφάνιση· η επαλήθευση γίνεται στον server. */
export function usernameFromToken(token: string | null): string | null {
  if (!token) return null
  try {
    const payload = token.split('.')[1] ?? ''
    const json = JSON.parse(atob(payload.replace(/-/g, '+').replace(/_/g, '/'))) as { sub?: unknown }
    return typeof json.sub === 'string' ? json.sub : null
  } catch {
    return null
  }
}

export const AUTH_EXPIRED_EVENT = 'zai:auth-expired'

export class ApiError extends Error {
  readonly status: number
  readonly detail: string | null
  constructor(status: number, detail: string | null) {
    super(userMessage(status, detail))
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/**
 * Το κείμενο που βλέπει ο χρήστης, ανά κωδικό. Τα μηνύματα του server είναι στα ελληνικά, οπότε το UI (αγγλικά)
 * δείχνει δικό του κείμενο και κρατά το μήνυμα του server ως λεπτομέρεια (`detail`).
 */
export function userMessage(status: number, detail: string | null): string {
  const e = strings.errors
  if (status === 0) return e.network
  if (status === 401) return e.unauthorized
  if (status === 404) return e.notFound
  if (status === 409) return e.duplicate
  if (status === 413) return e.tooLarge
  // Το 422 φέρνει το πρώτο μήνυμα του pydantic, στα αγγλικά (π.χ. «Password must be at least 8 characters long.»).
  if (status === 422) return detail ? detail.replace(/^Value error,\s*/i, '') : e.invalid
  if (status === 429) return e.rateLimited
  if (status >= 500) return e.server
  return detail ?? e.generic
}

async function detailOf(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown }
    return typeof body.detail === 'string' ? body.detail : null
  } catch {
    return null
  }
}

export interface RequestOptions extends Omit<RequestInit, 'body'> {
  json?: unknown
  body?: BodyInit
}

/** fetch με token. Σε 401 σβήνει το token και ειδοποιεί την εφαρμογή να γυρίσει στη σύνδεση. */
export async function apiFetch(path: string, options: RequestOptions = {}): Promise<Response> {
  const { json, headers, ...rest } = options
  const h = new Headers(headers)
  const token = tokenStore.get()
  if (token) h.set('Authorization', `Bearer ${token}`)
  let body = rest.body
  if (json !== undefined) {
    h.set('Content-Type', 'application/json')
    body = JSON.stringify(json)
  }
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, { ...rest, headers: h, body })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError(0, null)
  }
  if (res.status === 401 && token) expireSession()
  if (!res.ok) throw new ApiError(res.status, await detailOf(res))
  return res
}

/** Το token δεν ισχύει πια: σβήνεται, και η εφαρμογή γυρίζει στη σύνδεση. */
export function expireSession(): void {
  tokenStore.clear()
  window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT))
}

/** JSON με έλεγχο σχήματος: αν ο backend αλλάξει, το λάθος φαίνεται εδώ, με το όνομα του endpoint. */
export async function apiJson<T>(path: string, schema: z.ZodType<T>, options?: RequestOptions): Promise<T> {
  const res = await apiFetch(path, options)
  const data: unknown = await res.json()
  const parsed = schema.safeParse(data)
  if (!parsed.success) {
    console.error(`Unexpected response from ${path}`, parsed.error.issues)
    throw new ApiError(-1, `Unexpected response from ${path}`)
  }
  return parsed.data
}

/** Σύνδεση: το /login του FastAPI θέλει φόρμα (OAuth2), όχι JSON. */
export async function login(username: string, password: string): Promise<string> {
  const form = new URLSearchParams({ username, password })
  const res = await apiFetch('/login', {
    method: 'POST',
    body: form,
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  })
  const parsed = LoginResponseSchema.parse(await res.json())
  tokenStore.set(parsed.access_token)
  return parsed.access_token
}
