/**
 * Οι κλήσεις του backend, μία συνάρτηση ανά endpoint. Οι διαδρομές είναι αυτές του `backend/main.py` (χωρίς `/api`:
 * το πρόθεμα το βάζει ο client και το αφαιρεί το proxy).
 */
import { z } from 'zod'
import { API_BASE, ApiError, apiFetch, apiJson, expireSession, tokenStore } from './client'
import { readNdjson } from './ndjson'
import {
  ConversationSchema,
  DocumentSchema,
  MessageSchema,
  parseHighlight,
  PassagesSchema,
  StreamEventSchema,
  type HighlightResult,
  type Passages,
  type Persona,
  type StreamEvent,
} from './types'

export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024 // ίδιο όριο με το /upload

export const getDocuments = () => apiJson('/documents', z.array(DocumentSchema))

export const deleteDocument = (id: number) => apiFetch(`/documents/${id}`, { method: 'DELETE' })

export const getConversations = () => apiJson('/conversations', z.array(ConversationSchema))

export const createConversation = (title: string) =>
  apiJson('/conversations', ConversationSchema, { method: 'POST', json: { title } })

export const deleteConversation = (id: number) => apiFetch(`/conversations/${id}`, { method: 'DELETE' })

export const getMessages = (conversationId: number) =>
  apiJson(`/conversations/${conversationId}/messages`, z.array(MessageSchema))

/**
 * Χωρία ανά πρόταση (βήμα 6). ΜΙΑ κλήση μοντέλου, που ο backend σώζει στο μήνυμα: δεύτερο αίτημα δεν ξαναπληρώνει.
 * Το «failed» ΔΕΝ σώζεται εκεί, και εδώ γίνεται σφάλμα, ώστε να μη μείνει στη μνήμη και να μπορεί να ξαναδοκιμαστεί.
 */
export async function getPassages(messageId: number): Promise<Passages> {
  const data = await apiJson(`/messages/${messageId}/passages`, PassagesSchema, { method: 'POST' })
  if (data.status === 'failed') throw new ApiError(-1, 'The passage check did not complete.')
  return data
}

export const sendFeedback = (body: { message_id: number; is_positive: boolean; comment?: string | null }) =>
  apiFetch('/feedback', { method: 'POST', json: body })

export const register = (body: { username: string; email: string; password: string }) =>
  apiFetch('/register', { method: 'POST', json: body })

export interface PageImage {
  blob: Blob
  highlight: HighlightResult
}

/** Η σελίδα ως PNG. Το χωρίο πάει στο σώμα του POST, όχι στο URL, ώστε να μη γράφεται κείμενο σε logs. */
export async function getPageImage(
  docId: number,
  page: number,
  quote: string | null,
  signal?: AbortSignal,
): Promise<PageImage> {
  const res = await apiFetch(`/documents/${docId}/page-image`, {
    method: 'POST',
    json: { page, quote },
    signal,
  })
  return { blob: await res.blob(), highlight: parseHighlight(res.headers.get('X-Highlight')) }
}

export interface ChatRequest {
  conversation_id: number
  question: string
  target_doc_ids: number[]
  target_filenames: string[]
  persona: Persona
}

/**
 * Η απάντηση ως ροή γεγονότων. Το ιστορικό ΔΕΝ στέλνεται: ο backend το διαβάζει από τη συνομιλία (δεν εμπιστεύεται
 * τον client). Με `signal.abort()` (κουμπί Stop) κλείνει η σύνδεση· ο backend σταματά την παραγωγή και σώζει ό,τι
 * γράφτηκε ως εκείνη τη στιγμή. Άγνωστα γεγονότα αγνοούνται, ώστε νέος τύπος στον backend να μη σπάει το UI.
 */
export async function* streamChat(body: ChatRequest, signal: AbortSignal): AsyncGenerator<StreamEvent> {
  const res = await apiFetch('/chat', { method: 'POST', json: body, signal })
  if (!res.body) throw new ApiError(-1, 'The server sent no response stream.')
  for await (const raw of readNdjson(res.body)) {
    const event = StreamEventSchema.safeParse(raw)
    if (event.success) yield event.data
  }
}

function detailFromXhr(xhr: XMLHttpRequest): string | null {
  try {
    const body = JSON.parse(xhr.responseText) as { detail?: unknown }
    return typeof body.detail === 'string' ? body.detail : null
  } catch {
    return null
  }
}

/**
 * Ανέβασμα PDF με πρόοδο. Με XMLHttpRequest και όχι fetch: το fetch δεν αναφέρει πρόοδο αποστολής. Η επεξεργασία
 * (κείμενο, chunks) γίνεται μετά, στο παρασκήνιο του backend· η λίστα εγγράφων τη δείχνει ως «processing».
 */
export function uploadDocument(file: File, onProgress: (fraction: number) => void, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/upload`)
    const token = tokenStore.get()
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve()
      if (xhr.status === 401 && token) expireSession()
      reject(new ApiError(xhr.status, detailFromXhr(xhr)))
    }
    xhr.onerror = () => reject(new ApiError(0, null))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    signal?.addEventListener('abort', () => xhr.abort(), { once: true })
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}
