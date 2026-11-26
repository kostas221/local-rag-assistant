/**
 * Τα σχήματα των δεδομένων που γυρίζει ο backend (FastAPI, `backend/main.py`).
 *
 * Γιατί zod και όχι τύποι από το OpenAPI: τα endpoints δεν δηλώνουν `response_model`, άρα το OpenAPI περιγράφει μόνο
 * τα αιτήματα. Εδώ τα σχήματα γράφονται μία φορά και ελέγχονται εκεί όπου μπαίνουν τα δεδομένα: αν ο backend αλλάξει
 * σχήμα, το λάθος φαίνεται σε ένα σημείο με σαφές μήνυμα, όχι ως σπασμένη οθόνη.
 */
import { z } from 'zod'

/** Μία πηγή της απάντησης: μία σελίδα ενός εγγράφου. Η ετικέτα (S3) είναι αυτή που γράφει η απάντηση. */
export const SourceSchema = z.looseObject({
  label: z.string().optional(),
  file: z.string(),
  page: z.number().int().nullable().optional(),
  doc_id: z.number().int().nullable().optional(),
  preview: z.string().nullable().optional(),
  second_search: z.boolean().optional(),
  full_text: z.boolean().optional(),
})
export type Source = z.infer<typeof SourceSchema>

export const DocumentSchema = z.looseObject({
  id: z.number().int(),
  file_name: z.string(),
  is_mine: z.boolean(),
  uploader_name: z.string().optional(),
  status: z.string(),
  pages_total: z.number().int().nullable().optional(),
  pages_text: z.number().int().nullable().optional(),
  pages_empty: z.array(z.number().int()).nullable().optional(),
  pages_scanned: z.array(z.number().int()).nullable().optional(),
  extraction: z.string().nullable().optional(),
})
export type LibraryDocument = z.infer<typeof DocumentSchema>

export const ConversationSchema = z.looseObject({
  id: z.number().int(),
  title: z.string().nullable(),
  created_at: z.string().nullable().optional(),
})
export type Conversation = z.infer<typeof ConversationSchema>

/** Το χωρίο μιας πρότασης (βήμα 6). Ποτέ «verified»: το χωρίο δείχνει πού να κοιτάξεις, δεν αποδεικνύει. */
export const PassageItemSchema = z.looseObject({
  i: z.number().int(),
  sentence: z.string(),
  labels: z.array(z.string()).default([]),
  status: z.enum(['found', 'found_elsewhere', 'not_found', 'no_citation', 'absence']),
  quote: z.string().nullable().optional(),
  label: z.string().nullable().optional(),
  file: z.string().nullable().optional(),
  page: z.number().int().nullable().optional(),
})
export type PassageItem = z.infer<typeof PassageItemSchema>

export const PassagesSchema = z.looseObject({
  status: z.enum(['done', 'no_pages', 'failed']),
  items: z.array(PassageItemSchema).default([]),
})
export type Passages = z.infer<typeof PassagesSchema>

function parseJsonText(value: unknown): unknown {
  if (typeof value !== 'string') return value
  try {
    return JSON.parse(value)
  } catch {
    return undefined
  }
}

/**
 * Ένα μήνυμα του ιστορικού. Ο backend σώζει πηγές και χωρία ως ΚΕΙΜΕΝΟ JSON μέσα στο μήνυμα, οπότε εδώ
 * μετατρέπονται. Αν κάτι είναι χαλασμένο (ή πολύ παλιό μήνυμα), η απάντηση φαίνεται χωρίς πηγές αντί να σπάσει η
 * οθόνη. Πολύ παλιές πηγές ήταν σκέτα ονόματα αρχείων: γίνονται πηγή χωρίς σελίδα.
 */
export const MessageSchema = z.looseObject({
  id: z.number().int(),
  conversation_id: z.number().int().optional(),
  role: z.string(),
  content: z
    .string()
    .nullable()
    .transform((v) => v ?? ''),
  sources: z
    .preprocess(
      (v) => parseJsonText(v) ?? [],
      z.array(z.union([SourceSchema, z.string().transform((file): Source => ({ file }))])),
    )
    .catch([]),
  passages: z.preprocess((v) => parseJsonText(v) ?? null, PassagesSchema.nullable()).catch(null),
})
export type Message = z.infer<typeof MessageSchema>

export const MetricsSchema = z.looseObject({
  retrieval_s: z.number().optional(),
  generation_s: z.number().optional(),
  pages: z.number().optional(),
  prompt_tokens: z.number().nullable().optional(),
  completion_tokens: z.number().nullable().optional(),
})
export type Metrics = z.infer<typeof MetricsSchema>

/** Τα γεγονότα του streaming του /chat (NDJSON: ένα JSON ανά γραμμή). */
export const StreamEventSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('sources'), data: z.array(SourceSchema) }),
  z.object({ type: z.literal('text'), data: z.string() }),
  z.object({ type: z.literal('status'), data: z.string(), reason: z.string().nullable().optional() }),
  z.object({ type: z.literal('metrics'), data: MetricsSchema }),
])
export type StreamEvent = z.infer<typeof StreamEventSchema>

export const LoginResponseSchema = z.looseObject({
  access_token: z.string(),
})

/** Η κατάσταση του κίτρινου στην εικόνα της σελίδας (header X-Highlight του /documents/{id}/page-image). */
export type HighlightResult = 'highlighted' | 'ambiguous' | 'not_found' | 'none'

export function parseHighlight(value: string | null): HighlightResult {
  return value === 'highlighted' || value === 'ambiguous' || value === 'not_found' ? value : 'none'
}

export const PERSONAS = ['Researcher', 'Educator', 'Concise'] as const
export type Persona = (typeof PERSONAS)[number]
