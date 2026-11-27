/**
 * Ψεύτικος backend για τα e2e tests: απαντά στα /api/* μέσα στον browser (page.route), με τα ίδια σχήματα που
 * γυρίζει το `backend/main.py`. Κρατά κατάσταση (συνομιλίες, μηνύματα, κλήσεις), ώστε τα tests να ελέγχουν και ΠΟΣΕΣ
 * κλήσεις έγιναν, π.χ. ότι τα χωρία ζητούνται μία φορά ανά απάντηση.
 *
 * Τα κείμενα είναι γραμμένα για τα tests, όχι από άρθρα (το repo είναι δημόσιο).
 */
import type { Page, Route } from '@playwright/test'

// PNG 1x1, αρκετό για να φανεί ότι η εικόνα της σελίδας ήρθε.
const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
)

export function fakeToken(sub = 'tester'): string {
  const part = (o: object) => Buffer.from(JSON.stringify(o)).toString('base64url')
  return `${part({ alg: 'HS256', typ: 'JWT' })}.${part({ sub })}.signature`
}

export const ANSWER = 'Functions can run for at most 900 seconds [S1]. Memory goes up to 3 GiB [S1, S2].'

export const SOURCES = [
  { label: 'S1', file: 'alpha.pdf', page: 5, doc_id: 1, preview: 'search match text' },
  { label: 'S2', file: 'beta.pdf', page: 3, doc_id: 2, preview: 'search match text' },
  { label: 'S3', file: 'alpha.pdf', page: 7, doc_id: 1, preview: 'search match text' },
]

/** Πρόταση 1: χωρίο στη σελίδα που παραθέτει. Πρόταση 2: χωρίο σε σελίδα που ΔΕΝ παραθέτει (λάθος σελίδα). */
export const PASSAGES = {
  status: 'done',
  items: [
    { i: 1, sentence: 'Functions can run for at most 900 seconds.', labels: ['S1'], status: 'found', quote: 'run time 900 seconds', label: 'S1', file: 'alpha.pdf', page: 5 },
    { i: 2, sentence: 'Memory goes up to 3 GiB.', labels: ['S1', 'S2'], status: 'found_elsewhere', quote: 'up to 3 GiB', label: 'S3', file: 'alpha.pdf', page: 7 },
  ],
}

export interface FakeChat {
  sources: object[]
  chunks: string[]
  /** Καθυστέρηση πριν από την απάντηση (για το Stop). */
  delayMs?: number
  /** Ο backend σώζει την απάντηση μόνο αν γράφτηκε κείμενο· με Stop νωρίς δεν σώζει τίποτα. */
  save?: boolean
}

interface FakeMessage {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string
  sources: string | null
  passages: string | null
}

export class FakeBackend {
  documents: object[] = [
    { id: 1, file_name: 'alpha.pdf', is_mine: true, uploader_name: 'tester', status: 'ready', pages_total: 12, pages_text: 12, extraction: 'Text extracted from 12/12 pages' },
    { id: 2, file_name: 'beta.pdf', is_mine: true, uploader_name: 'tester', status: 'ready', pages_total: 8, pages_text: 8, extraction: 'Text extracted from 8/8 pages' },
  ]
  conversations: { id: number; title: string; created_at: string }[] = []
  messages: FakeMessage[] = []
  chats: FakeChat[] = []
  passagesCalls = 0
  pageImages: { docId: number; page: number; quote: string | null }[] = []
  private nextId = 100

  async install(page: Page, opts: { signedIn?: boolean } = { signedIn: true }) {
    if (opts.signedIn !== false) await page.addInitScript((t) => sessionStorage.setItem('zai.token', t), fakeToken())
    await page.route('**/api/**', (route) => this.handle(route))
  }

  private async handle(route: Route) {
    const req = route.request()
    const path = new URL(req.url()).pathname.replace(/^\/api/, '')
    const method = req.method()
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    let m: RegExpMatchArray | null

    if (method === 'POST' && path === '/login') return json({ access_token: fakeToken(), token_type: 'bearer' })
    if (method === 'GET' && path === '/documents') return json(this.documents)
    if (method === 'POST' && path === '/upload') {
      const id = this.nextId++
      this.documents.push({ id, file_name: 'gamma.pdf', is_mine: true, uploader_name: 'tester', status: 'processing' })
      return json({ message: 'ok' })
    }
    if (method === 'GET' && path === '/conversations') return json([...this.conversations].reverse())
    if (method === 'POST' && path === '/conversations') {
      const conv = { id: this.nextId++, title: (req.postDataJSON() as { title: string }).title, created_at: new Date().toISOString() }
      this.conversations.push(conv)
      return json(conv)
    }
    if (method === 'GET' && (m = path.match(/^\/conversations\/(\d+)\/messages$/)))
      return json(this.messages.filter((x) => x.conversation_id === Number(m?.[1])))
    if (method === 'POST' && path === '/chat') return this.chat(route)
    if (method === 'POST' && /^\/messages\/\d+\/passages$/.test(path)) {
      this.passagesCalls++
      return json(PASSAGES)
    }
    if (method === 'POST' && (m = path.match(/^\/documents\/(\d+)\/page-image$/))) {
      const body = req.postDataJSON() as { page: number; quote: string | null }
      this.pageImages.push({ docId: Number(m[1]), page: body.page, quote: body.quote })
      return route.fulfill({
        status: 200,
        contentType: 'image/png',
        headers: { 'X-Highlight': body.quote ? 'highlighted' : 'none', 'Cache-Control': 'private, no-store' },
        body: PNG,
      })
    }
    if (method === 'POST' && path === '/feedback') return json({ status: 'success' })
    return json({ detail: `not mocked: ${method} ${path}` }, 404)
  }

  private async chat(route: Route) {
    const { conversation_id, question } = route.request().postDataJSON() as { conversation_id: number; question: string }
    // Όπως ο πραγματικός backend: η ερώτηση σώζεται ΠΡΙΝ από την απάντηση.
    this.messages.push({ id: this.nextId++, conversation_id, role: 'user', content: question, sources: null, passages: null })
    const reply = this.chats.shift() ?? { sources: SOURCES, chunks: [ANSWER] }
    if (reply.delayMs) await new Promise((r) => setTimeout(r, reply.delayMs))
    const text = reply.chunks.join('')
    if (reply.save !== false && text)
      this.messages.push({
        id: this.nextId++,
        conversation_id,
        role: 'assistant',
        content: text,
        sources: JSON.stringify(reply.sources),
        passages: null,
      })
    const lines = [
      { type: 'sources', data: reply.sources },
      ...reply.chunks.map((c) => ({ type: 'text', data: c })),
      { type: 'metrics', data: { retrieval_s: 0.42, generation_s: 1.3, pages: reply.sources.length, prompt_tokens: 1200, completion_tokens: 80 } },
    ]
    try {
      await route.fulfill({ status: 200, contentType: 'application/x-ndjson', body: lines.map((l) => JSON.stringify(l)).join('\n') + '\n' })
    } catch {
      // Ο browser έκλεισε τη σύνδεση (Stop) πριν από την απάντηση.
    }
  }
}
