import { expect, test } from '@playwright/test'
import { FakeBackend, SOURCES } from './fakeBackend.ts'

/**
 * Το σύμπτωμα που είδε ο χρήστης (27/11/2026): το Gemini στέλνει την απάντηση σε δύο μεγάλα κομμάτια, και η οθόνη
 * «κολλούσε» και μετά πετούσε μισή απάντηση μονομιάς. Εδώ η ροή έρχεται ακριβώς έτσι (κομμάτι, παύση 1,5 s, κομμάτι)
 * και μετράμε κάθε 100 ms πόσο κείμενο φαίνεται: πρέπει να μεγαλώνει σταδιακά, χωρίς άλματα.
 */
const HALF = 'Serverless functions scale automatically with demand and bill per request, '.repeat(8) // ~600 χαρακτήρες

test('a long answer arriving in two big pieces appears gradually, not in jumps', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.addInitScript(
    ({ sources, half }) => {
      const original = window.fetch.bind(window)
      window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
        if (!url.endsWith('/api/chat')) return original(input, init)
        const enc = new TextEncoder()
        const line = (o: unknown) => enc.encode(`${JSON.stringify(o)}\n`)
        const body = new ReadableStream<Uint8Array>({
          async start(controller) {
            controller.enqueue(line({ type: 'sources', data: sources }))
            controller.enqueue(line({ type: 'text', data: half }))
            await new Promise((r) => setTimeout(r, 1500))
            controller.enqueue(line({ type: 'text', data: `${half}The end [S1].` }))
            controller.close()
          },
        })
        return new Response(body, { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } })
      }
    },
    { sources: SOURCES, half: HALF },
  )
  await page.goto('/')
  const box = page.getByRole('textbox', { name: 'Ask about your papers…' })
  await box.fill('Tell me about serverless')
  await expect(page.getByRole('button', { name: 'Send' })).toBeEnabled()
  await box.press('Enter')

  const lengths = await page.evaluate(async () => {
    const out: number[] = []
    for (let i = 0; i < 40; i++) {
      out.push(document.querySelector('.prose-answer')?.textContent?.length ?? 0)
      await new Promise((r) => setTimeout(r, 100))
    }
    return out
  })
  const total = HALF.length * 2 + 'The end 1.'.length
  const jumps = lengths.slice(1).map((n, i) => n - (lengths[i] ?? 0))
  // Σταδιακά: κανένα βήμα 100 ms δεν δείχνει πάνω από ~1/4 του κομματιού (χωρίς εξομάλυνση θα ήταν ~600 μονομιάς).
  expect(Math.max(...jumps)).toBeLessThan(HALF.length / 4)
  // Και φτάνει ως το τέλος.
  expect(Math.max(...lengths)).toBeGreaterThan(total - 20)
})
