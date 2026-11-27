import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'
import { FakeBackend } from './fakeBackend.ts'

/** Προσβασιμότητα WCAG 2.1 A/AA στην τρέχουσα οθόνη. */
async function expectAccessible(page: Page) {
  const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
  const summary = result.violations.map((v) => `${v.id}: ${v.nodes.length} × ${v.nodes[0]?.target.join(' ')}`)
  expect(summary).toEqual([])
}

async function ask(page: Page, question: string) {
  const box = page.getByRole('textbox', { name: 'Ask about your papers…' })
  await box.fill(question)
  // Όπως ένας χρήστης: το Send ενεργοποιείται όταν φορτώσει η βιβλιοθήκη (χρειάζεται τσεκαρισμένο έγγραφο).
  await expect(page.getByRole('button', { name: 'Send' })).toBeEnabled()
  await box.press('Enter')
}

const chip = (page: Page, n: number) => page.getByRole('button', { name: new RegExp(`^Source ${n}:`) })
const viewer = (page: Page) => page.getByRole('region', { name: 'Source' })

test('signs in from the login page; both screens are accessible', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page, { signedIn: false })
  await page.goto('/')
  await expect(page).toHaveURL(/\/login$/)
  await expectAccessible(page)
  await page.getByLabel('Username').fill('tester')
  await page.getByLabel('Password').fill('not-a-real-password-1')
  await page.getByRole('button', { name: 'Sign in', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'How can I assist your research?' })).toBeVisible()
  await expectAccessible(page)
})

test('answers with citations; a citation opens the page with its passage, with one passage check per answer', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  await ask(page, 'What are the limits?')

  await expect(page.getByText('Functions can run for at most 900 seconds')).toBeVisible()
  await expect(page).toHaveURL(/\/c\/\d+$/)
  await expect(chip(page, 1).first()).toBeVisible()

  await chip(page, 1).first().click()
  await expect(viewer(page)).toBeVisible()
  await expect(viewer(page).getByText('The passage is highlighted in yellow.')).toBeVisible()
  expect(api.passagesCalls).toBe(1)
  expect(api.pageImages.some((r) => r.docId === 1 && r.page === 5 && r.quote === 'run time 900 seconds')).toBe(true)

  // Δεύτερο κλικ: τα χωρία είναι ήδη γνωστά, καμία νέα κλήση.
  await page.keyboard.press(']')
  await chip(page, 1).first().click()
  expect(api.passagesCalls).toBe(1)
  await expectAccessible(page)

  // Το Esc κλείνει πρώτα ό,τι είναι από πάνω (την κάρτα της παραπομπής κάτω από το ποντίκι), μετά το πάνελ.
  await page.mouse.move(5, 5)
  await expect(page.locator('[data-slot=hover-card-content]')).toHaveCount(0)
  await page.keyboard.press('Escape')
  await expect(viewer(page)).toBeHidden()
})

test('a passage found on another page: the panel says which page, and switches both ways', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  await ask(page, 'What are the limits?')
  await chip(page, 2).click() // [2] = S2, στην πρόταση 2, που το χωρίο της βρέθηκε στη σ. 7 του alpha.pdf

  const panel = viewer(page)
  await expect(panel.getByText('This sentence cites beta.pdf, p. 3, but its passage was found on this page.')).toBeVisible()
  await expect(panel.getByText('alpha.pdf', { exact: true })).toBeVisible()
  await panel.getByRole('button', { name: 'Open the cited page 3' }).click()
  await expect(panel.getByText('This is the page the sentence cites. Its passage was found on alpha.pdf, p. 7.')).toBeVisible()
  await panel.getByRole('button', { name: 'See the passage on page 7' }).click()
  await expect(panel.getByText('This sentence cites beta.pdf, p. 3, but its passage was found on this page.')).toBeVisible()
})

test('evidence view summarises where passages were found, never as verified', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  await ask(page, 'What are the limits?')
  await page.getByRole('button', { name: 'Show evidence' }).click()
  await expect(page.getByText('1 of 2 cited sentences have a passage on a cited page · 1 on another page')).toBeVisible()
  await expect(page.getByText(/verified/i)).toHaveCount(0)
  expect(api.passagesCalls).toBe(1)
})

test('compare only these asks for confirmation before a new model call', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  await ask(page, 'Compare the two papers')
  await page.getByRole('button', { name: 'Compare only these' }).click()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText('This is one new model call.')
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toBeHidden()
  expect(api.messages.filter((m) => m.role === 'user')).toHaveLength(1)
})

test('a refusal shows the server text, not an invented title', async ({ page }) => {
  const api = new FakeBackend()
  api.chats.push({ sources: [], chunks: ['The question refers to something it does not name. Which system or paper do you mean?'] })
  await api.install(page)
  await page.goto('/')
  await ask(page, 'How does it work?')
  await expect(page.getByText('Which system or paper do you mean?')).toBeVisible()
  await expect(page.getByText('Check which documents are checked in the library.')).toBeVisible()
  await expect(page.getByText('Not found in your documents')).toHaveCount(0)
})

test('Stop before any text leaves the question marked as unanswered', async ({ page }) => {
  const api = new FakeBackend()
  api.chats.push({ sources: [], chunks: [], delayMs: 8000, save: false })
  await api.install(page)
  await page.goto('/')
  await ask(page, 'A slow question')
  await expect(page.getByText('Searching your documents…')).toBeVisible()
  await page.getByRole('button', { name: 'Stop' }).click()
  await expect(page.getByText(/No answer was saved/)).toBeVisible()
  await page.getByRole('button', { name: 'Ask again' }).click()
  await expect(page.getByRole('textbox', { name: 'Ask about your papers…' })).toHaveValue('A slow question')
})

test('uploads a PDF and shows it as processing', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'gamma.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4 test') })
  await expect(page.getByText(/gamma\.pdf: Uploaded/)).toBeVisible()
  await expect(page.getByText('Processing…')).toBeVisible()
})

test('Ctrl+K opens the palette with conversations and documents', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  await ask(page, 'What are the limits?')
  await expect(page.getByText('Functions can run for at most 900 seconds')).toBeVisible()
  await page.keyboard.press('Control+k')
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText('What are the limits?')).toBeVisible()
  await expect(dialog.getByText('beta.pdf')).toBeVisible()
  await expectAccessible(page)
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
})
