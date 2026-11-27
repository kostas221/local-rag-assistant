import { expect, test } from '@playwright/test'
import { FakeBackend } from './fakeBackend.ts'

test('on a phone, the source page opens over the whole width', async ({ page }) => {
  const api = new FakeBackend()
  await api.install(page)
  await page.goto('/')
  const box = page.getByRole('textbox', { name: 'Ask about your papers…' })
  await box.fill('What are the limits?')
  await expect(page.getByRole('button', { name: 'Send' })).toBeEnabled()
  await box.press('Enter')
  await page.getByRole('button', { name: /^Source 1:/ }).first().click()
  const panel = page.getByRole('region', { name: 'Source' })
  await expect(panel.getByText('The passage is highlighted in yellow.')).toBeVisible()
  const width = (await panel.boundingBox())?.width ?? 0
  expect(width).toBeGreaterThan((page.viewportSize()?.width ?? 0) - 4)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})
