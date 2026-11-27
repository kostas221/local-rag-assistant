import { defineConfig, devices } from '@playwright/test'

/**
 * Δοκιμές από άκρη σε άκρη (Φάση 3) πάνω στο build της παραγωγής, με ΨΕΥΤΙΚΟ API μέσα στον browser (e2e/fakeBackend):
 * χωρίς backend, χωρίς Gemini, χωρίς κόστος. Ελέγχουν ροές και προσβασιμότητα (axe), όχι την ποιότητα των απαντήσεων.
 *
 * Τοπικά τρέχουν με τον Chrome που είναι ήδη εγκατεστημένος (χωρίς κατέβασμα browser)· στο CI με το Chromium του
 * Playwright.
 */
const CI = Boolean(process.env.CI)
const channel = CI ? undefined : 'chrome'
const PORT = 4173

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  // Λίγοι παράλληλοι browsers: στο laptop (5,9 GB RAM) τέσσερις Chrome μαζί έκαναν τα tests να λήγουν από φόρτο.
  workers: 2,
  reporter: CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], channel, viewport: { width: 1440, height: 900 } }, testIgnore: /mobile/ },
    { name: 'mobile', use: { ...devices['Pixel 7'], channel }, testMatch: /mobile/ },
  ],
  webServer: {
    command: `npm run build && npx vite preview --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !CI,
    timeout: 180_000,
  },
})
