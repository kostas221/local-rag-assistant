/// <reference types="vitest/config" />
import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Ο backend (FastAPI) τρέχει στο Docker στο 127.0.0.1:8010. Στην ανάπτυξη όλα τα αιτήματα πάνε στο /api του ίδιου
// origin και το proxy τα προωθεί, αφαιρώντας το /api (οι διαδρομές του backend δεν το έχουν). Ίδιο origin = κανένα
// θέμα CORS, ίδια συμπεριφορά με το nginx της παραγωγής (Φάση 3).
const BACKEND = process.env.BACKEND_URL ?? 'http://127.0.0.1:8010'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
})
