import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

const backend = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The browser only talks to the Vite origin; /api is forwarded to FastAPI.
    proxy: { '/api': backend },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
