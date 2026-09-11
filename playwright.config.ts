import { defineConfig } from '@playwright/test';

const BASE_URL = 'http://localhost:5173';

export default defineConfig({
  testDir: 'e2e',
  /** The first run downloads a model from the Hub; later runs hit the persistent profile's cache. */
  timeout: 300_000,
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: { baseURL: BASE_URL },
  projects: [{ name: 'chromium' }],
  webServer: {
    command: 'npm run dev',
    url: BASE_URL,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
