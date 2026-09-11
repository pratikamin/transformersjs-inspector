import { defineConfig } from '@playwright/test';

const BASE_URL = 'http://localhost:5173';
const SCREENSHOTS = '**/screenshots.spec.ts';

export default defineConfig({
  testDir: 'e2e',
  /** The first run downloads a model from the Hub; later runs hit the persistent profile's cache. */
  timeout: 300_000,
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: { baseURL: BASE_URL, viewport: { width: 1280, height: 900 } },
  projects: [
    /** `npm run e2e`: the verification suite; never rewrites `docs/img/`. */
    { name: 'chromium', testIgnore: SCREENSHOTS },
    /** `npm run screenshots`: the README images, at device scale factor 2. */
    { name: 'screenshots', testMatch: SCREENSHOTS },
  ],
  webServer: {
    command: 'npm run dev',
    url: BASE_URL,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
