/**
 * Shared Playwright fixtures. The browser context is a *persistent* Chromium profile under
 * `.cache/pw-profile`, so the Cache API entries Transformers.js writes for model files and
 * the HTTP cache for the CDN module survive between runs: the first run downloads, the rest
 * are fast and offline-tolerant. The built-in `page` fixture derives from `context`, so this
 * is the only override needed.
 */
import { fileURLToPath } from 'node:url';
import { test as base, chromium, expect } from '@playwright/test';
import type { Locator, Page } from '@playwright/test';

export const PROFILE_DIR = fileURLToPath(new URL('../.cache/pw-profile', import.meta.url));

export const test = base.extend({
  context: async ({ baseURL, viewport, deviceScaleFactor }, use) => {
    // `viewport` comes from playwright.config.ts (1280x900); a spec may `test.use()` both.
    const context = await chromium.launchPersistentContext(PROFILE_DIR, {
      headless: true,
      baseURL,
      viewport,
      deviceScaleFactor,
      // Playwright's default, stated: the Export button's Blob download must be accepted so
      // `page.waitForEvent('download')` yields a saved file (e2e/export.spec.ts).
      acceptDownloads: true,
    });
    await use(context);
    await context.close();
  },
  /** `E2E_DEBUG=1` echoes the page console and every Hub/CDN response with a timestamp. */
  page: async ({ page }, use) => {
    if (process.env.E2E_DEBUG) {
      const t0 = Date.now();
      const stamp = (): string => `[${((Date.now() - t0) / 1000).toFixed(1)}s]`;
      page.on('console', (m) => process.stdout.write(`${stamp()} console.${m.type()} ${m.text().slice(0, 300)}\n`));
      page.on('requestfailed', (r) => process.stdout.write(`${stamp()} FAILED ${r.url().slice(0, 140)} ${r.failure()?.errorText ?? ''}\n`));
      page.on('response', (r) => {
        const url = r.url();
        if (/huggingface|jsdelivr/.test(url)) process.stdout.write(`${stamp()} ${r.status()} ${url.slice(0, 140)}\n`);
      });
    }
    await use(page);
  },
});

export { expect };

/** Generous: the first run of a task downloads its model. */
export const TASK_TIMEOUT = 240_000;

/** Upper bound for the CDN import of Transformers.js on a cold profile; Run buttons are disabled until then. */
export const READY_TIMEOUT = 60_000;

/**
 * Waits for the section's Run button (shipped `disabled`, enabled by `demo/main.ts` once the
 * CDN import has resolved), fills the textarea when `text` is given (a section without one,
 * like image classification, takes no text), clicks Run and waits for
 * `[data-task=<task>][data-status="done"]`. Fails fast, quoting the page's progress line and
 * output, when the section reports `error` or never finishes.
 */
export async function runTask(page: Page, task: string, text?: string): Promise<Locator> {
  const section = page.locator(`[data-task="${task}"]`);
  const run = section.getByRole('button', { name: 'Run' });
  await expect(run, 'demo page did not finish loading Transformers.js').toBeEnabled({ timeout: READY_TIMEOUT });
  if (text !== undefined) await section.locator('textarea').fill(text);
  await run.click();
  const describe = async (): Promise<string> =>
    `status=${await section.getAttribute('data-status')} progress="${await page.locator('[data-progress]').textContent()}" output="${(await section.locator('[data-output]').textContent())?.slice(0, 300)}"`;
  await expect
    .poll(() => section.getAttribute('data-status'), { timeout: TASK_TIMEOUT, message: `task ${task} did not finish` })
    .toMatch(/^(done|error)$/)
    .catch(async (e: unknown) => {
      throw new Error(`task ${task} timed out: ${await describe()}`, { cause: e });
    });
  if ((await section.getAttribute('data-status')) !== 'done') throw new Error(`task ${task} failed: ${await describe()}`);
  await expect(page.locator(`[data-task="${task}"][data-status="done"]`)).toHaveCount(1);
  return section;
}

/**
 * Clicks the header's Simple / Detail segment for `view` and waits for it to be marked active.
 * The panel starts in the simple view (0.3.0), so specs that read tensor and top-k tables
 * switch to `'detail'` first; the click lands on the button, not the header, so the panel
 * stays open.
 */
export async function setView(panel: Locator, view: 'simple' | 'detail'): Promise<void> {
  const segment = panel.locator(`[data-action="view"][data-view="${view}"]`);
  await segment.click();
  await expect(segment).toHaveAttribute('aria-pressed', 'true');
}

/** `<section class="section">` of an expanded call whose `<h3>` is exactly `title`. */
export function sectionTitled(details: Locator, title: string): Locator {
  return details.locator('section.section').filter({ has: details.page().locator('h3', { hasText: new RegExp(`^${title}$`) }) });
}

/** The `[data-tensor]` row named `name` inside `table.tensors`. */
export function tensorRow(scope: Locator, name: string): Locator {
  return scope.locator('table.tensors tr[data-tensor]').filter({ has: scope.page().locator('td.name', { hasText: new RegExp(`^${name}$`) }) });
}

/** The dims cell of a tensor row (third column). */
export const dimsOf = (row: Locator): Locator => row.locator('td').nth(2);
