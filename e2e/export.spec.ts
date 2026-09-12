/**
 * Export end to end: after a real feature-extraction call, the panel's Export button
 * produces a download (a Blob URL on a temporary `<a download>`, intercepted through
 * Playwright's `download` event) that parses to `{ version, exportedAt, events }`.
 * The persistent context accepts downloads (see `e2e/fixtures.ts`).
 */
import { readFileSync } from 'node:fs';
import { expect, runTask, test } from './fixtures';

const FILENAME = /^transformersjs-inspector-\d{8}-\d{6}\.json$/;

type Exported = { version: string; exportedAt: string; events: { type: string; callId?: string | null }[] };

test('Export downloads the bus history as versioned JSON', async ({ page }) => {
  await page.goto('/');
  await runTask(page, 'feature-extraction', 'Export me.');

  const panel = page.locator('[data-tjsi-panel]');
  await panel.locator('[data-action="toggle"]').click();
  await expect(panel.locator('[data-call]')).toHaveCount(1);

  const [download] = await Promise.all([page.waitForEvent('download'), panel.locator('[data-action="export"]').click()]);
  expect(download.suggestedFilename()).toMatch(FILENAME);
  await expect(panel.locator('[data-export-status]')).toHaveText('exported');

  const path = await download.path();
  expect(path).toBeTruthy();
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Exported;
  expect(Object.keys(parsed).sort()).toEqual(['events', 'exportedAt', 'version']);

  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { version: string };
  expect(parsed.version).toBe(pkg.version);
  await expect(page.locator('[data-version]')).toHaveText(parsed.version);
  expect(Number.isNaN(Date.parse(parsed.exportedAt))).toBe(false);
  expect(Math.abs(Date.parse(parsed.exportedAt) - Date.now())).toBeLessThan(5 * 60_000);

  const start = parsed.events.find((e) => e.type === 'call:start');
  expect(start?.callId).toBeTruthy();
  const ofCall = parsed.events.filter((e) => e.callId === start?.callId).map((e) => e.type);
  expect(ofCall).toEqual(['call:start', 'tokenize', 'run:start', 'run:end', 'result']);

  // The status text clears itself.
  await expect(panel.locator('[data-export-status]')).toHaveText('', { timeout: 5000 });
});
