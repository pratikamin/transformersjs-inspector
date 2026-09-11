/**
 * README images. Not part of `npm run e2e` (the `chromium` project ignores this file); run
 * `npm run screenshots`, which uses the `screenshots` project. Each test drives the demo the
 * way `demo.spec.ts` / `generation.spec.ts` do, expands the call row, scrolls the section of
 * interest to the top of the panel body and takes an element screenshot of the panel host
 * (`[data-tjsi-panel]`, fixed bottom-right, 560 px wide) at device scale factor 2, so the
 * PNGs are 1120 px wide and stay readable when the README scales them down.
 */
import { fileURLToPath } from 'node:url';
import type { Locator, Page } from '@playwright/test';
import { expect, runTask, test } from './fixtures';

const IMG_DIR = new URL('../docs/img/', import.meta.url);
const imgPath = (name: string): string => fileURLToPath(new URL(name, IMG_DIR));

// A taller viewport lets the panel (max-height: 100vh - 32px) show more of the details.
test.use({ deviceScaleFactor: 2, viewport: { width: 1280, height: 1100 } });

/** `<section class="section">` whose `<h3>` is exactly `title`. */
function sectionTitled(details: Locator, title: string): Locator {
  return details.locator('section.section').filter({ has: details.page().locator('h3', { hasText: new RegExp(`^${title}$`) }) });
}

/** Opens the panel, expands the only call row and returns its details block. */
async function expandOnlyRow(page: Page): Promise<{ panel: Locator; details: Locator }> {
  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel).toHaveCount(1);
  await expect(panel.locator('[data-badge]')).toHaveText('1');
  await panel.locator('[data-action="toggle"]').click();
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  await rows.first().click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(1);
  return { panel, details };
}

/** Scrolls the panel body so `target` sits at its top, then screenshots the whole panel host. */
async function shoot(panel: Locator, target: Locator, file: string): Promise<void> {
  await target.evaluate((el) => el.scrollIntoView({ block: 'start' }));
  await panel.page().waitForTimeout(100); // let the scroll and any bar widths settle
  await panel.screenshot({ path: imgPath(file), animations: 'disabled' });
}

test('panel-embedding.png: feature extraction with tokens, session tensors and loaded values', async ({ page }) => {
  await page.goto('/');
  await runTask(page, 'feature-extraction', 'The inspector sees everything.');
  const { panel, details } = await expandOnlyRow(page);

  const run = sectionTitled(details, 'Session runs').locator('.run').first();
  const hidden = run.locator('table.tensors tr[data-tensor]').filter({ has: page.locator('td.name', { hasText: /^last_hidden_state$/ }) });
  await hidden.locator('button[data-action="load"]').click();
  const valuesRow = run.locator('tr.values');
  await expect(valuesRow.locator('.values-list')).toContainText(',');
  await expect(valuesRow.locator('[data-values-error]')).toHaveCount(0);

  await shoot(panel, sectionTitled(details, 'Tokenizer'), 'panel-embedding.png');
});

test('panel-generation.png: text generation with per-step top-k', async ({ page }) => {
  await page.goto('/');
  await runTask(page, 'text-generation', 'hi');
  const { panel, details } = await expandOnlyRow(page);

  const generation = sectionTitled(details, 'Generation');
  await expect(generation.locator('.step')).toHaveCount(3);
  await expect(generation.locator('table.topk tbody tr.picked')).toHaveCount(3);

  await shoot(panel, generation, 'panel-generation.png');
});
