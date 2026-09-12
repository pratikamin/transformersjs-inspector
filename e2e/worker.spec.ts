/**
 * The worker bridge end to end: `demo/worker.html` never loads Transformers.js itself. A
 * module worker builds `Xenova/all-MiniLM-L6-v2`, calls `attach(pipe, { bus, panel: false })`
 * and `exposeToPage(bus)`; the page's `connectWorker(worker)` bus feeds an ordinary
 * `mountPanel`. The row, its sections and — through `request('tensor')` relayed back to the
 * worker's `TensorStore` — the full `last_hidden_state` values must all show up on the page.
 * Panel selectors resolve through the open shadow root of `[data-tjsi-panel]`.
 */
import type { Locator } from '@playwright/test';
import { READY_TIMEOUT, TASK_TIMEOUT, expect, setView, test } from './fixtures';

const SENTENCE = 'The inspector sees everything.';

/** `<section class="section">` whose `<h3>` is exactly `title`. */
function sectionTitled(details: Locator, title: string): Locator {
  return details.locator('section.section').filter({ has: details.page().locator('h3', { hasText: new RegExp(`^${title}$`) }) });
}

function tensorRow(scope: Locator, name: string): Locator {
  return scope.locator('table.tensors tr[data-tensor]').filter({ has: scope.page().locator('td.name', { hasText: new RegExp(`^${name}$`) }) });
}

const dimsOf = (row: Locator): Locator => row.locator('td').nth(2);

test('worker: a pipeline attached inside a module worker fills the page panel, and Load values fetches from the worker', async ({ page }) => {
  await page.goto('/worker.html');
  const main = page.locator('main[data-status]');
  const run = page.getByRole('button', { name: 'Run' });
  // Shipped disabled; the worker enables it by posting { ready } once its CDN import resolved.
  await expect(run, 'worker did not finish loading Transformers.js').toBeEnabled({ timeout: READY_TIMEOUT });
  await expect(main).toHaveAttribute('data-status', 'idle');

  await page.locator('textarea').fill(SENTENCE);
  await run.click();
  const describe = async (): Promise<string> =>
    `status=${await main.getAttribute('data-status')} progress="${await page.locator('[data-progress]').textContent()}" output="${(await page.locator('[data-output]').textContent())?.slice(0, 300)}"`;
  await expect
    .poll(() => main.getAttribute('data-status'), { timeout: TASK_TIMEOUT, message: 'worker run did not finish' })
    .toMatch(/^(done|error)$/)
    .catch(async (e: unknown) => {
      throw new Error(`worker run timed out: ${await describe()}`, { cause: e });
    });
  if ((await main.getAttribute('data-status')) !== 'done') throw new Error(`worker run failed: ${await describe()}`);
  await expect(page.locator('[data-output]')).toContainText('dims [1, 384]');

  // The page holds no pipeline, no default bus and no default store: nothing here could
  // answer a tensor request locally, so whatever the panel renders came across the worker port.
  const pageSide = await page.evaluate(() => {
    const g = (globalThis as unknown as Record<symbol, { bus?: unknown; store?: unknown } | undefined>)[Symbol.for('transformersjs-inspector')];
    return { defaultBus: g?.bus !== undefined, defaultStore: g?.store !== undefined };
  });
  expect(pageSide).toEqual({ defaultBus: false, defaultStore: false });

  // mountPanel(bus, { open: true }): already expanded, one row relayed from the worker.
  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel).toHaveCount(1);
  await expect(panel.locator('[data-badge]')).toHaveText('1');
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  await expect(rows.first().locator('.label')).toContainText('feature-extraction');
  await expect(rows.first().locator('.excerpt')).toContainText(SENTENCE);

  await setView(panel, 'detail'); // tensor rows are detail-view content
  await rows.first().click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(1);

  // The full wrapper set ran in the worker: tokenizer chips and the session run both arrived.
  const chipTexts = await sectionTitled(details, 'Tokenizer').locator('.chip .chip-str').allTextContents();
  expect(chipTexts).toContain('[CLS]');
  const n = chipTexts.length;
  expect(n).toBeGreaterThan(2);
  const runs = sectionTitled(details, 'Session runs').locator('.run');
  await expect(runs).toHaveCount(1);
  const hidden = tensorRow(runs.first(), 'last_hidden_state');
  await expect(hidden).toHaveCount(1);
  await expect(dimsOf(hidden)).toHaveText(`[1, ${n}, 384]`);

  // Load values: request('tensor') leaves the page bus, is answered by the worker's store,
  // and the Float32Array survives the structured clone back.
  await expect(runs.first().locator('tr.values')).toHaveCount(0);
  await hidden.locator('button[data-action="load"]').click();
  const valuesRow = runs.first().locator(`tr.values[data-values="${await hidden.getAttribute('data-tensor')}"]`);
  await expect(valuesRow.locator('.meta')).toContainText(`float32 [1, ${n}, 384] · ${n * 384} values`);
  await expect(valuesRow.locator('[data-values-error]')).toHaveCount(0);
  const values = (await valuesRow.locator('.values-list').innerText()).split(',').map((s) => Number(s.trim()));
  expect(values.length).toBeGreaterThanOrEqual(384);
  expect(values.every((v) => Number.isFinite(v))).toBe(true);
});
