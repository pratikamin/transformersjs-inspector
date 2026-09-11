/**
 * Zero-touch end to end: `demo/preload.html` loads `src/preload.ts` as its first module
 * script and then an *unmodified* Transformers.js host (no `attach()`, only
 * `device: 'auto'`) that runs one `Xenova/all-MiniLM-L6-v2` call. The only way the panel
 * gets a row is if the ORT symbol was set before Transformers.js evaluated and the shim's
 * `InferenceSession.create` wrapped the session — that ordering is what this spec proves.
 * Panel selectors resolve through the open shadow root of `[data-tjsi-panel]`.
 */
import type { Locator } from '@playwright/test';
import { expect, test, TASK_TIMEOUT } from './fixtures';

/**
 * The browser hands `InferenceSession.create` a `Uint8Array`, never a path, so `nameFor`
 * falls back to `session#<n> → <first output>`; the reducer prefixes orphan runs with `direct ·`.
 */
const ROW_LABEL = 'direct · session#1 → last_hidden_state';

/** `<section class="section">` whose `<h3>` is exactly `title`. */
function sectionTitled(details: Locator, title: string): Locator {
  return details.locator('section.section').filter({ has: details.page().locator('h3', { hasText: new RegExp(`^${title}$`) }) });
}

function tensorRow(scope: Locator, name: string): Locator {
  return scope.locator('table.tensors tr[data-tensor]').filter({ has: scope.page().locator('td.name', { hasText: new RegExp(`^${name}$`) }) });
}

const dimsOf = (row: Locator): Locator => row.locator('td').nth(2);

test('preload: an attach()-free host page yields one direct session row with last_hidden_state [1, N, 384]', async ({ page }) => {
  await page.goto('/preload.html');
  const main = page.locator('main[data-status]');
  const describe = async (): Promise<string> =>
    `status=${await main.getAttribute('data-status')} progress="${await page.locator('[data-progress]').textContent()}" output="${(await page.locator('[data-output]').textContent())?.slice(0, 300)}"`;
  await expect
    .poll(() => main.getAttribute('data-status'), { timeout: TASK_TIMEOUT, message: 'preload host did not finish' })
    .toMatch(/^(done|error)$/)
    .catch(async (e: unknown) => {
      throw new Error(`preload host timed out: ${await describe()}`, { cause: e });
    });
  if ((await main.getAttribute('data-status')) !== 'done') throw new Error(`preload host failed: ${await describe()}`);
  await expect(page.locator('[data-output]')).toContainText('dims [1, 384]');

  // The shim is installed under the key Transformers.js reads, and it is ours.
  const installed = await page.evaluate(() => {
    const ort = (globalThis as unknown as Record<symbol, unknown>)[Symbol.for('onnxruntime')];
    return {
      defined: ort !== undefined,
      marked: typeof ort === 'object' && ort !== null && (ort as Record<symbol, unknown>)[Symbol.for('transformersjs-inspector.preload')] === true,
      hasCreate: typeof (ort as { InferenceSession?: { create?: unknown } })?.InferenceSession?.create === 'function',
    };
  });
  expect(installed).toEqual({ defined: true, marked: true, hasCreate: true });

  // No attach() anywhere on the page, yet the preload mounted the panel and it counted the run.
  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel).toHaveCount(1);
  await expect(panel.locator('[data-badge]')).toHaveText('1');

  await panel.locator('[data-action="toggle"]').click();
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  await expect(rows.first().locator('.label')).toHaveText(ROW_LABEL);

  await rows.first().click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(1);
  // No tokenizer or pipeline wrapper on this path: only the session boundary is visible.
  await expect(sectionTitled(details, 'Tokenizer')).toHaveCount(0);
  const runs = sectionTitled(details, 'Session runs').locator('.run');
  await expect(runs).toHaveCount(1);
  const run = runs.first();
  const inputDims = await dimsOf(tensorRow(run, 'input_ids')).textContent();
  const n = Number(/^\[1, (\d+)\]$/.exec(inputDims ?? '')?.[1]);
  expect(n).toBeGreaterThan(2);
  await expect(dimsOf(tensorRow(run, 'attention_mask'))).toHaveText(`[1, ${n}]`);
  await expect(dimsOf(tensorRow(run, 'last_hidden_state'))).toHaveText(`[1, ${n}, 384]`);
});
