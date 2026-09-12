/**
 * Feature extraction end to end: a real `Xenova/all-MiniLM-L6-v2` call through the CDN
 * build of Transformers.js, observed by `attach()` and rendered by the panel. Every panel
 * selector below resolves through the open shadow root of `[data-tjsi-panel]` (Playwright's
 * CSS engine pierces open shadow roots).
 */
import type { Locator } from '@playwright/test';
import { expect, runTask, setView, test } from './fixtures';

const SENTENCE = 'The inspector sees everything.';

/** `<section class="section">` whose `<h3>` is exactly `title`. */
function sectionTitled(details: Locator, title: string): Locator {
  return details.locator('section.section').filter({ has: details.page().locator('h3', { hasText: new RegExp(`^${title}$`) }) });
}

/** The `[data-tensor]` row named `name` inside `table.tensors`; the dims cell is the third column. */
function tensorRow(scope: Locator, name: string): Locator {
  return scope.locator('table.tensors tr[data-tensor]').filter({ has: scope.page().locator('td.name', { hasText: new RegExp(`^${name}$`) }) });
}

const dimsOf = (row: Locator): Locator => row.locator('td').nth(2);

test('feature extraction: simple view first (Model line, embedding line, no tables), then tokens, session tensors, lazy values and a $tensor result in detail', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('[data-task="feature-extraction"]')).toHaveAttribute('data-status', 'idle');

  await runTask(page, 'feature-extraction', SENTENCE);
  await expect(page.locator('[data-task="feature-extraction"] [data-output]')).toContainText('dims [1, 384]');

  // The panel was auto-mounted by attach(); collapsed, it only counts.
  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel).toHaveCount(1);
  await expect(panel.locator('[data-badge]')).toHaveText('1');

  await panel.locator('[data-action="toggle"]').click();
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  await expect(rows.first().locator('.label')).toContainText('feature-extraction');
  await expect(rows.first().locator('.excerpt')).toContainText(SENTENCE);

  await rows.first().click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(1);

  // Simple view (the default): one Model line instead of the tensor tables, text-only chips,
  // the pooled [1, 384] result as one embedding line, and no tensor rows or JSON anywhere.
  await expect(panel.locator('[data-action="view"][data-view="simple"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(details).toHaveAttribute('data-view', 'simple');
  await expect(sectionTitled(details, 'Model').locator('[data-model-line]')).toHaveText(/^1 model run · \d+(\.\d+)? ms$/);
  await expect(sectionTitled(details, 'Session runs')).toHaveCount(0);
  await expect(details.locator('table')).toHaveCount(0);
  await expect(details.locator('[data-tensor]')).toHaveCount(0);
  await expect(details.locator('button[data-action="load"]')).toHaveCount(0);
  await expect(sectionTitled(details, 'Tokenizer').locator('.chip.chip-simple').first()).toBeVisible();
  await expect(sectionTitled(details, 'Tokenizer').locator('.chip-id')).toHaveCount(0);
  await expect(sectionTitled(details, 'Result').locator('[data-result-summary="tensor"]')).toHaveText('embedding · float32 · 384 values');
  await expect(sectionTitled(details, 'Result').locator('pre')).toHaveCount(0);

  // Detail view: the row stays expanded and the tables appear.
  await setView(panel, 'detail');
  await expect(details).toHaveCount(1);
  await expect(details).toHaveAttribute('data-view', 'detail');

  // Tokenizer: BERT wraps the sentence in [CLS] … [SEP].
  const chips = sectionTitled(details, 'Tokenizer').locator('.chip .chip-str');
  const chipTexts = await chips.allTextContents();
  expect(chipTexts).toContain('[CLS]');
  expect(chipTexts).toContain('[SEP]');
  const n = chipTexts.length;
  expect(n).toBeGreaterThan(2);
  // v1.1: the chip shows the decoded text and keeps the vocab string in its title.
  const clsChip = sectionTitled(details, 'Tokenizer').locator('.chip').filter({ has: details.page().locator('.chip-str', { hasText: /^\[CLS\]$/ }) });
  await expect(clsChip.first()).toHaveAttribute('title', /raw \[CLS\]/);

  // Session runs: one run with the three encoder inputs [1, N] and last_hidden_state [1, N, 384].
  const runs = sectionTitled(details, 'Session runs').locator('.run');
  await expect(runs).toHaveCount(1);
  const run = runs.first();
  for (const name of ['input_ids', 'attention_mask', 'token_type_ids']) {
    await expect(dimsOf(tensorRow(run, name))).toHaveText(`[1, ${n}]`);
  }
  const hidden = tensorRow(run, 'last_hidden_state');
  await expect(dimsOf(hidden)).toHaveText(`[1, ${n}, 384]`);

  // Lazy values: nothing rendered until the button is clicked, then all N * 384 floats.
  await expect(run.locator('tr.values')).toHaveCount(0);
  await hidden.locator('button[data-action="load"]').click();
  const valuesRow = run.locator(`tr.values[data-values="${await hidden.getAttribute('data-tensor')}"]`);
  await expect(valuesRow.locator('.meta')).toContainText(`float32 [1, ${n}, 384] · ${n * 384} values`);
  await expect(valuesRow.locator('[data-values-error]')).toHaveCount(0);
  const values = (await valuesRow.locator('.values-list').innerText()).split(',').map((s) => Number(s.trim()));
  expect(values.length).toBeGreaterThanOrEqual(384);
  expect(values.every((v) => Number.isFinite(v))).toBe(true);

  // Result: the pipeline returned a pooled, normalised [1, 384] tensor, shown as a $tensor marker.
  const result = sectionTitled(details, 'Result');
  await expect(result.locator('pre')).toContainText('"$tensor"');
  const pooled = result.locator('table.tensors tr[data-tensor]');
  await expect(pooled).toHaveCount(1);
  await expect(dimsOf(pooled.first())).toHaveText('[1, 384]');
});
