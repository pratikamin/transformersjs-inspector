/**
 * Replay end to end: a real feature-extraction call, then the row's Replay button. The
 * request travels panel → bus → the registry `attach()` registered, which re-runs the
 * pipeline's wrapped `_call` with the original arguments: a second row appears, marked as a
 * replay of the first, with the same tokenizer ids and the same `last_hidden_state` dims.
 */
import { TASK_TIMEOUT, dimsOf, expect, runTask, sectionTitled, setView, tensorRow, test } from './fixtures';

const SENTENCE = 'Replay the inspector once more.';

test('feature extraction: Replay adds a second row marked as a replay of #1 with identical tokens and dims', async ({ page }) => {
  await page.goto('/');
  await runTask(page, 'feature-extraction', SENTENCE);

  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel.locator('[data-badge]')).toHaveText('1');
  await panel.locator('[data-action="toggle"]').click();
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  const first = rows.first();
  const firstId = await first.getAttribute('data-call');
  expect(firstId).toMatch(/^c\d+$/);
  // An ordinary row carries no marker and an enabled Replay button.
  await expect(first.locator('[data-replay-of]')).toHaveCount(0);
  const replay = first.locator('button[data-action="replay"]');
  await expect(replay).toBeEnabled();

  await replay.click();
  // The click neither expands the row nor errors the button.
  await expect(panel.locator('[data-details]')).toHaveCount(0);
  await expect(rows).toHaveCount(2);
  await expect(panel.locator('[data-badge]')).toHaveText('2');
  await expect(first.locator('button[data-replay-error]')).toHaveCount(0);

  const second = rows.nth(1);
  await expect(second.locator('.replay-of')).toHaveAttribute('data-replay-of', firstId as string);
  await expect(second.locator('.replay-of')).toHaveText('replay of #1');
  await expect(second.locator('.label')).toContainText('feature-extraction');
  await expect(second.locator('.excerpt')).toContainText(SENTENCE);
  // The replayed call finishes on its own; the response came back at call start.
  await expect(second.locator('.dot.ok')).toHaveCount(1, { timeout: TASK_TIMEOUT });
  await expect(replay).toBeEnabled();

  // Same tokens, same session shapes (the ids and tensor rows are detail-view content).
  await setView(panel, 'detail');
  await first.click();
  await second.click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(2);
  const idsOf = async (i: number): Promise<string[]> => sectionTitled(details.nth(i), 'Tokenizer').locator('.chip .chip-id').allTextContents();
  const ids1 = await idsOf(0);
  const ids2 = await idsOf(1);
  expect(ids1.length).toBeGreaterThan(2);
  expect(ids2).toEqual(ids1);
  const hiddenDims = (i: number) => dimsOf(tensorRow(sectionTitled(details.nth(i), 'Session runs'), 'last_hidden_state'));
  await expect(hiddenDims(0)).toHaveText(/^\[1, \d+, 384\]$/);
  await expect(hiddenDims(1)).toHaveText((await hiddenDims(0).textContent()) as string);
  await expect(sectionTitled(details.nth(1), 'Result').locator('pre')).toContainText('"$tensor"');
});
