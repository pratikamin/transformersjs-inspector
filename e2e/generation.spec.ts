/**
 * Text generation end to end: `onnx-community/tiny-random-LlamaForCausalLM-ONNX` with
 * `max_new_tokens: 3` through the CDN build, observed by `attach()`. The generation wrapper
 * hands the real `generate` a *plain array* as `logits_processor` and a duck-typed streamer;
 * this spec is the proof that 4.2.0 accepts both (the plan's fallback is the
 * `transformers: { LogitsProcessorList }` escape hatch). Panel selectors resolve through the
 * open shadow root of `[data-tjsi-panel]`.
 */
import type { Locator } from '@playwright/test';
import { expect, runTask, setView, test } from './fixtures';

const PROMPT = 'hi';
/** `demo/main.ts` runs text-generation with `{ max_new_tokens: 3 }`. */
const MAX_NEW_TOKENS = 3;
/** The wrapper's default `topK`. */
const TOP_K = 10;
/** Alternatives per step in the simple view. */
const SIMPLE_TOP_N = 5;

/** `<section class="section">` whose `<h3>` is exactly `title`. */
function sectionTitled(details: Locator, title: string): Locator {
  return details.locator('section.section').filter({ has: details.page().locator('h3', { hasText: new RegExp(`^${title}$`) }) });
}

test('text generation: simple step lines first, then >= 3 session runs, 3 steps of sorted top-k with a token each, result equals the page output', async ({ page }) => {
  await page.goto('/');
  const section = await runTask(page, 'text-generation', PROMPT);
  const output = (await section.locator('[data-output]').textContent()) ?? '';
  expect(output.length).toBeGreaterThan(0);

  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel).toHaveCount(1);
  await expect(panel.locator('[data-badge]')).toHaveText('1');

  await panel.locator('[data-action="toggle"]').click();
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  await expect(rows.first().locator('.label')).toContainText('text-generation');
  await rows.first().click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(1);

  // Simple view (the default): the Model line counts the runs with the first one as prefill, each
  // step is `step n → "text"` with at most 5 alternatives as percentages, and the result is the text.
  await expect(details).toHaveAttribute('data-view', 'simple');
  await expect(sectionTitled(details, 'Model').locator('[data-model-line]')).toHaveText(/^\d+ model runs · 1 prefill \+ \d+ decode · \d+(\.\d+)? (ms|s)$/);
  await expect(details.locator('table')).toHaveCount(0);
  const simpleSteps = sectionTitled(details, 'Generation').locator('.step');
  await expect(simpleSteps).toHaveCount(MAX_NEW_TOKENS);
  for (let i = 0; i < MAX_NEW_TOKENS; i++) {
    await expect(simpleSteps.nth(i).locator('.step-head')).toHaveText(new RegExp(`^step ${i} → `));
    const alts = simpleSteps.nth(i).locator('.alt');
    await expect(alts).toHaveCount(SIMPLE_TOP_N);
    await expect(simpleSteps.nth(i).locator('.alt.picked')).toHaveCount(1);
    for (const pct of await alts.locator('.alt-pct').allTextContents()) expect(pct).toMatch(/^(<0\.01|\d+(\.\d+)?)%$/);
  }
  await expect(sectionTitled(details, 'Result').locator('[data-result-summary="text"]')).toHaveText(output);

  await setView(panel, 'detail');
  await expect(details).toHaveAttribute('data-view', 'detail');

  // Session runs: one prefill plus one decode per further token (the spike saw 3 for 3 tokens).
  const runs = sectionTitled(details, 'Session runs').locator('.run');
  expect(await runs.count(), 'session run rows').toBeGreaterThanOrEqual(MAX_NEW_TOKENS);

  // Generation: exactly one step per new token, each with the top-k table and the picked id.
  const generation = sectionTitled(details, 'Generation');
  await expect(generation).toHaveCount(1);
  await expect(generation.locator('.meta').first()).toHaveText(`${MAX_NEW_TOKENS} steps`);
  const steps = generation.locator('.step');
  await expect(steps).toHaveCount(MAX_NEW_TOKENS);

  const tokenIds: number[] = [];
  for (let i = 0; i < MAX_NEW_TOKENS; i++) {
    const step = steps.nth(i);
    await expect(step).toHaveAttribute('data-step', String(i));

    const head = (await step.locator('.step-head').textContent()) ?? '';
    const m = /^step (\d+) · token (\d+)\b/.exec(head);
    if (!m) throw new Error(`step ${i}: no token id in "${head}"`);
    expect(Number(m[1])).toBe(i);
    tokenIds.push(Number(m[2]));

    const trs = step.locator('table.topk tbody tr');
    await expect(trs).toHaveCount(TOP_K);
    const probs = (await trs.locator('td.prob .prob-text').allTextContents()).map(Number);
    expect(probs).toHaveLength(TOP_K);
    for (const p of probs) {
      expect(p, `step ${i} probs ${probs.join(', ')}`).toBeGreaterThan(0);
      expect(p).toBeLessThanOrEqual(1);
    }
    for (let j = 1; j < probs.length; j++) expect(probs[j], `step ${i} not sorted: ${probs.join(', ')}`).toBeLessThanOrEqual(probs[j - 1]);

    const ids = await trs.evaluateAll((els) => els.map((el) => Number((el as HTMLElement).dataset.token)));
    expect(new Set(ids).size).toBe(TOP_K);
    expect(ids.every((id) => Number.isInteger(id) && id >= 0)).toBe(true);

    // Greedy decoding picks the argmax, i.e. the first top-k row, and the panel marks it.
    const picked = step.locator('table.topk tbody tr.picked');
    await expect(picked).toHaveCount(1);
    await expect(picked).toHaveAttribute('data-token', String(tokenIds[i]));
    expect(ids[0]).toBe(tokenIds[i]);
  }
  expect(tokenIds).toHaveLength(MAX_NEW_TOKENS);
  expect(tokenIds.every((id) => Number.isInteger(id) && id >= 0)).toBe(true);

  // Result: the pipeline's [{ generated_text }] as JSON; the page printed generated_text verbatim.
  const resultJson = (await sectionTitled(details, 'Result').locator('pre').textContent()) ?? '';
  const result = JSON.parse(resultJson) as { generated_text?: unknown }[];
  expect(Array.isArray(result)).toBe(true);
  expect(result[0].generated_text).toBe(output);
});
