/**
 * Timing budget: the demo's Benchmark button times 30 MiniLM embeddings with the pipeline
 * detached and 30 more attached (panel closed) and publishes both medians on
 * `window.__bench`. The brief's ~5 % figure is *printed* here and recorded by hand in
 * the README; the assertion is the plan's noise-tolerant `ratio < 1.25`, because a
 * 5 % gate on ~10 ms runs would flake.
 */
import { READY_TIMEOUT, TASK_TIMEOUT, expect, test } from './fixtures';

interface BenchResult {
  detachedMs: number;
  attachedMs: number;
  ratio: number;
}

/** Warm-up (5, attached) + 30 attached; the 30 detached runs must not produce call rows. */
const EXPECTED_ATTACHED_CALLS = 5 + 30;

test('overhead: attached embeddings cost less than 1.25x detached', async ({ page }) => {
  await page.goto('/');
  const button = page.getByRole('button', { name: 'Benchmark' });
  await expect(button, 'demo page did not finish loading Transformers.js').toBeEnabled({ timeout: READY_TIMEOUT });
  await button.click();

  const holder = page.locator('[data-bench-status]');
  const output = page.locator('[data-bench]');
  await expect
    .poll(() => holder.getAttribute('data-bench-status'), { timeout: TASK_TIMEOUT, message: 'benchmark did not finish' })
    .toMatch(/^(done|error)$/);
  expect(await holder.getAttribute('data-bench-status'), `benchmark failed: ${await output.textContent()}`).toBe('done');

  const bench = await page.evaluate(() => (window as unknown as { __bench?: BenchResult }).__bench);
  if (!bench) throw new Error('window.__bench was not set');
  // eslint-disable-next-line no-console
  console.log(`overhead: detached median ${bench.detachedMs.toFixed(2)} ms · attached median ${bench.attachedMs.toFixed(2)} ms · ratio ${bench.ratio.toFixed(3)}`);

  await expect(output).toContainText(`ratio ${bench.ratio.toFixed(3)}`);
  await expect(output).toContainText(`detached median ${bench.detachedMs.toFixed(2)} ms`);
  await expect(output).toContainText(`attached median ${bench.attachedMs.toFixed(2)} ms`);
  expect(bench.detachedMs).toBeGreaterThan(0);
  expect(bench.attachedMs).toBeGreaterThan(0);
  expect(bench.ratio).toBeCloseTo(bench.attachedMs / bench.detachedMs, 6);
  expect(bench.ratio).toBeLessThan(1.25);

  // Proof the two phases really were detached / attached: only the attached runs made call rows.
  await expect(page.locator('[data-tjsi-panel] [data-badge]')).toHaveText(String(EXPECTED_ATTACHED_CALLS));
});
