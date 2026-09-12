/**
 * Media previews end to end: a real `onnx-community/mobilenet_v2_1.0_224` classification
 * of the demo's synthetic canvas image (`RawImage.fromCanvas`, 4 channels), observed by
 * `attach()`. Checks the two halves of Phase D against real bytes: the `call:start` thumbnail
 * in the Input section, and the `pixel_values [1, 3, 224, 224]` tensor painted as an image by
 * the Preview button (values fetched through `bus.request('tensor')`). Selectors resolve
 * through the open shadow root of `[data-tjsi-panel]`.
 */
import { dimsOf, expect, runTask, sectionTitled, setView, tensorRow, test } from './fixtures';

test('image classification: input thumbnail, pixel_values preview as an image, label/score result', async ({ page }) => {
  await page.goto('/');
  const section = page.locator('[data-task="image-classification"]');
  await expect(section).toHaveAttribute('data-status', 'idle');
  // The synthetic input is painted on load, before Run.
  const canvasSize = await section.locator('canvas[data-source]').evaluate((c: HTMLCanvasElement) => [c.width, c.height]);
  expect(canvasSize).toEqual([224, 224]);

  await runTask(page, 'image-classification');
  // Three `label · score` lines from `top_k: 3`.
  const lines = ((await section.locator('[data-output]').textContent()) ?? '').trim().split('\n');
  expect(lines).toHaveLength(3);
  for (const line of lines) expect(line).toMatch(/^.+ · 0\.\d{4}$/);

  const panel = page.locator('[data-tjsi-panel]');
  await expect(panel).toHaveCount(1);
  await panel.locator('[data-action="toggle"]').click();
  const rows = panel.locator('[data-call]');
  await expect(rows).toHaveCount(1);
  await expect(rows.first().locator('.label')).toContainText('image-classification');
  await expect(rows.first().locator('.excerpt')).toHaveText('image 224×224');

  await rows.first().click();
  const details = panel.locator('[data-details]');
  await expect(details).toHaveCount(1);

  // Input: the capture-side JPEG thumbnail and the RawImage's size and channel count.
  const input = sectionTitled(details, 'Input');
  const thumb = input.locator('img.thumb');
  await expect(thumb).toHaveCount(1);
  await expect(thumb).toHaveAttribute('src', /^data:image\//);
  await expect(input.locator('.meta').filter({ hasText: /^image \d/ })).toHaveText('image 224×224×4');
  // No audio on this page: no waveform anywhere in the details.
  await expect(details.locator('svg.wave')).toHaveCount(0);

  // Simple view (the default): three label/percent rows and one Model line, no tables.
  await expect(sectionTitled(details, 'Result').locator('[data-result-summary="labels"] .alt')).toHaveCount(3);
  await expect(sectionTitled(details, 'Result').locator('.alt-pct').first()).toHaveText(/%$/);
  await expect(sectionTitled(details, 'Model').locator('[data-model-line]')).toHaveText(/^1 model run · /);
  await expect(details.locator('table')).toHaveCount(0);
  await setView(panel, 'detail');

  // Session run: the processor's pixel_values in, ImageNet logits out.
  const runs = sectionTitled(details, 'Session runs').locator('.run');
  await expect(runs).toHaveCount(1);
  const run = runs.first();
  const pixels = tensorRow(run, 'pixel_values');
  await expect(dimsOf(pixels)).toHaveText('[1, 3, 224, 224]');
  // 1001, not 1000: Google's MobileNet checkpoints keep ImageNet's extra "background" class.
  await expect(dimsOf(tensorRow(run, 'logits'))).toHaveText('[1, 1001]');
  // Only the image-shaped row offers Preview; the logits row has Load values alone.
  await expect(pixels.locator('button[data-action="preview"]')).toHaveCount(1);
  await expect(tensorRow(run, 'logits').locator('button[data-action="preview"]')).toHaveCount(0);

  // Preview: the tensor is fetched and rasterised into a 224×224 canvas with a mapping caption.
  await expect(run.locator('tr.values')).toHaveCount(0);
  await pixels.locator('button[data-action="preview"]').click();
  const valuesRow = run.locator(`tr.values[data-values="${await pixels.getAttribute('data-tensor')}"]`);
  await expect(valuesRow.locator('[data-values-error]')).toHaveCount(0);
  const image = valuesRow.locator('canvas.tensor-image');
  await expect(image).toHaveCount(1);
  await expect(image).toHaveAttribute('width', '224');
  await expect(image).toHaveAttribute('height', '224');
  // Something was actually painted: the normalised gradient is not a blank canvas.
  const distinct = await image.evaluate((c: HTMLCanvasElement) => {
    const data = c.getContext('2d')?.getImageData(0, 0, c.width, c.height).data ?? new Uint8ClampedArray();
    return new Set(data).size;
  });
  expect(distinct).toBeGreaterThan(16);
  const caption = valuesRow.locator('.tensor-image-caption');
  await expect(caption).toContainText('float32 [1, 3, 224, 224]');
  await expect(caption).toContainText('rgb chw 224×224');
  await expect(caption).toContainText('min…max');

  // Result: the pipeline's plain label/score objects (no tensors).
  const result = sectionTitled(details, 'Result');
  await expect(result.locator('pre')).toContainText('"label"');
  await expect(result.locator('pre')).toContainText('"score"');
  await expect(result.locator('table.tensors')).toHaveCount(0);
});
