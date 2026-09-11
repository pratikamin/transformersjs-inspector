/**
 * Dock option and drag-to-resize in a real Chromium: the grip on the corner opposite the
 * anchor is dragged with `page.mouse`, the panel's box grows by the drag delta and stays
 * inside the viewport, and a double-click on the grip restores the stylesheet's size.
 * `?dock=` on the demo picks the anchored corner. Runs feature extraction only because the
 * panel is mounted by `attach()`; the model is in the persistent profile after the first run.
 */
import type { Locator, Page } from '@playwright/test';
import { expect, runTask, test } from './fixtures';

const SENTENCE = 'Resize me.';
const VIEWPORT = { width: 1280, height: 900 };
const MARGIN = 16;
const DEFAULT_WIDTH = 560;
const TOLERANCE = 2;

type Box = { x: number; y: number; width: number; height: number };

/** Loads `path`, runs the task so the panel mounts, opens it, and returns the panel locators. */
async function openPanel(page: Page, path: string): Promise<{ host: Locator; panel: Locator; grip: Locator }> {
  await page.goto(path);
  await runTask(page, 'feature-extraction', SENTENCE);
  const host = page.locator('[data-tjsi-panel]');
  await expect(host).toHaveCount(1);
  await host.locator('[data-action="toggle"]').click();
  const panel = host.locator('[data-panel]');
  await expect(panel).not.toHaveClass(/closed/);
  const grip = host.locator('[data-grip]');
  await expect(grip).toBeVisible();
  return { host, panel, grip };
}

async function boxOf(locator: Locator): Promise<Box> {
  const box = await locator.boundingBox();
  if (!box) throw new Error(`${String(locator)} has no box`);
  return box;
}

/** Presses on the grip's centre and drags by `(dx, dy)` in steps, as a user would. */
async function dragGrip(page: Page, grip: Locator, dx: number, dy: number): Promise<void> {
  const g = await boxOf(grip);
  const cx = g.x + g.width / 2;
  const cy = g.y + g.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + dx, cy + dy, { steps: 8 });
  await page.mouse.up();
}

/** Within `TOLERANCE`: the grip sits inside the panel's 1px border, and boxes are fractional. */
const near = (actual: number, expected: number): void => expect(Math.abs(actual - expected)).toBeLessThanOrEqual(TOLERANCE);

function expectInsideViewport(box: Box): void {
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(VIEWPORT.width);
  expect(box.y + box.height).toBeLessThanOrEqual(VIEWPORT.height);
}

test('bottom-right (default): dragging the top-left grip grows the panel by the delta; double-click resets', async ({ page }) => {
  const { host, panel, grip } = await openPanel(page, '/');
  await expect(host).toHaveAttribute('data-dock', 'bottom-right');
  const box0 = await boxOf(panel);
  expect(box0.width).toBeCloseTo(DEFAULT_WIDTH, 0);
  // The anchor: 16px from the right and bottom edges.
  expect(box0.x + box0.width).toBeCloseTo(VIEWPORT.width - MARGIN, 0);
  expect(box0.y + box0.height).toBeCloseTo(VIEWPORT.height - MARGIN, 0);
  // The grip sits on the opposite (top-left) corner.
  const g = await boxOf(grip);
  near(g.x, box0.x);
  near(g.y, box0.y);
  expect(g.width).toBe(16);
  expect(g.height).toBe(16);
  await expect(grip).toHaveCSS('cursor', 'nwse-resize');

  // A single open row is well under the 160px minimum height, so the vertical delta is 200.
  await dragGrip(page, grip, -120, -200);
  const box1 = await boxOf(panel);
  expect(Math.abs(box1.width - (box0.width + 120))).toBeLessThanOrEqual(TOLERANCE);
  expect(Math.abs(box1.height - (box0.height + 200))).toBeLessThanOrEqual(TOLERANCE);
  // The anchor did not move and the panel is still fully on screen.
  expect(box1.x + box1.width).toBeCloseTo(box0.x + box0.width, 0);
  expect(box1.y + box1.height).toBeCloseTo(box0.y + box0.height, 0);
  expectInsideViewport(box1);

  // The stylesheet's caps still hold: a huge drag cannot push the panel off the top or left.
  await dragGrip(page, grip, -2000, -2000);
  const box2 = await boxOf(panel);
  expect(box2.width).toBeLessThanOrEqual(VIEWPORT.width - 2 * MARGIN + TOLERANCE);
  expect(box2.height).toBeLessThanOrEqual(VIEWPORT.height - 2 * MARGIN + TOLERANCE);
  expectInsideViewport(box2);

  await grip.dblclick();
  const box3 = await boxOf(panel);
  expect(Math.abs(box3.width - DEFAULT_WIDTH)).toBeLessThanOrEqual(TOLERANCE);
  expect(Math.abs(box3.height - box0.height)).toBeLessThanOrEqual(TOLERANCE);

  // The collapsed badge never inherits a dragged width.
  await dragGrip(page, grip, -120, -200);
  await host.locator('[data-action="toggle"]').click();
  await expect(panel).toHaveClass(/closed/);
  const closed = await boxOf(panel);
  expect(closed.width).toBeLessThan(DEFAULT_WIDTH);
  await expect(grip).toBeHidden();
});

test('?dock=top-left: the host sits at (16, 16) and a drag down-right grows the panel', async ({ page }) => {
  const { host, panel, grip } = await openPanel(page, '/?dock=top-left');
  await expect(host).toHaveAttribute('data-dock', 'top-left');
  const hostBox = await boxOf(host);
  expect(hostBox.x).toBeCloseTo(MARGIN, 0);
  expect(hostBox.y).toBeCloseTo(MARGIN, 0);
  const box0 = await boxOf(panel);
  // The grip sits on the opposite (bottom-right) corner.
  const g = await boxOf(grip);
  near(g.x + g.width, box0.x + box0.width);
  near(g.y + g.height, box0.y + box0.height);
  await expect(grip).toHaveCSS('cursor', 'nwse-resize');

  await dragGrip(page, grip, 100, 200);
  const box1 = await boxOf(panel);
  expect(Math.abs(box1.width - (box0.width + 100))).toBeLessThanOrEqual(TOLERANCE);
  expect(Math.abs(box1.height - (box0.height + 200))).toBeLessThanOrEqual(TOLERANCE);
  expect(box1.x).toBeCloseTo(box0.x, 0);
  expect(box1.y).toBeCloseTo(box0.y, 0);
  expectInsideViewport(box1);
});

test('?dock=bottom-left: the panel is fixed 16px from the left edge with the grip top-right', async ({ page }) => {
  const { host, panel, grip } = await openPanel(page, '/?dock=bottom-left');
  await expect(host).toHaveAttribute('data-dock', 'bottom-left');
  const hostBox = await boxOf(host);
  expect(hostBox.x).toBeCloseTo(MARGIN, 0);
  expect(hostBox.y + hostBox.height).toBeCloseTo(VIEWPORT.height - MARGIN, 0);
  const box0 = await boxOf(panel);
  const g = await boxOf(grip);
  near(g.x + g.width, box0.x + box0.width);
  near(g.y, box0.y);
  await expect(grip).toHaveCSS('cursor', 'nesw-resize');

  await dragGrip(page, grip, 100, -200);
  const box1 = await boxOf(panel);
  expect(Math.abs(box1.width - (box0.width + 100))).toBeLessThanOrEqual(TOLERANCE);
  expect(Math.abs(box1.height - (box0.height + 200))).toBeLessThanOrEqual(TOLERANCE);
  expect(box1.x).toBeCloseTo(MARGIN, 0);
  expectInsideViewport(box1);
});
