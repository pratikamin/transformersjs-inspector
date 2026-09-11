// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, test } from 'vitest';
import { InspectorBus } from '../src/bus';
import type { Dock } from '../src/panel/fit';
import { FIT_MARGIN } from '../src/panel/fit';
import type { InspectorPanel, PanelOptions } from '../src/panel/panel';
import { mountPanel } from '../src/panel/panel';
import { capOf, installResize, MIN_HEIGHT, MIN_WIDTH, resizeFromPointer } from '../src/panel/resize';

const VP = { width: 1280, height: 900 };
const START = { width: 560, height: 400 };
const DOCKS: Dock[] = ['bottom-right', 'bottom-left', 'top-right', 'top-left'];

describe('resizeFromPointer', () => {
  test('bottom-right: moving the pointer up-left grows the panel by the delta', () => {
    expect(resizeFromPointer('bottom-right', { x: 700, y: 500 }, { x: 600, y: 420 }, START)).toEqual({ width: 660, height: 480 });
    expect(resizeFromPointer('bottom-right', { x: 700, y: 500 }, { x: 750, y: 530 }, START)).toEqual({ width: 510, height: 370 });
  });

  test('bottom-left: right grows the width, up grows the height', () => {
    expect(resizeFromPointer('bottom-left', { x: 100, y: 500 }, { x: 200, y: 420 }, START)).toEqual({ width: 660, height: 480 });
  });

  test('top-right: left grows the width, down grows the height', () => {
    expect(resizeFromPointer('top-right', { x: 700, y: 100 }, { x: 600, y: 180 }, START)).toEqual({ width: 660, height: 480 });
  });

  test('top-left: moving down-right grows the panel', () => {
    expect(resizeFromPointer('top-left', { x: 100, y: 100 }, { x: 200, y: 180 }, START)).toEqual({ width: 660, height: 480 });
    expect(resizeFromPointer('top-left', { x: 100, y: 100 }, { x: 50, y: 60 }, START)).toEqual({ width: 510, height: 360 });
  });

  test('no movement keeps the start size for every dock', () => {
    for (const dock of DOCKS) expect(resizeFromPointer(dock, { x: 300, y: 300 }, { x: 300, y: 300 }, START), dock).toEqual(START);
  });

  test('never goes below MIN_WIDTH x MIN_HEIGHT', () => {
    expect(resizeFromPointer('bottom-right', { x: 0, y: 0 }, { x: 5000, y: 5000 }, START)).toEqual({ width: MIN_WIDTH, height: MIN_HEIGHT });
    expect(resizeFromPointer('top-left', { x: 0, y: 0 }, { x: -5000, y: -5000 }, START)).toEqual({ width: MIN_WIDTH, height: MIN_HEIGHT });
  });

  test('caps at the given maximums; a missing or non-finite cap means unbounded', () => {
    const grown = resizeFromPointer('bottom-right', { x: 0, y: 0 }, { x: -5000, y: -5000 }, START, { width: 1248, height: 868 });
    expect(grown).toEqual({ width: 1248, height: 868 });
    const partial = resizeFromPointer('bottom-right', { x: 0, y: 0 }, { x: -5000, y: -5000 }, START, { width: 1000 });
    expect(partial).toEqual({ width: 1000, height: 5400 });
    const none = resizeFromPointer('bottom-right', { x: 0, y: 0 }, { x: -5000, y: -5000 }, START, { width: Infinity, height: Number.NaN });
    expect(none).toEqual({ width: 5560, height: 5400 });
  });

  test('capOf parses a px length and treats anything else as no cap', () => {
    expect(capOf('1248px')).toBe(1248);
    expect(capOf('12.5px')).toBe(12.5);
    expect(capOf('')).toBe(Infinity);
    expect(capOf('none')).toBe(Infinity);
  });
});

/** happy-dom's `PointerEvent` lacks `pointerId` init in some versions; patch it on after construction. */
const pointer = (type: string, x: number, y: number, pointerId = 1, init: PointerEventInit = {}): PointerEvent => {
  const ev = new PointerEvent(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, pointerId, ...init });
  if (ev.pointerId !== pointerId) Object.defineProperty(ev, 'pointerId', { value: pointerId });
  return ev;
};

describe('installResize (pure binding)', () => {
  test('writes the clamped size on every move, ends on pointerup, resets on dblclick, and the disposer detaches', () => {
    const root = document.createElement('div');
    const grip = document.createElement('div');
    root.appendChild(grip);
    Object.defineProperty(root, 'offsetWidth', { value: 560, configurable: true });
    Object.defineProperty(root, 'offsetHeight', { value: 400, configurable: true });
    root.style.maxWidth = '1248px';
    root.style.maxHeight = '868px';
    let dock: Dock = 'bottom-right';
    let resizes = 0;
    const dispose = installResize(root, grip, { dock: () => dock, onResize: () => resizes++ });

    grip.dispatchEvent(pointer('pointerdown', 700, 500));
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    expect(root.style.width).toBe('660px');
    expect(root.style.height).toBe('480px');
    expect(resizes).toBe(1);
    // Another pointer's moves are ignored while pointer 1 drags.
    grip.dispatchEvent(pointer('pointermove', 0, 0, 2));
    expect(root.style.width).toBe('660px');
    grip.dispatchEvent(pointer('pointermove', -1000, -1000));
    expect(root.style.width).toBe('1248px');
    expect(root.style.height).toBe('868px');
    grip.dispatchEvent(pointer('pointermove', 1500, 1500));
    expect(root.style.width).toBe(`${MIN_WIDTH}px`);
    expect(root.style.height).toBe(`${MIN_HEIGHT}px`);
    grip.dispatchEvent(pointer('pointerup', 1500, 1500));
    const after = resizes;
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    expect(root.style.width).toBe(`${MIN_WIDTH}px`);
    expect(resizes).toBe(after);

    // A secondary button does not start a drag.
    grip.dispatchEvent(pointer('pointerdown', 700, 500, 1, { button: 2 }));
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    expect(root.style.width).toBe(`${MIN_WIDTH}px`);

    // The dock is read at pointerdown, so a top-left dock grows on a down-right drag.
    dock = 'top-left';
    grip.dispatchEvent(pointer('pointerdown', 100, 100));
    grip.dispatchEvent(pointer('pointermove', 200, 180));
    expect(root.style.width).toBe('660px');
    expect(root.style.height).toBe('480px');
    grip.dispatchEvent(pointer('pointercancel', 200, 180));

    grip.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(root.style.width).toBe('');
    expect(root.style.height).toBe('');
    expect(resizes).toBe(after + 2);

    dispose();
    grip.dispatchEvent(pointer('pointerdown', 700, 500));
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    expect(root.style.width).toBe('');
    expect(resizes).toBe(after + 2);
  });
});

describe('mountPanel resize grip', () => {
  const panels: InspectorPanel[] = [];
  const innerWidth = window.innerWidth;
  const innerHeight = window.innerHeight;

  beforeEach(() => {
    Object.defineProperty(window, 'innerWidth', { value: VP.width, configurable: true, writable: true });
    Object.defineProperty(window, 'innerHeight', { value: VP.height, configurable: true, writable: true });
  });

  afterEach(() => {
    for (const p of panels.splice(0)) p.destroy();
    Object.defineProperty(window, 'innerWidth', { value: innerWidth, configurable: true, writable: true });
    Object.defineProperty(window, 'innerHeight', { value: innerHeight, configurable: true, writable: true });
  });

  /**
   * Mounts an open panel measuring 560x400 whose anchor corner sits `FIT_MARGIN` inside the
   * viewport, as the stylesheet's fixed inset would place it.
   */
  const mount = (opts: PanelOptions = {}): { panel: InspectorPanel; root: HTMLElement; grip: HTMLElement } => {
    const panel = mountPanel(new InspectorBus(), { open: true, ...opts });
    panels.push(panel);
    const root = panel.shadow.querySelector<HTMLElement>('[data-panel]');
    const grip = panel.shadow.querySelector<HTMLElement>('[data-grip]');
    if (!root || !grip) throw new Error('panel root or grip missing');
    const dock = opts.dock ?? 'bottom-right';
    const left = dock.endsWith('right') ? VP.width - FIT_MARGIN - START.width : FIT_MARGIN;
    const top = dock.startsWith('bottom') ? VP.height - FIT_MARGIN - START.height : FIT_MARGIN;
    root.getBoundingClientRect = () =>
      ({ x: left, y: top, left, top, width: START.width, height: START.height, right: left + START.width, bottom: top + START.height, toJSON: () => ({}) }) as DOMRect;
    Object.defineProperty(root, 'offsetWidth', { value: START.width, configurable: true });
    Object.defineProperty(root, 'offsetHeight', { value: START.height, configurable: true });
    panel.close();
    panel.open(); // re-fit against the stubbed rect
    return { panel, root, grip };
  };

  test('a drag from the grip writes width and height through the CSSOM and re-fits', () => {
    const { root, grip } = mount();
    expect(root.style.maxWidth).toBe(`${VP.width - 2 * FIT_MARGIN}px`);
    expect(root.style.maxHeight).toBe(`${VP.height - 2 * FIT_MARGIN}px`);
    grip.dispatchEvent(pointer('pointerdown', 700, 500));
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    grip.dispatchEvent(pointer('pointerup', 600, 420));
    expect(root.style.width).toBe('660px');
    expect(root.style.height).toBe('480px');
    expect(root.getAttribute('style')).toContain('width'); // CSSOM write, never an authored markup attribute
  });

  test('a drag past the viewport clamps to the fit caps; a drag the other way clamps to the minimums', () => {
    const { root, grip } = mount();
    grip.dispatchEvent(pointer('pointerdown', 700, 500));
    grip.dispatchEvent(pointer('pointermove', -1000, -1000));
    expect(root.style.width).toBe(root.style.maxWidth);
    expect(root.style.height).toBe(root.style.maxHeight);
    grip.dispatchEvent(pointer('pointermove', 1500, 1500));
    expect(root.style.width).toBe(`${MIN_WIDTH}px`);
    expect(root.style.height).toBe(`${MIN_HEIGHT}px`);
    grip.dispatchEvent(pointer('pointerup', 1500, 1500));
  });

  test('double-click on the grip resets both dimensions', () => {
    const { root, grip } = mount();
    grip.dispatchEvent(pointer('pointerdown', 700, 500));
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    grip.dispatchEvent(pointer('pointerup', 600, 420));
    expect(root.style.width).toBe('660px');
    grip.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(root.style.width).toBe('');
    expect(root.style.height).toBe('');
    // The grip is not an action target: the click delegation leaves the panel open.
    grip.click();
    expect(root.classList.contains('closed')).toBe(false);
  });

  test('with dock: top-left the same drag down-right grows the panel', () => {
    const { panel, root, grip } = mount({ dock: 'top-left' });
    expect(panel.host.dataset.dock).toBe('top-left');
    grip.dispatchEvent(pointer('pointerdown', 100, 100));
    grip.dispatchEvent(pointer('pointermove', 200, 180));
    grip.dispatchEvent(pointer('pointerup', 200, 180));
    expect(root.style.width).toBe('660px');
    expect(root.style.height).toBe('480px');
    expect(root.style.translate).toBe('');
  });

  test('destroy() removes the pointer listeners', () => {
    const { panel, root, grip } = mount();
    panel.destroy();
    grip.dispatchEvent(pointer('pointerdown', 700, 500));
    grip.dispatchEvent(pointer('pointermove', 600, 420));
    expect(root.style.width).toBe('');
    expect(root.style.height).toBe('');
  });
});
