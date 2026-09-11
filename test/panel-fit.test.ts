// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';

import { computeFit, DEFAULT_DOCK, fitElement, FIT_MARGIN, MIN_HEIGHT, MIN_WIDTH } from '../src/panel/fit';
import type { Dock, Viewport } from '../src/panel/fit';
import { InspectorBus } from '../src/bus';
import { mountPanel } from '../src/panel/panel';
import { fixtureEvents } from './fakes';

const vp = (width: number, height: number, offsetLeft = 0, offsetTop = 0): Viewport => ({ width, height, offsetLeft, offsetTop });

describe('computeFit', () => {
  it('caps a viewport-anchored panel to the viewport minus margins', () => {
    const f = computeFit(1000 - FIT_MARGIN, 800 - FIT_MARGIN, vp(1000, 800));
    expect(f).toEqual({ maxWidth: 1000 - 2 * FIT_MARGIN, maxHeight: 800 - 2 * FIT_MARGIN, dx: 0, dy: 0 });
  });

  it('shrinks to the space above an anchor that sits above the viewport bottom (dock case)', () => {
    // Anchor 208px above the bottom: the panel must not grow past the top of the screen.
    const anchorBottom = 800 - 208;
    const f = computeFit(1000 - FIT_MARGIN, anchorBottom, vp(1000, 800));
    expect(f.maxHeight).toBe(anchorBottom - FIT_MARGIN);
    expect(f.dy).toBe(0);
  });

  it('shrinks to the space left of an anchor near the left edge', () => {
    const f = computeFit(400, 700, vp(1000, 800));
    expect(f.maxWidth).toBe(400 - FIT_MARGIN);
    expect(f.dx).toBe(0);
  });

  it('translates an anchor that is off the right or bottom edge back inside', () => {
    const f = computeFit(1200, 900, vp(1000, 800));
    expect(f.dx).toBe(1000 - FIT_MARGIN - 1200);
    expect(f.dy).toBe(800 - FIT_MARGIN - 900);
    expect(f.maxWidth).toBe(1000 - 2 * FIT_MARGIN);
    expect(f.maxHeight).toBe(800 - 2 * FIT_MARGIN);
  });

  it('pushes the anchor out rather than shrinking below the minimum size', () => {
    const f = computeFit(100, 100, vp(1000, 800));
    expect(f.maxWidth).toBe(MIN_WIDTH);
    expect(f.maxHeight).toBe(MIN_HEIGHT);
    expect(f.dx).toBe(MIN_WIDTH - (100 - FIT_MARGIN));
    expect(f.dy).toBe(MIN_HEIGHT - (100 - FIT_MARGIN));
  });

  it('never asks for more than the viewport on a tiny screen', () => {
    const f = computeFit(300, 400, vp(320, 200));
    // The anchor is 300px in: 284px of room to its left, which is above the minimum, so no shift.
    expect(f.maxWidth).toBe(300 - FIT_MARGIN);
    expect(f.dx).toBe(0);
    // The anchor is below the screen: pulled up to the margin and capped to the viewport.
    expect(f.maxHeight).toBe(200 - 2 * FIT_MARGIN);
    expect(f.dy).toBe(200 - FIT_MARGIN - 400);
  });

  it('accounts for the visual viewport offset when pinch-zoomed or scrolled', () => {
    const f = computeFit(600, 500, vp(400, 300, 300, 250));
    // Visible box is x 300..700, y 250..550; the anchor is inside it.
    expect(f.maxWidth).toBe(600 - (300 + FIT_MARGIN));
    expect(f.maxHeight).toBe(500 - (250 + FIT_MARGIN));
    expect(f.dx).toBe(0);
    expect(f.dy).toBe(0);
  });
});

describe('computeFit with a dock', () => {
  it('defaults to bottom-right', () => {
    expect(DEFAULT_DOCK).toBe('bottom-right');
    const explicit = computeFit(1200, 900, vp(1000, 800), 'bottom-right');
    expect(explicit).toEqual(computeFit(1200, 900, vp(1000, 800)));
  });

  describe('top-left', () => {
    it('caps a viewport-anchored panel to the viewport minus margins with no shift', () => {
      const f = computeFit(FIT_MARGIN, FIT_MARGIN, vp(1000, 800), 'top-left');
      expect(f).toEqual({ maxWidth: 1000 - 2 * FIT_MARGIN, maxHeight: 800 - 2 * FIT_MARGIN, dx: 0, dy: 0 });
    });

    it('shrinks to the space below and to the right of an anchor inside the viewport', () => {
      const f = computeFit(400, 300, vp(1000, 800), 'top-left');
      expect(f.maxWidth).toBe(1000 - FIT_MARGIN - 400);
      expect(f.maxHeight).toBe(800 - FIT_MARGIN - 300);
      expect(f.dx).toBe(0);
      expect(f.dy).toBe(0);
    });

    it('translates an anchor that is off the left or top edge back inside', () => {
      const f = computeFit(-50, -50, vp(1000, 800), 'top-left');
      expect(f.dx).toBe(66);
      expect(f.dy).toBe(66);
      expect(f.maxWidth).toBe(1000 - 2 * FIT_MARGIN);
      expect(f.maxHeight).toBe(800 - 2 * FIT_MARGIN);
    });

    it('pushes the anchor out (towards the origin) rather than shrinking below the minimum size', () => {
      const f = computeFit(900, 700, vp(1000, 800), 'top-left');
      expect(f.maxWidth).toBe(MIN_WIDTH);
      expect(f.maxHeight).toBe(MIN_HEIGHT);
      expect(f.dx).toBe(-(MIN_WIDTH - (1000 - FIT_MARGIN - 900)));
      expect(f.dy).toBe(-(MIN_HEIGHT - (800 - FIT_MARGIN - 700)));
    });

    it('never asks for more than the viewport on a tiny screen', () => {
      const f = computeFit(20, -200, vp(320, 200), 'top-left');
      // 284px of room to the right of the anchor, above the minimum, so no shift.
      expect(f.maxWidth).toBe(320 - FIT_MARGIN - 20);
      expect(f.dx).toBe(0);
      // The anchor is above the screen: pulled down to the margin and capped to the viewport.
      expect(f.maxHeight).toBe(200 - 2 * FIT_MARGIN);
      expect(f.dy).toBe(FIT_MARGIN + 200);
    });

    it('accounts for the visual viewport offset when pinch-zoomed or scrolled', () => {
      const f = computeFit(400, 300, vp(400, 300, 300, 250), 'top-left');
      // Visible box is x 300..700, y 250..550; the anchor is inside it.
      expect(f.maxWidth).toBe(700 - FIT_MARGIN - 400);
      expect(f.maxHeight).toBe(550 - FIT_MARGIN - 300);
      expect(f.dx).toBe(0);
      expect(f.dy).toBe(0);
    });

    it('translates an anchor that is off the visual viewport even when inside the layout one', () => {
      const f = computeFit(100, 100, vp(400, 300, 300, 250), 'top-left');
      expect(f.dx).toBe(300 + FIT_MARGIN - 100);
      expect(f.dy).toBe(250 + FIT_MARGIN - 100);
      expect(f.maxWidth).toBe(400 - 2 * FIT_MARGIN);
      expect(f.maxHeight).toBe(300 - 2 * FIT_MARGIN);
    });
  });

  describe('bottom-left', () => {
    it('caps a viewport-anchored panel to the viewport minus margins', () => {
      const f = computeFit(FIT_MARGIN, 800 - FIT_MARGIN, vp(1000, 800), 'bottom-left');
      expect(f).toEqual({ maxWidth: 1000 - 2 * FIT_MARGIN, maxHeight: 800 - 2 * FIT_MARGIN, dx: 0, dy: 0 });
    });

    it('shrinks to the space right of an anchor near the right edge', () => {
      const f = computeFit(700, 700, vp(1000, 800), 'bottom-left');
      expect(f.maxWidth).toBe(1000 - FIT_MARGIN - 700);
      expect(f.maxHeight).toBe(700 - FIT_MARGIN);
      expect(f.dx).toBe(0);
      expect(f.dy).toBe(0);
    });

    it('translates an anchor that is off the left or bottom edge back inside', () => {
      const f = computeFit(-40, 900, vp(1000, 800), 'bottom-left');
      expect(f.dx).toBe(FIT_MARGIN + 40);
      expect(f.dy).toBe(800 - FIT_MARGIN - 900);
      expect(f.maxWidth).toBe(1000 - 2 * FIT_MARGIN);
      expect(f.maxHeight).toBe(800 - 2 * FIT_MARGIN);
    });

    it('pushes the anchor out rather than shrinking below the minimum size', () => {
      const f = computeFit(900, 100, vp(1000, 800), 'bottom-left');
      expect(f.maxWidth).toBe(MIN_WIDTH);
      expect(f.maxHeight).toBe(MIN_HEIGHT);
      expect(f.dx).toBe(-(MIN_WIDTH - (1000 - FIT_MARGIN - 900)));
      expect(f.dy).toBe(MIN_HEIGHT - (100 - FIT_MARGIN));
    });

    it('accounts for the visual viewport offset when pinch-zoomed or scrolled', () => {
      const f = computeFit(400, 500, vp(400, 300, 300, 250), 'bottom-left');
      expect(f.maxWidth).toBe(700 - FIT_MARGIN - 400);
      expect(f.maxHeight).toBe(500 - (250 + FIT_MARGIN));
      expect(f.dx).toBe(0);
      expect(f.dy).toBe(0);
    });
  });

  describe('top-right', () => {
    it('caps a viewport-anchored panel to the viewport minus margins', () => {
      const f = computeFit(1000 - FIT_MARGIN, FIT_MARGIN, vp(1000, 800), 'top-right');
      expect(f).toEqual({ maxWidth: 1000 - 2 * FIT_MARGIN, maxHeight: 800 - 2 * FIT_MARGIN, dx: 0, dy: 0 });
    });

    it('shrinks to the space below an anchor near the bottom edge', () => {
      const f = computeFit(600, 500, vp(1000, 800), 'top-right');
      expect(f.maxWidth).toBe(600 - FIT_MARGIN);
      expect(f.maxHeight).toBe(800 - FIT_MARGIN - 500);
      expect(f.dx).toBe(0);
      expect(f.dy).toBe(0);
    });

    it('translates an anchor that is off the right or top edge back inside', () => {
      const f = computeFit(1200, -30, vp(1000, 800), 'top-right');
      expect(f.dx).toBe(1000 - FIT_MARGIN - 1200);
      expect(f.dy).toBe(FIT_MARGIN + 30);
      expect(f.maxWidth).toBe(1000 - 2 * FIT_MARGIN);
      expect(f.maxHeight).toBe(800 - 2 * FIT_MARGIN);
    });

    it('pushes the anchor out rather than shrinking below the minimum size', () => {
      const f = computeFit(100, 700, vp(1000, 800), 'top-right');
      expect(f.maxWidth).toBe(MIN_WIDTH);
      expect(f.maxHeight).toBe(MIN_HEIGHT);
      expect(f.dx).toBe(MIN_WIDTH - (100 - FIT_MARGIN));
      expect(f.dy).toBe(-(MIN_HEIGHT - (800 - FIT_MARGIN - 700)));
    });

    it('accounts for the visual viewport offset when pinch-zoomed or scrolled', () => {
      const f = computeFit(600, 300, vp(400, 300, 300, 250), 'top-right');
      expect(f.maxWidth).toBe(600 - (300 + FIT_MARGIN));
      expect(f.maxHeight).toBe(550 - FIT_MARGIN - 300);
      expect(f.dx).toBe(0);
      expect(f.dy).toBe(0);
    });
  });

  it('is symmetric: mirroring the anchor and the dock mirrors the fit', () => {
    const v = vp(1000, 800);
    const br = computeFit(1200, 900, v, 'bottom-right');
    const tl = computeFit(-200, -100, v, 'top-left');
    expect(tl).toEqual({ maxWidth: br.maxWidth, maxHeight: br.maxHeight, dx: -br.dx, dy: -br.dy });
    const bl = computeFit(-200, 900, v, 'bottom-left');
    expect(bl).toEqual({ maxWidth: br.maxWidth, maxHeight: br.maxHeight, dx: -br.dx, dy: br.dy });
    const tr = computeFit(1200, -100, v, 'top-right');
    expect(tr).toEqual({ maxWidth: br.maxWidth, maxHeight: br.maxHeight, dx: br.dx, dy: -br.dy });
  });
});

describe('fitElement', () => {
  const withRect = (el: HTMLElement, rect: Partial<DOMRect>): void => {
    el.getBoundingClientRect = () => ({ x: 0, y: 0, top: 0, left: 0, width: 0, height: 0, right: 0, bottom: 0, toJSON: () => ({}), ...rect }) as DOMRect;
  };

  it('is a no-op before layout (zero-size rect)', () => {
    const el = document.createElement('div');
    withRect(el, {});
    expect(fitElement(el, vp(1000, 800))).toBeNull();
    expect(el.style.maxHeight).toBe('');
  });

  it('writes size caps and translation through the CSSOM', () => {
    const el = document.createElement('div');
    withRect(el, { width: 560, height: 600, right: 1100, bottom: 700 });
    const f = fitElement(el, vp(1000, 800));
    expect(f).not.toBeNull();
    expect(el.style.maxWidth).toBe(`${1000 - 2 * FIT_MARGIN}px`);
    expect(el.style.maxHeight).toBe(`${700 - FIT_MARGIN}px`);
    expect(el.style.translate).toBe(`${1000 - FIT_MARGIN - 1100}px 0px`);
    expect(el.getAttribute('style')).not.toBeNull(); // CSSOM writes are fine; no markup attribute was authored
  });

  it('reads the anchor corner named by the dock from the rect', () => {
    const rect = { width: 560, height: 600, left: -50, top: -50, right: 510, bottom: 550 };
    const el = document.createElement('div');
    withRect(el, rect);
    const f = fitElement(el, vp(1000, 800), 'top-left');
    expect(f).toEqual(computeFit(rect.left, rect.top, vp(1000, 800), 'top-left'));
    expect(el.style.translate).toBe('66px 66px');
    expect(el.style.maxWidth).toBe(`${1000 - 2 * FIT_MARGIN}px`);
    expect(el.style.maxHeight).toBe(`${800 - 2 * FIT_MARGIN}px`);

    const cases: Array<[Dock, number, number]> = [
      ['bottom-right', rect.right, rect.bottom],
      ['bottom-left', rect.left, rect.bottom],
      ['top-right', rect.right, rect.top],
      ['top-left', rect.left, rect.top],
    ];
    for (const [dock, ax, ay] of cases) {
      const e = document.createElement('div');
      withRect(e, rect);
      expect(fitElement(e, vp(1000, 800), dock), dock).toEqual(computeFit(ax, ay, vp(1000, 800), dock));
    }
  });
});

describe('mountPanel fitting', () => {
  it('re-fits when opened and when a row is expanded, and stops watching on destroy', () => {
    const bus = new InspectorBus();
    const panel = mountPanel(bus, { open: false });
    const root = panel.shadow.querySelector<HTMLElement>('[data-panel]');
    if (!root) throw new Error('panel root missing');
    // Simulate a host that parked the panel 208px above the viewport bottom (a dock).
    let calls = 0;
    root.getBoundingClientRect = () => {
      calls += 1;
      return { x: 0, y: 0, top: 100, left: 464, width: 560, height: 600, right: 1024, bottom: 768 - 208, toJSON: () => ({}) } as DOMRect;
    };
    for (const ev of fixtureEvents()) bus.emit(ev);
    panel.open();
    expect(calls).toBeGreaterThan(0);
    expect(root.style.maxHeight).toBe(`${768 - 208 - FIT_MARGIN}px`);
    const before = calls;
    panel.shadow.querySelector<HTMLElement>('[data-action="expand"]')?.click();
    expect(calls).toBeGreaterThan(before);
    panel.destroy();
    const after = calls;
    window.dispatchEvent(new Event('resize'));
    expect(calls).toBe(after);
  });
});
