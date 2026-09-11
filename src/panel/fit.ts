/**
 * Keeps the panel inside the visible viewport whatever its containing block is.
 *
 * The stylesheet anchors the panel bottom-right with `position: fixed`, but a host page can
 * mount it inside a transformed or otherwise positioned container (a common trick to park
 * the panel above a dock), and then `right`/`bottom` resolve against that container while
 * `100vw`/`100vh` still mean the layout viewport. On mobile, `100vh` also overshoots the
 * visible area. So instead of trusting CSS units the panel measures itself: the bottom-right
 * corner is the host's chosen anchor and stays put; the panel is capped to the space above
 * and to the left of that anchor; and only when the anchor itself is outside the visible
 * viewport is the panel translated inward. All writes go through the CSSOM, never an inline
 * style attribute, so CSP `style-src` is not involved.
 */

export const FIT_MARGIN = 16;
/** Below these the panel would be unusable, so the anchor is moved instead of shrinking further. */
export const MIN_WIDTH = 280;
export const MIN_HEIGHT = 160;

export interface Viewport {
  /** Visible width/height (the visual viewport when available, else the window's inner size). */
  width: number;
  height: number;
  /** Visual-viewport offset from the layout viewport (pinch zoom / scrolled mobile URL bar). */
  offsetLeft: number;
  offsetTop: number;
}

export interface Fit {
  maxWidth: number;
  maxHeight: number;
  /** Translation applied to the panel, in px; `0,0` when the anchor is already on screen. */
  dx: number;
  dy: number;
}

/**
 * Reads the visible viewport. `visualViewport` is preferred because on mobile the layout
 * viewport (`innerHeight`, `100vh`) is taller than what the user can actually see.
 */
export function readViewport(win: Window = window): Viewport {
  const vv = win.visualViewport;
  if (vv && vv.width > 0 && vv.height > 0) {
    return { width: vv.width, height: vv.height, offsetLeft: vv.offsetLeft, offsetTop: vv.offsetTop };
  }
  return { width: win.innerWidth, height: win.innerHeight, offsetLeft: 0, offsetTop: 0 };
}

/**
 * Computes size caps and a translation for a panel whose bottom-right corner sits at
 * (`anchorRight`, `anchorBottom`) in layout-viewport coordinates. Pure, so it is unit-testable
 * without layout.
 */
export function computeFit(anchorRight: number, anchorBottom: number, vp: Viewport, margin = FIT_MARGIN): Fit {
  const left = vp.offsetLeft + margin;
  const top = vp.offsetTop + margin;
  const right = vp.offsetLeft + vp.width - margin;
  const bottom = vp.offsetTop + vp.height - margin;
  const viewportW = Math.max(0, right - left);
  const viewportH = Math.max(0, bottom - top);

  const fitAxis = (anchor: number, lo: number, hi: number, min: number, room: number): { cap: number; delta: number } => {
    let delta = 0;
    // Anchor past the far edge: pull it back to the margin.
    if (anchor > hi) delta = hi - anchor;
    // Space between the near margin and the anchor; if too small, push the anchor out to make room.
    let available = anchor + delta - lo;
    const wanted = Math.min(min, room);
    if (available < wanted) {
      delta += wanted - available;
      available = wanted;
    }
    return { cap: Math.min(available, room), delta };
  };

  const x = fitAxis(anchorRight, left, right, MIN_WIDTH, viewportW);
  const y = fitAxis(anchorBottom, top, bottom, MIN_HEIGHT, viewportH);
  return { maxWidth: x.cap, maxHeight: y.cap, dx: x.delta, dy: y.delta };
}

/** Applies `computeFit` to a laid-out element; a no-op before first layout (zero-size rect). */
export function fitElement(el: HTMLElement, vp: Viewport = readViewport()): Fit | null {
  // Measure without the previous translation so the anchor is the host's, not ours.
  el.style.translate = '';
  const r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0) return null;
  const fit = computeFit(r.right, r.bottom, vp);
  el.style.maxWidth = `${Math.round(fit.maxWidth)}px`;
  el.style.maxHeight = `${Math.round(fit.maxHeight)}px`;
  el.style.translate = fit.dx || fit.dy ? `${Math.round(fit.dx)}px ${Math.round(fit.dy)}px` : '';
  return fit;
}

/**
 * Re-fits on window and visual-viewport changes, coalesced to one call per frame.
 * Returns a disposer. Safe to call in environments without `requestAnimationFrame`.
 */
export function watchViewport(onChange: () => void, win: Window = window): () => void {
  let scheduled = false;
  const raf = typeof win.requestAnimationFrame === 'function' ? win.requestAnimationFrame.bind(win) : (cb: () => void) => setTimeout(cb, 16);
  const schedule = (): void => {
    if (scheduled) return;
    scheduled = true;
    raf(() => {
      scheduled = false;
      onChange();
    });
  };
  win.addEventListener('resize', schedule);
  const vv = win.visualViewport;
  vv?.addEventListener('resize', schedule);
  vv?.addEventListener('scroll', schedule);
  return () => {
    win.removeEventListener('resize', schedule);
    vv?.removeEventListener('resize', schedule);
    vv?.removeEventListener('scroll', schedule);
  };
}
