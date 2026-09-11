/**
 * Drag-to-resize for the panel. A 16px grip sits on the corner opposite the dock's anchor;
 * dragging it grows or shrinks the panel away from the anchor, writing `width`/`height`
 * through the CSSOM (never a markup attribute). The arithmetic is the pure
 * `resizeFromPointer`, so the four dock orientations are unit-testable without layout;
 * `installResize` is the thin pointer-event binding around it. The size lives only for the
 * panel's lifetime: nothing is persisted, and a double-click on the grip resets it.
 */
import { MIN_HEIGHT, MIN_WIDTH } from './fit';
import type { Dock } from './fit';

export { MIN_HEIGHT, MIN_WIDTH };

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

/**
 * Upper bounds for a drag, normally the `maxWidth`/`maxHeight` that `fitElement` wrote;
 * a missing or non-finite entry means "no cap".
 */
export interface SizeCaps {
  width?: number;
  height?: number;
}

const clamp = (v: number, min: number, max: number): number => Math.max(min, Math.min(v, max));

/**
 * The size a drag from `start` to `current` gives a panel that measured `startSize` at
 * `pointerdown`. The grip is on the corner opposite the anchor, so moving the pointer
 * *towards* the anchor shrinks the panel: for a `*-right` dock a leftward move grows it, for
 * a `*-left` dock a rightward one; likewise `bottom-*` grows upward and `top-*` downward.
 * The result is never below `MIN_WIDTH`x`MIN_HEIGHT` and never above `caps`.
 */
export function resizeFromPointer(dock: Dock, start: Point, current: Point, startSize: Size, caps: SizeCaps = {}): Size {
  const dw = dock.endsWith('right') ? start.x - current.x : current.x - start.x;
  const dh = dock.startsWith('bottom') ? start.y - current.y : current.y - start.y;
  const capW = Number.isFinite(caps.width) ? (caps.width as number) : Infinity;
  const capH = Number.isFinite(caps.height) ? (caps.height as number) : Infinity;
  return {
    width: clamp(startSize.width + dw, MIN_WIDTH, capW),
    height: clamp(startSize.height + dh, MIN_HEIGHT, capH),
  };
}

/** `'1248px'` → `1248`; anything unparsable (including `''`) → `Infinity`, i.e. no cap. */
export function capOf(css: string): number {
  const n = Number.parseFloat(css);
  return Number.isFinite(n) ? n : Infinity;
}

export interface ResizeOptions {
  /** Read on every `pointerdown`, so a later dock change needs no re-install. */
  dock: () => Dock;
  /** Called after every size write (each move and the reset); the panel re-fits here. */
  onResize(): void;
}

interface DragStart extends Point, Size {
  pointerId: number;
}

/**
 * Binds the grip: `pointerdown` records the pointer and the panel's current size and takes
 * pointer capture (when the runtime has it) so the drag survives leaving the grip;
 * `pointermove` writes the clamped size; `pointerup`/`pointercancel` end the drag; `dblclick`
 * clears the explicit size so the stylesheet's defaults apply again. Returns a disposer that
 * removes every listener.
 */
export function installResize(root: HTMLElement, grip: HTMLElement, opts: ResizeOptions): () => void {
  let drag: DragStart | null = null;

  const setSize = (size: Size): void => {
    root.style.width = `${Math.round(size.width)}px`;
    root.style.height = `${Math.round(size.height)}px`;
    opts.onResize();
  };

  const onDown = (e: PointerEvent): void => {
    if (e.button !== 0 || drag) return;
    drag = { pointerId: e.pointerId, x: e.clientX, y: e.clientY, width: root.offsetWidth, height: root.offsetHeight };
    if (typeof grip.setPointerCapture === 'function') grip.setPointerCapture(e.pointerId);
    // Stops the compatibility mousedown from starting a text selection; `click` and
    // `dblclick` are unaffected, so the reset still works.
    e.preventDefault();
  };

  const onMove = (e: PointerEvent): void => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    const caps = { width: capOf(root.style.maxWidth), height: capOf(root.style.maxHeight) };
    setSize(resizeFromPointer(opts.dock(), drag, { x: e.clientX, y: e.clientY }, drag, caps));
  };

  const onEnd = (e: PointerEvent): void => {
    if (!drag || e.pointerId !== drag.pointerId) return;
    if (typeof grip.hasPointerCapture === 'function' && grip.hasPointerCapture(e.pointerId)) grip.releasePointerCapture(e.pointerId);
    drag = null;
  };

  const onReset = (): void => {
    root.style.width = '';
    root.style.height = '';
    opts.onResize();
  };

  grip.addEventListener('pointerdown', onDown);
  grip.addEventListener('pointermove', onMove);
  grip.addEventListener('pointerup', onEnd);
  grip.addEventListener('pointercancel', onEnd);
  grip.addEventListener('dblclick', onReset);
  return () => {
    drag = null;
    grip.removeEventListener('pointerdown', onDown);
    grip.removeEventListener('pointermove', onMove);
    grip.removeEventListener('pointerup', onEnd);
    grip.removeEventListener('pointercancel', onEnd);
    grip.removeEventListener('dblclick', onReset);
  };
}
