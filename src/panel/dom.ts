/**
 * Tiny DOM helpers for the panel. `h()` only ever sets `textContent`, `className`,
 * `dataset` entries, `role`/`aria-*` attributes and a few whitelisted properties: no markup strings and no inline
 * style attributes, so the panel works under a strict CSP and never interprets HTML.
 * `svg()` builds SVG the same way through `createElementNS` + `setAttribute`, and the only
 * `src` ever written is a `data:image/` URL on an `<img>`, so the panel never fetches.
 */

export type Child = Node | string | number | null | undefined | false | Child[];

export interface Attrs {
  class?: string;
  /** `data-*` entries, camelCase keys as in `el.dataset` (`{ callId: 'c1' }` → `data-call-id="c1"`). */
  data?: Record<string, string | number>;
  title?: string;
  /** `<button type>`; defaults to `'button'` for buttons so they never submit a host form. */
  type?: 'button' | 'submit' | 'reset';
  hidden?: boolean;
  colSpan?: number;
  /** Applied only to `<img>` and only when it starts with `data:image/`; a remote URL would make the panel fetch. */
  src?: string;
  /** Pixel size, applied only to `<img>` and `<canvas>`. */
  width?: number;
  height?: number;
  /** ARIA `role`, set as an attribute. */
  role?: string;
  /** `aria-*` entries (`{ pressed: 'true' }` → `aria-pressed="true"`), set as attributes. */
  aria?: Record<string, string>;
}

const DATA_IMAGE = 'data:image/';

/** True for the one kind of `src` the panel will ever set. */
export function isDataImageUrl(s: unknown): s is string {
  return typeof s === 'string' && s.startsWith(DATA_IMAGE);
}

function append(el: Element, child: Child): void {
  if (child === null || child === undefined || child === false) return;
  if (Array.isArray(child)) {
    for (const c of child) append(el, c);
    return;
  }
  if (typeof child === 'string' || typeof child === 'number') {
    el.appendChild(document.createTextNode(String(child)));
    return;
  }
  el.appendChild(child);
}

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs?: Attrs | null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    if (attrs.class !== undefined) el.className = attrs.class;
    if (attrs.data) {
      for (const [k, v] of Object.entries(attrs.data)) el.dataset[k] = String(v);
    }
    if (attrs.title !== undefined) el.title = attrs.title;
    if (attrs.role !== undefined) el.setAttribute('role', attrs.role);
    if (attrs.aria) {
      for (const [k, v] of Object.entries(attrs.aria)) el.setAttribute(`aria-${k}`, v);
    }
    if (attrs.hidden !== undefined) el.hidden = attrs.hidden;
    if (attrs.colSpan !== undefined && el instanceof HTMLTableCellElement) el.colSpan = attrs.colSpan;
    if (attrs.type !== undefined && el instanceof HTMLButtonElement) el.type = attrs.type;
    if (attrs.src !== undefined && el instanceof HTMLImageElement && isDataImageUrl(attrs.src)) el.src = attrs.src;
    if (el instanceof HTMLImageElement || el instanceof HTMLCanvasElement) {
      if (attrs.width !== undefined) el.width = attrs.width;
      if (attrs.height !== undefined) el.height = attrs.height;
    }
  }
  if (el instanceof HTMLButtonElement && (!attrs || attrs.type === undefined)) el.type = 'button';
  for (const c of children) append(el, c);
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

export type SvgTag = 'svg' | 'path' | 'polyline' | 'rect';

/**
 * An SVG element via `createElementNS`; attributes are set one by one from computed numbers
 * and strings (a `viewBox`, a path `d`), never from host text, and there is no markup parse.
 */
export function svg(tag: SvgTag, attrs: Record<string, string | number> = {}, ...children: Child[]): SVGElement {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  for (const c of children) append(el, c);
  return el;
}

/** Removes every child of `el` (via `textContent`, never a markup string). */
export function empty(el: Element): void {
  el.textContent = '';
}

// ---- formatting ---------------------------------------------------------------

/** `12.7 ms`, `0.40 ms`, `1.25 s`; `…` while unknown. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '…';
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  if (ms >= 1) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(2)} ms`;
}

const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];

/** `0 B`, `512 B`, `1.5 KiB`, `64.0 MiB`. */
export function fmtBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '?';
  let v = bytes;
  let u = 0;
  while (v >= 1024 && u < BYTE_UNITS.length - 1) {
    v /= 1024;
    u++;
  }
  return u === 0 ? `${Math.round(v)} ${BYTE_UNITS[u]}` : `${v.toFixed(1)} ${BYTE_UNITS[u]}`;
}

/** Numbers to 4 significant digits (`-0.03470` → `-0.0347`, `128256` → `128300`); strings pass through. */
export function fmtNum(v: number | string | bigint | boolean | null | undefined): string {
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    return String(Number(v.toPrecision(4)));
  }
  if (typeof v === 'bigint' || typeof v === 'boolean') return String(v);
  if (v === null || v === undefined) return '';
  return v;
}

/** `[1, 7, 384]`; `[]` for scalars. */
export function fmtDims(dims: readonly number[]): string {
  return `[${dims.join(', ')}]`;
}
