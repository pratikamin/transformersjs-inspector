/**
 * Turns arbitrary pipeline inputs and results into small, structured-clone-safe values
 * for `call:start` / `result` events. Tensors are reduced to summaries (registered in
 * the `TensorStore` when one is given), typed arrays and long strings are truncated,
 * bigint becomes number, and anything unclonable is dropped.
 *
 * Media inputs get a bounded preview at capture time (v1.1): audio a downsampled min/max
 * waveform, RawImage-shaped objects a small data-URL thumbnail drawn through an injectable
 * canvas factory. No enrichment may throw into the pipeline call: each is its own
 * `try/catch` and `previewInput` as a whole falls back to `{ kind: 'other' }`.
 */
import type { InputPreview } from './events';
import type { TensorStore } from './store';
import { isTensorLike, summarizeTensor } from './summarize';

export const MAX_STRING = 2000;
export const MAX_ARRAY = 64;
export const MAX_DEPTH = 6;

/**
 * Waveform budget: `WAVEFORM_BUCKETS` min/max pairs → at most 400 numbers of 3 decimals in
 * [-1, 1], i.e. ≤ ~3 KB of JSON per audio preview. Beyond `MAX_WAVEFORM_SAMPLES` a stride
 * skips samples so at most 4 M reads happen however long the clip is (a 10-minute
 * 16 kHz clip is 9.6 M samples → stride 3).
 */
export const WAVEFORM_BUCKETS = 200;
export const MAX_WAVEFORM_SAMPLES = 1 << 22;
/** Thumbnail budget: longest side ≤ 96 px, JPEG at quality 0.75 (typically 3–6 KB), hard cap 32 768 chars. */
export const THUMB_MAX = 96;
export const MAX_THUMB_CHARS = 32768;
const THUMB_TYPE = 'image/jpeg';
const THUMB_QUALITY = 0.75;

export type CloneSafeOptions = { maxString?: number; maxArray?: number; maxDepth?: number };

/** The slice of a `<canvas>` (or an equivalent) that `thumbnailOf` needs; a real canvas element satisfies it. */
export interface CanvasLike {
  width: number;
  height: number;
  getContext(kind: '2d'): { createImageData(w: number, h: number): { data: Uint8ClampedArray }; putImageData(img: unknown, x: number, y: number): void } | null;
  toDataURL(type?: string, quality?: number): string;
}

/** Makes a `w`×`h` canvas, or `null` where none is available (a worker, Node). */
export type CanvasFactory = (w: number, h: number) => CanvasLike | null;

export type PreviewOptions = { createCanvas?: CanvasFactory | null };

/** RawImage-shaped input (`RawImage` in Transformers.js has exactly these fields; `data` is a `Uint8ClampedArray`). */
export type ImageLike = { width: number; height: number; channels?: number; data: ArrayLike<number> };

/**
 * Process-wide canvas factory used by `previewInput` when the caller passes none. It is
 * `null` until the host registers one: `OffscreenCanvas` cannot produce a data URL
 * synchronously and this module never looks at the DOM, so a `<canvas>` factory has to be
 * injected from the page side (the panel / demo does it). Without one, image previews are
 * metadata-only, which is also the worker behaviour.
 */
let defaultCanvasFactory: CanvasFactory | null = null;

export function setCanvasFactory(factory: CanvasFactory | null): void {
  defaultCanvasFactory = factory;
}

export function getCanvasFactory(): CanvasFactory | null {
  return defaultCanvasFactory;
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null;
const isNum = (x: unknown): x is number => typeof x === 'number';
const isStr = (x: unknown): x is string => typeof x === 'string';
const isHttp = (s: string): boolean => /^https?:\/\//i.test(s);
const isSamples = (x: unknown): x is Float32Array | Float64Array => x instanceof Float32Array || x instanceof Float64Array;
const isBlob = (x: unknown): x is Blob => typeof Blob !== 'undefined' && x instanceof Blob;
const isUrl = (x: unknown): x is URL => typeof URL !== 'undefined' && x instanceof URL;
const isImageLike = (x: Record<string, unknown>): x is Record<string, unknown> & ImageLike =>
  isNum(x.width) && isNum(x.height) && (ArrayBuffer.isView(x.data) || Array.isArray(x.data));
const clip = (s: string, max: number): string => (s.length > max ? s.slice(0, max) : s);
const round3 = (v: number): number => Math.round(v * 1000) / 1000 + 0; // `+ 0` folds -0 into 0 (JSON has no -0)

export function previewInput(x: unknown, opts: PreviewOptions = {}): InputPreview {
  try {
    return previewInputUnsafe(x, opts);
  } catch {
    try {
      return { kind: 'other', json: toCloneSafe(x) };
    } catch {
      return { kind: 'other', json: '[unpreviewable]' };
    }
  }
}

function previewInputUnsafe(x: unknown, opts: PreviewOptions): InputPreview {
  if (isStr(x)) return isHttp(x) ? { kind: 'image', src: x } : { kind: 'text', text: clip(x, MAX_STRING) };
  if (Array.isArray(x) && x.every(isStr)) return { kind: 'texts', texts: x.map((s) => clip(s, MAX_STRING)) };
  if (isSamples(x)) return audioPreview(x);
  if (isUrl(x)) return { kind: 'image', src: x.href };
  // Decoding a Blob / URL / http input is async, so those keep `src` metadata only: no fetching, no drawing.
  if (isBlob(x)) return { kind: 'image', src: `blob:${x.type || 'application/octet-stream'};${x.size}` };
  if (isObj(x) && !isTensorLike(x)) {
    if ('audio' in x && (isSamples(x.audio) || isObj(x.audio) || Array.isArray(x.audio))) {
      const audio = x.audio as ArrayLike<unknown>;
      const rate = x.sampling_rate;
      if (isSamples(audio)) return audioPreview(audio, rate);
      // A chunked `Float32Array[]` (or any other array-like): total sample count, no waveform.
      const samples = Array.isArray(audio) && audio.every(isSamples) ? audio.reduce((n, c) => n + c.length, 0) : isNum(audio.length) ? audio.length : 0;
      return { kind: 'audio', samples, ...(isNum(rate) ? { sampleRate: rate } : {}) };
    }
    if (isNum(x.width) && isNum(x.height)) {
      const preview: InputPreview = { kind: 'image', width: x.width, height: x.height, ...(isNum(x.channels) ? { channels: x.channels } : {}) };
      if (isImageLike(x)) {
        const thumb = thumbnailOf(x, { createCanvas: opts.createCanvas === undefined ? defaultCanvasFactory : opts.createCanvas });
        if (thumb !== undefined) preview.thumb = thumb;
      }
      return preview;
    }
  }
  return { kind: 'other', json: toCloneSafe(x) };
}

// ---- audio -----------------------------------------------------------------

/**
 * Interleaved `[min0, max0, min1, max1, …]` over `buckets` equal slices of `samples`, one
 * pass, each value rounded to 3 decimals and clamped to [-1, 1]. Fewer samples than
 * buckets → one bucket per sample. NaN samples are skipped; a bucket with nothing readable
 * is `0, 0`. At most `MAX_WAVEFORM_SAMPLES` reads happen (a stride skips the rest).
 */
export function waveformOf(samples: ArrayLike<number>, buckets: number = WAVEFORM_BUCKETS): number[] {
  const n = samples.length;
  const b = Math.min(Math.max(0, Math.floor(buckets)), n);
  const out: number[] = [];
  if (b === 0) return out;
  const stride = Math.max(1, Math.ceil(n / MAX_WAVEFORM_SAMPLES));
  for (let k = 0; k < b; k++) {
    const start = Math.floor((k * n) / b);
    const end = Math.max(start + 1, Math.floor(((k + 1) * n) / b));
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = start; i < end; i += stride) {
      const v = Number(samples[i]);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo > hi) {
      lo = 0;
      hi = 0;
    }
    out.push(round3(Math.min(1, Math.max(-1, lo))), round3(Math.min(1, Math.max(-1, hi))));
  }
  return out;
}

/** `{ kind: 'audio' }` preview with `duration` (seconds, 3 dp) when the rate is known and `peaks` when there are samples. */
export function audioPreview(samples: ArrayLike<number>, sampleRate?: unknown): InputPreview {
  const preview: InputPreview = { kind: 'audio', samples: samples.length };
  if (isNum(sampleRate) && Number.isFinite(sampleRate) && sampleRate > 0) {
    preview.sampleRate = sampleRate;
    preview.duration = round3(samples.length / sampleRate);
  }
  if (samples.length > 0) {
    try {
      preview.peaks = waveformOf(samples);
    } catch {
      // no waveform; the count and rate still describe the input
    }
  }
  return preview;
}

// ---- image -----------------------------------------------------------------

/**
 * A `data:image/jpeg` thumbnail of a RawImage-shaped input, at most `max` px on the long
 * side, drawn with nearest-neighbour sampling through `createCanvas`. `undefined` when
 * there is no canvas factory (the caller then keeps metadata only), when the canvas has no
 * 2D context (happy-dom), when `data` does not match `width·height·channels`, when the
 * result is not a `data:image/` URL, when it exceeds `MAX_THUMB_CHARS`, or when anything
 * throws.
 */
export function thumbnailOf(img: ImageLike, opts: { max?: number; createCanvas?: CanvasFactory | null } = {}): string | undefined {
  try {
    const createCanvas = opts.createCanvas ?? null;
    if (createCanvas === null) return undefined;
    const max = opts.max ?? THUMB_MAX;
    const { width: w, height: h, data } = img;
    if (!Number.isInteger(w) || !Number.isInteger(h) || w <= 0 || h <= 0 || !(max > 0)) return undefined;
    const channels = img.channels ?? data.length / (w * h);
    if (channels !== 1 && channels !== 2 && channels !== 3 && channels !== 4) return undefined;
    if (data.length !== w * h * channels) return undefined;

    const scale = Math.min(1, max / Math.max(w, h));
    const tw = Math.max(1, Math.round(w * scale));
    const th = Math.max(1, Math.round(h * scale));
    const canvas = createCanvas(tw, th);
    if (canvas === null) return undefined;
    canvas.width = tw;
    canvas.height = th;
    const ctx = canvas.getContext('2d');
    if (ctx === null) return undefined;
    const image = ctx.createImageData(tw, th);
    const rgba = image.data;
    let o = 0;
    for (let y = 0; y < th; y++) {
      const sy = Math.floor((y * h) / th);
      for (let x = 0; x < tw; x++) {
        const sx = Math.floor((x * w) / tw);
        const p = (sy * w + sx) * channels;
        const c0 = data[p];
        switch (channels) {
          case 1:
            rgba[o] = rgba[o + 1] = rgba[o + 2] = c0;
            rgba[o + 3] = 255;
            break;
          case 2:
            rgba[o] = rgba[o + 1] = rgba[o + 2] = c0;
            rgba[o + 3] = data[p + 1];
            break;
          case 3:
            rgba[o] = c0;
            rgba[o + 1] = data[p + 1];
            rgba[o + 2] = data[p + 2];
            rgba[o + 3] = 255;
            break;
          default:
            rgba[o] = c0;
            rgba[o + 1] = data[p + 1];
            rgba[o + 2] = data[p + 2];
            rgba[o + 3] = data[p + 3];
        }
        o += 4;
      }
    }
    ctx.putImageData(image, 0, 0);
    const url = canvas.toDataURL(THUMB_TYPE, THUMB_QUALITY);
    if (!isStr(url) || !url.startsWith('data:image/') || url.length > MAX_THUMB_CHARS) return undefined;
    return url;
  } catch {
    return undefined;
  }
}

// ---- clone-safe walk -------------------------------------------------------

/** Sentinel for values that are dropped from their container (undefined, functions, symbols). */
const DROP: unique symbol = Symbol('drop');

type Ctx = { store: TensorStore | undefined; maxString: number; maxArray: number; maxDepth: number; stack: Set<object> };

export function toCloneSafe(v: unknown, store?: TensorStore, opts: CloneSafeOptions = {}): unknown {
  const ctx: Ctx = {
    store,
    maxString: opts.maxString ?? MAX_STRING,
    maxArray: opts.maxArray ?? MAX_ARRAY,
    maxDepth: opts.maxDepth ?? MAX_DEPTH,
    stack: new Set(),
  };
  const out = walk(v, '', 0, ctx);
  return out === DROP ? undefined : out;
}

function walk(v: unknown, name: string, depth: number, ctx: Ctx): unknown {
  switch (typeof v) {
    case 'string':
      return clip(v, ctx.maxString);
    case 'number':
    case 'boolean':
      return v;
    case 'bigint':
      return Number(v);
    case 'undefined':
    case 'function':
    case 'symbol':
      return DROP;
    case 'object':
      break;
  }
  if (v === null) return null;
  const o = v as object;

  if (isTensorLike(o)) {
    return { $tensor: ctx.store ? ctx.store.put(o, name) : summarizeTensor(o, name, '') };
  }
  if (o instanceof Error) return { message: o.message };
  if (o instanceof Date) return o.toISOString();
  if (ArrayBuffer.isView(o)) {
    if (o instanceof DataView) return { $truncated: o.byteLength, values: [] };
    return truncatedArray(o as unknown as ArrayLike<unknown>, ctx, (x) => Number(x));
  }

  if (ctx.stack.has(o)) return '[cycle]';
  if (depth >= ctx.maxDepth) return '[deep]';
  ctx.stack.add(o);
  try {
    if (Array.isArray(o)) {
      return truncatedArray(o, ctx, (x, i) => walk(x, `${name}[${i}]`, depth + 1, ctx));
    }
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(o)) {
      const val = walk((o as Record<string, unknown>)[key], key, depth + 1, ctx);
      if (val !== DROP) out[key] = val;
    }
    return out;
  } finally {
    ctx.stack.delete(o);
  }
}

/** First `maxArray` converted items; longer inputs become `{ $truncated: total, values }`. */
function truncatedArray(src: ArrayLike<unknown>, ctx: Ctx, conv: (x: unknown, i: number) => unknown): unknown {
  const total = src.length;
  const n = Math.min(total, ctx.maxArray);
  const values: unknown[] = [];
  for (let i = 0; i < n; i++) {
    const c = conv(src[i], i);
    if (c !== DROP) values.push(c);
  }
  return total > ctx.maxArray ? { $truncated: total, values } : values;
}
