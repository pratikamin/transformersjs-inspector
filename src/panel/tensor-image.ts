/**
 * Tensor-as-image helpers, pure and DOM-free: `imageShapeOf` decides whether a tensor's
 * dims look like pixels or a 2-D map, `rasterize` turns its values into RGBA bytes with a
 * min/max mapping, `describeMapping` prints that mapping for the caption. The panel calls
 * them on a `TensorData` response fetched through `bus.request('tensor')`, so the same code
 * serves the page and the worker bridge.
 */

export type ImageShape = { kind: 'rgb'; layout: 'chw' | 'hwc'; height: number; width: number } | { kind: 'gray'; height: number; width: number };

/** Both sides of a recognised image must be at least this long; keeps `[1, 7, 384]` from becoming a 7-row strip. */
export const MIN_IMAGE_SIDE = 8;
/** Longest side of the rasterised image; larger axes are nearest-neighbour sampled down to it. */
export const MAX_IMAGE_SIDE = 512;

const sideOk = (n: number): boolean => Number.isInteger(n) && n >= MIN_IMAGE_SIDE;

/**
 * Shape rules, in order: `float16` and `string` tensors are never images; `[1,3,H,W]` and
 * `[3,H,W]` are rgb chw; `[1,H,W,3]` and `[H,W,3]` are rgb hwc; `[H,W]`, `[1,H,W]`,
 * `[1,1,H,W]` and `[1,C,T]` (C rows, T columns; Whisper's `input_features [1,80,3000]`)
 * are grayscale maps. Every rule needs both sides ≥ `MIN_IMAGE_SIDE`, so `[1,7,384]`,
 * `[1,1,128256]` and `[1,2,5,16]` are `null`.
 */
export function imageShapeOf(dims: readonly number[], dtype: string): ImageShape | null {
  if (dtype === 'float16' || dtype === 'string') return null;
  const d = dims;
  const rgb = (layout: 'chw' | 'hwc', height: number, width: number): ImageShape | null => (sideOk(height) && sideOk(width) ? { kind: 'rgb', layout, height, width } : null);
  const gray = (height: number, width: number): ImageShape | null => (sideOk(height) && sideOk(width) ? { kind: 'gray', height, width } : null);
  switch (d.length) {
    case 2:
      return gray(d[0], d[1]);
    case 3:
      if (d[0] === 3) return rgb('chw', d[1], d[2]);
      if (d[2] === 3) return rgb('hwc', d[0], d[1]);
      if (d[0] === 1) return gray(d[1], d[2]);
      return null;
    case 4:
      if (d[0] !== 1) return null;
      if (d[1] === 3) return rgb('chw', d[2], d[3]);
      if (d[3] === 3) return rgb('hwc', d[1], d[2]);
      if (d[1] === 1) return gray(d[2], d[3]);
      return null;
    default:
      return null;
  }
}

export type Range = { min: number; max: number };

export type Raster = { width: number; height: number; rgba: Uint8ClampedArray; ranges: Range[] };

const num = (v: number | bigint): number => (typeof v === 'bigint' ? Number(v) : v);

/**
 * Turns tensor values into an RGBA byte image. One pass finds each channel's min and max
 * (rgb: one range per channel; gray: a single global range), then every output pixel is
 * `(v - min) / (max - min) * 255`, rounded, alpha 255; a flat channel (`min === max`) maps
 * to 0. Sampling rule: each axis longer than `maxSide` is nearest-neighbour downsampled
 * to exactly `maxSide` — output pixel `i` reads source index `floor(i * src / out)` — so
 * `[1, 80, 3000]` becomes 512×80 and axes already within the limit are copied 1:1. Throws
 * a `RangeError` when `data` is shorter than the shape needs.
 */
export function rasterize(data: ArrayLike<number | bigint>, shape: ImageShape, maxSide: number = MAX_IMAGE_SIDE): Raster {
  const { height: srcH, width: srcW } = shape;
  const channels = shape.kind === 'rgb' ? 3 : 1;
  const needed = srcH * srcW * channels;
  if (data.length < needed) throw new RangeError(`tensor has ${data.length} values, image shape needs ${needed}`);
  const at = shape.kind === 'rgb' && shape.layout === 'chw' ? (y: number, x: number, c: number) => c * srcH * srcW + y * srcW + x : (y: number, x: number, c: number) => (y * srcW + x) * channels + c;

  const ranges: Range[] = Array.from({ length: channels }, () => ({ min: Infinity, max: -Infinity }));
  for (let i = 0; i < needed; i++) {
    const v = num(data[i]);
    if (!Number.isFinite(v)) continue;
    const r = shape.kind === 'rgb' && shape.layout === 'chw' ? ranges[Math.floor(i / (srcH * srcW))] : ranges[i % channels];
    if (v < r.min) r.min = v;
    if (v > r.max) r.max = v;
  }
  for (const r of ranges) {
    if (!Number.isFinite(r.min)) r.min = r.max = 0;
  }

  const width = Math.min(srcW, maxSide);
  const height = Math.min(srcH, maxSide);
  const rgba = new Uint8ClampedArray(width * height * 4);
  const scale = ranges.map((r) => (r.max > r.min ? 255 / (r.max - r.min) : 0));
  for (let oy = 0; oy < height; oy++) {
    const sy = Math.floor((oy * srcH) / height);
    for (let ox = 0; ox < width; ox++) {
      const sx = Math.floor((ox * srcW) / width);
      const o = (oy * width + ox) * 4;
      for (let c = 0; c < 3; c++) {
        const ch = channels === 1 ? 0 : c;
        const v = num(data[at(sy, sx, ch)]);
        rgba[o + c] = Number.isFinite(v) ? Math.round((v - ranges[ch].min) * scale[ch]) : 0;
      }
      rgba[o + 3] = 255;
    }
  }
  return { width, height, rgba, ranges };
}

const fmt = (v: number): string => (Number.isFinite(v) ? String(Number(v.toPrecision(4))) : String(v));

/**
 * Caption text for the mapping, e.g. `gray 80×3000 → 80×512 · min -1.234 · max 2.5` or
 * `rgb chw 224×224 · min…max R -2.118…2.64 · G -2.036…2.429 · B -1.804…2.64` (the `→` part
 * appears only when the raster was downsampled).
 */
export function describeMapping(shape: ImageShape, raster: Raster): string {
  const size = `${shape.height}×${shape.width}`;
  const sampled = raster.width !== shape.width || raster.height !== shape.height ? ` → ${raster.height}×${raster.width}` : '';
  if (shape.kind === 'gray') {
    const r = raster.ranges[0];
    return `gray ${size}${sampled} · min ${fmt(r.min)} · max ${fmt(r.max)}`;
  }
  const channels = ['R', 'G', 'B'].map((name, i) => `${name} ${fmt(raster.ranges[i].min)}…${fmt(raster.ranges[i].max)}`);
  return `rgb ${shape.layout} ${size}${sampled} · min…max ${channels.join(' · ')}`;
}
