import { describe, expect, test } from 'vitest';
import { MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, describeMapping, imageShapeOf, rasterize } from '../src/panel/tensor-image';
import type { ImageShape } from '../src/panel/tensor-image';

describe('imageShapeOf', () => {
  const table: [number[], string, ImageShape | null][] = [
    [[1, 3, 224, 224], 'float32', { kind: 'rgb', layout: 'chw', height: 224, width: 224 }],
    [[3, 8, 8], 'float32', { kind: 'rgb', layout: 'chw', height: 8, width: 8 }],
    [[1, 224, 224, 3], 'uint8', { kind: 'rgb', layout: 'hwc', height: 224, width: 224 }],
    [[64, 32, 3], 'float32', { kind: 'rgb', layout: 'hwc', height: 64, width: 32 }],
    [[80, 3000], 'float32', { kind: 'gray', height: 80, width: 3000 }],
    [[1, 80, 3000], 'float32', { kind: 'gray', height: 80, width: 3000 }],
    [[1, 1, 128, 128], 'float32', { kind: 'gray', height: 128, width: 128 }],
    [[1, 8, 8], 'int64', { kind: 'gray', height: 8, width: 8 }],
    // too thin, wrong rank, wrong batch or channel count
    [[1, 7, 384], 'float32', null],
    [[1, 1, 128256], 'float32', null],
    [[1, 2, 5, 16], 'float32', null],
    [[2, 3, 16, 16], 'float32', null],
    [[1, 7], 'int64', null],
    [[1000], 'float32', null],
    [[], 'float32', null],
    [[1, 3, 16, 16, 2], 'float32', null],
    [[3, 4, 16], 'float32', null],
    // dtypes the panel cannot rasterise
    [[1, 3, 224, 224], 'float16', null],
    [[1, 80, 3000], 'string', null],
  ];
  test.each(table)('%j %s → %j', (dims, dtype, expected) => {
    expect(imageShapeOf(dims, dtype)).toEqual(expected);
  });

  test('MIN_IMAGE_SIDE is the boundary', () => {
    expect(imageShapeOf([MIN_IMAGE_SIDE, MIN_IMAGE_SIDE], 'float32')).not.toBeNull();
    expect(imageShapeOf([MIN_IMAGE_SIDE - 1, MIN_IMAGE_SIDE], 'float32')).toBeNull();
    expect(imageShapeOf([MIN_IMAGE_SIDE, MIN_IMAGE_SIDE - 1], 'float32')).toBeNull();
  });
});

describe('rasterize', () => {
  test('a 2×2 chw tensor gives exact bytes and one range per channel', () => {
    // R: 0..3 (ramp), G: constant 5, B: -1..1 with a NaN that is skipped
    const data = new Float32Array([0, 1, 2, 3, 5, 5, 5, 5, -1, NaN, 0, 1]);
    const shape: ImageShape = { kind: 'rgb', layout: 'chw', height: 2, width: 2 };
    const r = rasterize(data, shape);
    expect(r.width).toBe(2);
    expect(r.height).toBe(2);
    expect(r.ranges).toEqual([
      { min: 0, max: 3 },
      { min: 5, max: 5 },
      { min: -1, max: 1 },
    ]);
    expect(Array.from(r.rgba)).toEqual([0, 0, 0, 255, 85, 0, 0, 255, 170, 0, 128, 255, 255, 0, 255, 255]);
  });

  test('hwc reads interleaved channels', () => {
    const data = [0, 10, 20, 100, 10, 20, 0, 10, 20, 100, 10, 20];
    const r = rasterize(data, { kind: 'rgb', layout: 'hwc', height: 2, width: 2 });
    expect(r.ranges).toEqual([
      { min: 0, max: 100 },
      { min: 10, max: 10 },
      { min: 20, max: 20 },
    ]);
    expect(Array.from(r.rgba)).toEqual([0, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0, 255]);
  });

  test('a gray gradient maps min→0 and max→255 with one global range', () => {
    const w = 16;
    const h = 8;
    const data = Float32Array.from({ length: w * h }, (_, i) => (i % w) - 4);
    const r = rasterize(data, { kind: 'gray', height: h, width: w });
    expect(r.ranges).toEqual([{ min: -4, max: 11 }]);
    expect(Array.from(r.rgba.slice(0, 4))).toEqual([0, 0, 0, 255]);
    expect(Array.from(r.rgba.slice((w - 1) * 4, w * 4))).toEqual([255, 255, 255, 255]);
    const rows = Array.from({ length: h }, (_, y) => Array.from(r.rgba.slice(y * w * 4, (y + 1) * w * 4)));
    for (const row of rows) expect(row).toEqual(rows[0]);
    const grays = rows[0].filter((_, i) => i % 4 === 0);
    for (let i = 1; i < grays.length; i++) expect(grays[i]).toBeGreaterThan(grays[i - 1]);
  });

  test('a constant tensor (min === max) rasterises to opaque black', () => {
    const r = rasterize(new Float32Array(64).fill(0.5), { kind: 'gray', height: 8, width: 8 });
    expect(r.ranges).toEqual([{ min: 0.5, max: 0.5 }]);
    expect(r.rgba.every((b, i) => (i % 4 === 3 ? b === 255 : b === 0))).toBe(true);
  });

  test('a 3-channel tensor whose channels differ is normalised per channel', () => {
    const hw = 64;
    const data = new Float32Array(3 * hw);
    for (let i = 0; i < hw; i++) {
      data[i] = i; // R 0..63
      data[hw + i] = -100 + i; // G -100..-37
      data[2 * hw + i] = 1000; // B flat
    }
    const r = rasterize(data, { kind: 'rgb', layout: 'chw', height: 8, width: 8 });
    expect(r.ranges).toEqual([
      { min: 0, max: 63 },
      { min: -100, max: -37 },
      { min: 1000, max: 1000 },
    ]);
    expect(Array.from(r.rgba.slice(0, 4))).toEqual([0, 0, 0, 255]);
    expect(Array.from(r.rgba.slice(63 * 4, 64 * 4))).toEqual([255, 255, 0, 255]);
  });

  test('[1, 80, 3000] downsamples columns to MAX_IMAGE_SIDE with nearest-neighbour sampling', () => {
    const w = 3000;
    const h = 80;
    const data = Float32Array.from({ length: w * h }, (_, i) => i % w); // value = column
    const r = rasterize(data, { kind: 'gray', height: h, width: w });
    expect(r.width).toBe(MAX_IMAGE_SIDE);
    expect(r.height).toBe(h);
    expect(r.rgba).toHaveLength(MAX_IMAGE_SIDE * h * 4);
    expect(r.ranges).toEqual([{ min: 0, max: w - 1 }]);
    // output column x reads source column floor(x * 3000 / 512)
    for (const x of [0, 1, 255, 511]) {
      const src = Math.floor((x * w) / MAX_IMAGE_SIDE);
      expect(r.rgba[x * 4]).toBe(Math.round((src / (w - 1)) * 255));
    }
    expect(r.rgba[511 * 4]).toBe(255);
  });

  test('rows are downsampled too, and a custom maxSide is honoured', () => {
    const r = rasterize(new Float32Array(100 * 40), { kind: 'gray', height: 100, width: 40 }, 10);
    expect(r.width).toBe(10);
    expect(r.height).toBe(10);
  });

  test('bigint data is read through Number', () => {
    const data = BigInt64Array.from({ length: 64 }, (_, i) => BigInt(i));
    const r = rasterize(data, { kind: 'gray', height: 8, width: 8 });
    expect(r.ranges).toEqual([{ min: 0, max: 63 }]);
    expect(r.rgba[63 * 4]).toBe(255);
  });

  test('throws a RangeError when the data is shorter than the shape', () => {
    expect(() => rasterize(new Float32Array(10), { kind: 'gray', height: 8, width: 8 })).toThrow(RangeError);
  });
});

describe('describeMapping', () => {
  test('gray with downsampling, rgb per channel', () => {
    const gray = rasterize(Float32Array.from({ length: 80 * 3000 }, (_, i) => (i % 3000) / 1000 - 1.234), { kind: 'gray', height: 80, width: 3000 });
    expect(describeMapping({ kind: 'gray', height: 80, width: 3000 }, gray)).toBe('gray 80×3000 → 80×512 · min -1.234 · max 1.765');

    const shape: ImageShape = { kind: 'rgb', layout: 'chw', height: 8, width: 8 };
    const data = new Float32Array(3 * 64);
    data.set([-2.118], 0);
    data.set([2.64], 63);
    data.fill(1, 64, 128);
    data.set([-1, 0.5], 128);
    const rgb = rasterize(data, shape);
    expect(describeMapping(shape, rgb)).toBe('rgb chw 8×8 · min…max R -2.118…2.64 · G 1…1 · B -1…0.5');
  });
});
