import { describe, expect, test } from 'vitest';
import type { InputPreview, InspectorEvent } from '../src/events';
import { isInputPreview, isInspectorEvent } from '../src/events';
import type { CanvasFactory } from '../src/preview';
import {
  MAX_ARRAY,
  MAX_STRING,
  MAX_THUMB_CHARS,
  MAX_WAVEFORM_SAMPLES,
  THUMB_MAX,
  WAVEFORM_BUCKETS,
  audioPreview,
  getCanvasFactory,
  previewInput,
  setCanvasFactory,
  thumbnailOf,
  toCloneSafe,
  waveformOf,
} from '../src/preview';
import { TensorStore } from '../src/store';
import { FAKE_THUMB, fakeCanvas, fakeRawImage, fakeTensor, fixtureEvents, fixtureMediaEvents } from './fakes';

const roundTrips = (v: unknown) => {
  expect(structuredClone(v)).toEqual(v);
  expect(JSON.parse(JSON.stringify(v))).toEqual(v);
};

/** The shape the page side registers: a real `<canvas>` element satisfies `CanvasLike` (type-checked here, only called where a DOM exists). */
const domCanvasFactory: CanvasFactory = (w, h) => {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
};

const ramp = (n: number): Float32Array => Float32Array.from({ length: n }, (_, i) => (2 * i) / (n - 1) - 1);
const audioOf = (p: InputPreview) => p as Extract<InputPreview, { kind: 'audio' }>;
const imageOf = (p: InputPreview) => p as Extract<InputPreview, { kind: 'image' }>;
const throwingFactory: CanvasFactory = () => {
  throw new Error('factory should not be called');
};

describe('previewInput', () => {
  test('text: strings, truncated to 2000 chars', () => {
    expect(previewInput('the quick brown fox.')).toEqual({ kind: 'text', text: 'the quick brown fox.' });
    const long = 'x'.repeat(5000);
    const p = previewInput(long);
    expect(p.kind).toBe('text');
    expect((p as { text: string }).text).toHaveLength(MAX_STRING);
  });

  test('texts: arrays of strings', () => {
    expect(previewInput(['hi', 'hello world'])).toEqual({ kind: 'texts', texts: ['hi', 'hello world'] });
    expect(previewInput([])).toEqual({ kind: 'texts', texts: [] });
  });

  test('image: RawImage-shaped objects carry width/height/channels; no thumb without a canvas factory', () => {
    expect(getCanvasFactory()).toBeNull();
    expect(previewInput({ width: 224, height: 224, channels: 3, data: new Uint8ClampedArray(224 * 224 * 3) })).toEqual({
      kind: 'image',
      width: 224,
      height: 224,
      channels: 3,
    });
    expect(previewInput({ width: 8, height: 4 })).toEqual({ kind: 'image', width: 8, height: 4 });
  });

  test('image: an injected canvas factory adds a thumb; a factory returning null degrades to metadata (workers)', () => {
    const canvas = fakeCanvas();
    const p = previewInput(fakeRawImage(200, 100, 4), { createCanvas: () => canvas });
    expect(p).toEqual({ kind: 'image', width: 200, height: 100, channels: 4, thumb: FAKE_THUMB });
    expect(canvas.puts[0].image).toMatchObject({ width: 96, height: 48 });
    expect(previewInput(fakeRawImage(200, 100, 4), { createCanvas: () => null })).toEqual({ kind: 'image', width: 200, height: 100, channels: 4 });
    // `{ width, height }` without pixel data never touches the factory
    expect(previewInput({ width: 8, height: 4 }, { createCanvas: throwingFactory })).toEqual({ kind: 'image', width: 8, height: 4 });
  });

  test('image: the process-wide factory from setCanvasFactory is the default; an explicit null opts out', () => {
    const canvas = fakeCanvas();
    setCanvasFactory(() => canvas);
    try {
      expect(getCanvasFactory()).not.toBeNull();
      expect(imageOf(previewInput(fakeRawImage(10, 10, 3))).thumb).toBe(FAKE_THUMB);
      expect(imageOf(previewInput(fakeRawImage(10, 10, 3), { createCanvas: null })).thumb).toBeUndefined();
    } finally {
      setCanvasFactory(null);
    }
    expect(imageOf(previewInput(fakeRawImage(10, 10, 3))).thumb).toBeUndefined();
  });

  test('image: http(s) strings, URL and Blob keep src only, even with a canvas factory', () => {
    expect(previewInput('https://example.com/cat.png', { createCanvas: throwingFactory })).toEqual({ kind: 'image', src: 'https://example.com/cat.png' });
    expect(previewInput('http://example.com/cat.png')).toEqual({ kind: 'image', src: 'http://example.com/cat.png' });
    expect(previewInput(new URL('https://example.com/dog.jpg'), { createCanvas: throwingFactory })).toEqual({ kind: 'image', src: 'https://example.com/dog.jpg' });
    const p = previewInput(new Blob([new Uint8Array(10)], { type: 'image/png' }), { createCanvas: throwingFactory });
    expect(p).toEqual({ kind: 'image', src: 'blob:image/png;10' });
    expect(previewInput(new Blob([new Uint8Array(3)]))).toEqual({ kind: 'image', src: 'blob:application/octet-stream;3' });
  });

  test('audio: Float32Array / Float64Array samples get peaks; { audio, sampling_rate } (RawAudio) adds sampleRate and duration', () => {
    const f32 = previewInput(ramp(16000));
    expect(f32).toMatchObject({ kind: 'audio', samples: 16000 });
    expect(audioOf(f32).peaks).toHaveLength(2 * WAVEFORM_BUCKETS);
    expect(audioOf(f32).sampleRate).toBeUndefined();
    expect(audioOf(f32).duration).toBeUndefined();
    expect(previewInput(new Float64Array(8))).toEqual({ kind: 'audio', samples: 8, peaks: new Array<number>(16).fill(0) });

    const raw = previewInput({ audio: ramp(32000), sampling_rate: 16000 });
    expect(raw).toMatchObject({ kind: 'audio', samples: 32000, sampleRate: 16000, duration: 2 });
    expect(audioOf(raw).peaks).toHaveLength(2 * WAVEFORM_BUCKETS);
    expect(previewInput({ audio: ramp(24001), sampling_rate: 16000 })).toMatchObject({ samples: 24001, duration: 1.5 });
    expect(previewInput({ audio: new Float32Array(5) })).toEqual({ kind: 'audio', samples: 5, peaks: new Array<number>(10).fill(0) });
    expect(previewInput(new Float32Array(0))).toEqual({ kind: 'audio', samples: 0 });
  });

  test('audio: chunked Float32Array[] gives the summed sample count and no peaks; other array-likes give a count only', () => {
    expect(previewInput({ audio: [new Float32Array(3), new Float32Array(4)], sampling_rate: 8000 })).toEqual({ kind: 'audio', samples: 7, sampleRate: 8000 });
    expect(previewInput({ audio: new Int16Array(12) })).toEqual({ kind: 'audio', samples: 12 });
    expect(previewInput({ audio: [0.1, 0.2, 0.3] })).toEqual({ kind: 'audio', samples: 3 });
    expect(previewInput({ audio: {} })).toEqual({ kind: 'audio', samples: 0 });
    expect(previewInput({ audio: new Float32Array(4), sampling_rate: 'fast' })).toEqual({ kind: 'audio', samples: 4, peaks: new Array<number>(8).fill(0) });
  });

  test('other: everything else is toCloneSafe()d', () => {
    expect(previewInput(42)).toEqual({ kind: 'other', json: 42 });
    expect(previewInput({ question: 'why?', context: 'because', n: 3n })).toEqual({ kind: 'other', json: { question: 'why?', context: 'because', n: 3 } });
    expect(previewInput(['a', 1])).toEqual({ kind: 'other', json: ['a', 1] });
    expect(previewInput(null)).toEqual({ kind: 'other', json: null });
    const tensor = fakeTensor({ type: 'int64', dims: [1, 2], data: [1, 2] });
    const p = previewInput(tensor);
    expect(p.kind).toBe('other');
    expect((p as { json: { $tensor: unknown } }).json.$tensor).toMatchObject({ dtype: 'int64', dims: [1, 2], head: [1, 2] });
  });

  test('never throws: a getter that throws falls back to other, and an unwalkable value to [unpreviewable]', () => {
    const hostile = {
      width: 4,
      height: 4,
      channels: 1,
      get data(): Uint8ClampedArray {
        throw new Error('no pixels for you');
      },
    };
    // the throwing getter is hit inside the image branch; the fallback re-walks the object, which throws again
    expect(previewInput(hostile, { createCanvas: () => fakeCanvas() })).toEqual({ kind: 'other', json: '[unpreviewable]' });
    const audioHostile = {
      get audio(): Float32Array {
        throw new Error('no samples');
      },
    };
    expect(previewInput(audioHostile)).toEqual({ kind: 'other', json: '[unpreviewable]' });
    // a throwing canvas factory only costs the thumb
    const image = { width: 2, height: 2, channels: 1, data: new Uint8ClampedArray(4) };
    expect(previewInput(image, { createCanvas: throwingFactory })).toEqual({ kind: 'image', width: 2, height: 2, channels: 1 });
  });

  test('every branch survives structuredClone and JSON and passes isInputPreview', () => {
    const previews: InputPreview[] = [
      previewInput('hi'),
      previewInput(['a', 'b']),
      previewInput({ width: 1, height: 1, channels: 1 }),
      previewInput(fakeRawImage(300, 200, 3), { createCanvas: () => fakeCanvas() }),
      previewInput('https://x.y/z.png'),
      previewInput(new Float32Array(3)),
      previewInput(ramp(5000)),
      previewInput({ audio: new Float64Array(3), sampling_rate: 8000 }),
      previewInput({ audio: [new Float32Array(2)], sampling_rate: 8000 }),
      previewInput({ a: 1n, b: [new Int32Array([1, 2])] }),
    ];
    for (const p of previews) {
      roundTrips(p);
      expect(isInputPreview(p)).toBe(true);
    }
  });
});

describe('waveformOf', () => {
  test('a 1000-sample ramp gives 200 monotone min/max pairs at 3 dp', () => {
    const peaks = waveformOf(ramp(1000));
    expect(peaks).toHaveLength(2 * WAVEFORM_BUCKETS);
    for (let i = 0; i < peaks.length; i += 2) {
      expect(peaks[i]).toBeLessThanOrEqual(peaks[i + 1]);
      if (i > 0) expect(peaks[i]).toBeGreaterThan(peaks[i - 2]);
      expect(peaks[i]).toBe(Math.round(peaks[i] * 1000) / 1000);
    }
    expect(peaks[0]).toBe(-1);
    expect(peaks[peaks.length - 1]).toBe(1);
  });

  test('fewer samples than buckets: one pair per sample, in order', () => {
    expect(waveformOf(Float32Array.from([0.5, -0.25, 0, 1, -1, 0.125, 0.75, -0.5]))).toEqual([0.5, 0.5, -0.25, -0.25, 0, 0, 1, 1, -1, -1, 0.125, 0.125, 0.75, 0.75, -0.5, -0.5]);
    expect(waveformOf(new Float32Array(0))).toEqual([]);
    expect(waveformOf([0.2], 0)).toEqual([]);
  });

  test('custom bucket counts and plain arrays', () => {
    expect(waveformOf([1, 2, 3, 4], 2)).toEqual([1, 1, 1, 1]); // clamped to [-1, 1]
    expect(waveformOf([-0.5, 0.5, -0.25, 0.25], 2)).toEqual([-0.5, 0.5, -0.25, 0.25]);
    expect(waveformOf([0.1, 0.2, 0.3], 2)).toEqual([0.1, 0.1, 0.2, 0.3]);
  });

  test('clamps to [-1, 1], rounds to 3 dp, skips NaN and never emits -0', () => {
    expect(waveformOf([5, -5], 1)).toEqual([-1, 1]);
    expect(waveformOf([Infinity, -Infinity], 1)).toEqual([-1, 1]);
    expect(waveformOf([0.12345, 0.98765], 1)).toEqual([0.123, 0.988]);
    expect(waveformOf([NaN, 0.5, NaN], 1)).toEqual([0.5, 0.5]);
    expect(waveformOf([NaN, NaN], 1)).toEqual([0, 0]);
    const peaks = waveformOf([-0.0004, -0.0001], 1);
    expect(peaks).toEqual([0, 0]);
    expect(Object.is(peaks[0], -0)).toBe(false);
    expect(Object.is(peaks[1], -0)).toBe(false);
  });

  test('a 5 M-sample Float32Array completes with at most 400 numbers and ≤ MAX_WAVEFORM_SAMPLES reads', () => {
    const n = 5_000_000;
    expect(n).toBeGreaterThan(MAX_WAVEFORM_SAMPLES);
    let reads = 0;
    const samples = new Proxy(new Float32Array(n), {
      get(target, prop) {
        if (typeof prop === 'string' && /^\d+$/.test(prop)) reads++;
        return Reflect.get(target, prop) as unknown; // no receiver: TypedArray getters reject the proxy
      },
    }) as unknown as ArrayLike<number>;
    const peaks = waveformOf(samples);
    expect(peaks).toHaveLength(2 * WAVEFORM_BUCKETS);
    expect(peaks.length).toBeLessThanOrEqual(400);
    expect(reads).toBeLessThanOrEqual(MAX_WAVEFORM_SAMPLES);
    expect(reads).toBeGreaterThan(WAVEFORM_BUCKETS);
    expect(JSON.stringify(waveformOf(ramp(n))).length).toBeLessThanOrEqual(3 * 1024);
  });
});

describe('audioPreview', () => {
  test('duration is samples / rate at 3 dp; a bad rate is dropped', () => {
    expect(audioPreview(new Float32Array(16000), 16000)).toMatchObject({ samples: 16000, sampleRate: 16000, duration: 1 });
    expect(audioPreview(new Float32Array(1), 3)).toMatchObject({ duration: 0.333 });
    expect(audioPreview(new Float32Array(4), 0)).toEqual({ kind: 'audio', samples: 4, peaks: new Array<number>(8).fill(0) });
    expect(audioPreview(new Float32Array(4), -16000)).not.toHaveProperty('sampleRate');
    expect(audioPreview(new Float32Array(4), NaN)).not.toHaveProperty('duration');
    expect(audioPreview(new Float32Array(0), 16000)).toEqual({ kind: 'audio', samples: 0, sampleRate: 16000, duration: 0 });
  });

  test('a waveform that throws leaves the count and rate in place', () => {
    const evil = new Proxy(new Float32Array(4), {
      get(target, prop) {
        if (prop === '0') throw new Error('unreadable');
        return Reflect.get(target, prop) as unknown;
      },
    }) as unknown as ArrayLike<number>;
    expect(audioPreview(evil, 8000)).toEqual({ kind: 'audio', samples: 4, sampleRate: 8000, duration: 0.001 });
  });
});

describe('thumbnailOf', () => {
  test.each([1, 2, 3, 4] as const)('%i-channel 200×100 source puts a 96×48 RGBA image and encodes it as JPEG at 0.75', (channels) => {
    const canvas = fakeCanvas();
    const src = fakeRawImage(200, 100, channels);
    expect(thumbnailOf(src, { createCanvas: () => canvas })).toBe(FAKE_THUMB);
    expect(canvas.width).toBe(96);
    expect(canvas.height).toBe(48);
    expect(canvas.puts).toHaveLength(1);
    const { image, x, y } = canvas.puts[0];
    expect([x, y]).toEqual([0, 0]);
    expect(image).toMatchObject({ width: 96, height: 48 });
    expect(image.data).toHaveLength(96 * 48 * 4);
    expect(canvas.encodes).toEqual([{ type: 'image/jpeg', quality: 0.75 }]);
    // nearest neighbour: the first thumb pixel is source (0,0), the last is source (197, 97)
    const tl = Array.from(image.data.slice(0, 4));
    const br = Array.from(image.data.slice(-4));
    const expectPixel = (sx: number, sy: number): number[] => {
      const c0 = Math.round((sx * 255) / 199);
      const c1 = Math.round((sy * 255) / 99);
      switch (channels) {
        case 1:
          return [c0, c0, c0, 255];
        case 2:
          return [c0, c0, c0, c1];
        default:
          return [c0, c1, 128, 255];
      }
    };
    expect(tl).toEqual(expectPixel(0, 0));
    expect(br).toEqual(expectPixel(Math.floor((95 * 200) / 96), Math.floor((47 * 100) / 48)));
  });

  test('small sources are never upscaled; max is configurable; channels are inferred from the data length', () => {
    const canvas = fakeCanvas();
    thumbnailOf(fakeRawImage(20, 10, 3), { createCanvas: () => canvas });
    expect(canvas.puts[0].image).toMatchObject({ width: 20, height: 10 });
    const small = fakeCanvas();
    thumbnailOf(fakeRawImage(400, 400, 4), { createCanvas: () => small, max: 32 });
    expect(small.puts[0].image).toMatchObject({ width: 32, height: 32 });
    expect(THUMB_MAX).toBe(96);
    const tall = fakeCanvas();
    thumbnailOf(fakeRawImage(50, 1000, 1), { createCanvas: () => tall });
    expect(tall.puts[0].image).toMatchObject({ width: 5, height: 96 });
    const inferred = fakeCanvas();
    const { data } = fakeRawImage(8, 8, 3);
    expect(thumbnailOf({ width: 8, height: 8, data }, { createCanvas: () => inferred })).toBe(FAKE_THUMB);
    expect(Array.from(inferred.puts[0].image.data.slice(0, 4))).toEqual([0, 0, 128, 255]);
  });

  test('undefined without a factory, when the factory returns null, or when getContext returns null (happy-dom) or throws', () => {
    const src = fakeRawImage(16, 16, 4);
    expect(thumbnailOf(src)).toBeUndefined();
    expect(thumbnailOf(src, {})).toBeUndefined();
    expect(thumbnailOf(src, { createCanvas: null })).toBeUndefined();
    expect(thumbnailOf(src, { createCanvas: () => null })).toBeUndefined();
    expect(thumbnailOf(src, { createCanvas: () => fakeCanvas({ context: 'null' }) })).toBeUndefined();
    expect(thumbnailOf(src, { createCanvas: () => fakeCanvas({ context: 'throw' }) })).toBeUndefined();
    expect(thumbnailOf(src, { createCanvas: throwingFactory })).toBeUndefined();
    if (typeof document !== 'undefined') expect(thumbnailOf(src, { createCanvas: domCanvasFactory })).toBeUndefined();
  });

  test('undefined when toDataURL exceeds MAX_THUMB_CHARS, is not a data:image/ URL, or throws', () => {
    const src = fakeRawImage(16, 16, 4);
    const tooLong = `data:image/jpeg;base64,${'A'.repeat(MAX_THUMB_CHARS)}`;
    expect(thumbnailOf(src, { createCanvas: () => fakeCanvas({ dataUrl: tooLong }) })).toBeUndefined();
    const justFits = `data:image/jpeg;base64,${'A'.repeat(MAX_THUMB_CHARS - 'data:image/jpeg;base64,'.length)}`;
    expect(justFits).toHaveLength(MAX_THUMB_CHARS);
    expect(thumbnailOf(src, { createCanvas: () => fakeCanvas({ dataUrl: justFits }) })).toBe(justFits);
    expect(thumbnailOf(src, { createCanvas: () => fakeCanvas({ dataUrl: 'data:,' }) })).toBeUndefined();
    expect(thumbnailOf(src, { createCanvas: () => fakeCanvas({ dataUrl: 'https://example.com/x.jpg' }) })).toBeUndefined();
    const throwing = fakeCanvas();
    throwing.toDataURL = () => {
      throw new Error('tainted');
    };
    expect(thumbnailOf(src, { createCanvas: () => throwing })).toBeUndefined();
  });

  test('undefined for malformed images: bad dims, unsupported channel counts, data length mismatch', () => {
    const factory = () => fakeCanvas();
    expect(thumbnailOf({ width: 0, height: 4, channels: 1, data: new Uint8ClampedArray(0) }, { createCanvas: factory })).toBeUndefined();
    expect(thumbnailOf({ width: 2.5, height: 4, channels: 1, data: new Uint8ClampedArray(10) }, { createCanvas: factory })).toBeUndefined();
    expect(thumbnailOf({ width: 4, height: 4, channels: 5, data: new Uint8ClampedArray(80) }, { createCanvas: factory })).toBeUndefined();
    expect(thumbnailOf({ width: 4, height: 4, channels: 3, data: new Uint8ClampedArray(47) }, { createCanvas: factory })).toBeUndefined();
    expect(thumbnailOf({ width: 4, height: 4, data: new Uint8ClampedArray(5) }, { createCanvas: factory })).toBeUndefined();
    expect(thumbnailOf(fakeRawImage(4, 4, 3), { createCanvas: factory, max: 0 })).toBeUndefined();
  });
});

describe('isInputPreview', () => {
  test('accepts every kind with and without the optional media fields', () => {
    expect(isInputPreview({ kind: 'text', text: 'x' })).toBe(true);
    expect(isInputPreview({ kind: 'texts', texts: [] })).toBe(true);
    expect(isInputPreview({ kind: 'image' })).toBe(true);
    expect(isInputPreview({ kind: 'image', width: 1, height: 2, channels: 3, src: 's', thumb: 'data:image/jpeg;base64,' })).toBe(true);
    expect(isInputPreview({ kind: 'audio', samples: 0 })).toBe(true);
    expect(isInputPreview({ kind: 'audio', samples: 1, sampleRate: 2, duration: 0.5, peaks: [0, 0] })).toBe(true);
    expect(isInputPreview({ kind: 'other', json: null })).toBe(true);
  });

  test('rejects wrong-typed media fields and unknown kinds', () => {
    expect(isInputPreview({ kind: 'audio', samples: 1, peaks: ['a'] })).toBe(false);
    expect(isInputPreview({ kind: 'audio', samples: 1, duration: '1s' })).toBe(false);
    expect(isInputPreview({ kind: 'audio', samples: 1, peaks: new Float32Array(2) })).toBe(false);
    expect(isInputPreview({ kind: 'image', thumb: 42 })).toBe(false);
    expect(isInputPreview({ kind: 'image', width: '1' })).toBe(false);
    expect(isInputPreview({ kind: 'text' })).toBe(false);
    expect(isInputPreview({ kind: 'other' })).toBe(false);
    expect(isInputPreview({ kind: 'video' })).toBe(false);
    expect(isInputPreview(null)).toBe(false);
    expect(isInspectorEvent({ type: 'call:start', callId: 'c1', label: 'l', task: null, input: { kind: 'audio', samples: 1, peaks: 'x' }, t: 1 })).toBe(false);
  });
});

describe('fixtureMediaEvents', () => {
  const events = fixtureMediaEvents();

  test('one speech-recognition call and one image-classification call, apart from fixtureEvents()', () => {
    expect(events.filter((e) => e.type === 'call:start').map((e) => e.callId)).toEqual(['c3', 'c4']);
    expect(events.filter((e) => e.type === 'result')).toHaveLength(2);
    expect(fixtureEvents().filter((e) => e.type === 'call:start')).toHaveLength(2);
    const [asr, image] = events.filter((e): e is Extract<InspectorEvent, { type: 'call:start' }> => e.type === 'call:start');
    expect(asr.input).toMatchObject({ kind: 'audio', samples: 48000, sampleRate: 16000, duration: 3 });
    expect(audioOf(asr.input).peaks).toHaveLength(2 * WAVEFORM_BUCKETS);
    for (const v of audioOf(asr.input).peaks!) {
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
    expect(image.input).toMatchObject({ kind: 'image', width: 224, height: 224, channels: 4 });
    expect(imageOf(image.input).thumb).toMatch(/^data:image\/jpeg;base64,/);
    expect(imageOf(image.input).thumb!.length).toBeLessThanOrEqual(MAX_THUMB_CHARS);
    const runs = events.filter((e): e is Extract<InspectorEvent, { type: 'run:start' }> => e.type === 'run:start');
    expect(runs.map((r) => r.inputs[0].dims)).toEqual([
      [1, 80, 3000],
      [1, 3, 224, 224],
    ]);
  });

  test.each(events.map((e, i) => [i, e.type, e] as const))('event %i (%s) passes isInspectorEvent and survives structuredClone and JSON', (_i, _type, e) => {
    expect(isInspectorEvent(e)).toBe(true);
    expect(structuredClone(e)).toEqual(e);
    expect(JSON.parse(JSON.stringify(e))).toEqual(e);
  });

  test('stays within the per-event byte budget', () => {
    for (const e of events) {
      const bytes = JSON.stringify(e).length;
      if (e.type === 'call:start' && e.input.kind === 'audio') expect(bytes).toBeLessThanOrEqual(3 * 1024 + 256);
      if (e.type === 'call:start' && e.input.kind === 'image') expect(bytes).toBeLessThanOrEqual(32 * 1024);
    }
  });
});

describe('toCloneSafe', () => {
  test('passes plain JSON through unchanged', () => {
    const v = { a: 1, b: 'two', c: [true, null, { d: 4 }] };
    expect(toCloneSafe(v)).toEqual(v);
    expect(toCloneSafe(null)).toBeNull();
    expect(toCloneSafe('s')).toBe('s');
    expect(toCloneSafe(1.5)).toBe(1.5);
  });

  test('bigint becomes number, at any depth', () => {
    expect(toCloneSafe(7n)).toBe(7);
    expect(toCloneSafe({ ids: [1n, 2n], n: { m: 3n } })).toEqual({ ids: [1, 2], n: { m: 3 } });
    expect(toCloneSafe(new BigInt64Array([5n, 6n]))).toEqual([5, 6]);
  });

  test('functions, symbols and undefined are dropped', () => {
    const v = { f() {}, s: Symbol('s'), u: undefined, keep: 1, arr: [undefined, () => 1, 2] };
    expect(toCloneSafe(v)).toEqual({ keep: 1, arr: [2] });
    expect(toCloneSafe(undefined)).toBeUndefined();
    expect(toCloneSafe(() => 1)).toBeUndefined();
  });

  test('cycles become "[cycle]" but shared (acyclic) references are fine', () => {
    const self: Record<string, unknown> = { name: 'loop' };
    self.me = self;
    self.list = [self];
    expect(toCloneSafe(self)).toEqual({ name: 'loop', me: '[cycle]', list: ['[cycle]'] });
    const shared = { x: 1 };
    expect(toCloneSafe({ a: shared, b: shared })).toEqual({ a: { x: 1 }, b: { x: 1 } });
  });

  test('depth is capped', () => {
    const deep = { l1: { l2: { l3: { l4: { l5: { l6: { l7: 'bottom' } } } } } } };
    expect(toCloneSafe(deep)).toEqual({ l1: { l2: { l3: { l4: { l5: { l6: '[deep]' } } } } } });
    expect(toCloneSafe(deep, undefined, { maxDepth: 2 })).toEqual({ l1: { l2: '[deep]' } });
  });

  test('tensor-like values become { $tensor } summaries; with a store they get ids', () => {
    const t = fakeTensor({ type: 'float32', dims: [1, 7, 384] });
    const noStore = toCloneSafe({ out: t }) as { out: { $tensor: Record<string, unknown> } };
    expect(noStore.out.$tensor).toMatchObject({ id: '', name: 'out', dtype: 'float32', dims: [1, 7, 384], size: 2688, bytes: 10752 });
    expect(noStore.out.$tensor.head).toHaveLength(8);

    const store = new TensorStore();
    const withStore = toCloneSafe([t, { logits: t }], store) as [{ $tensor: { id: string } }, { logits: { $tensor: { id: string; name: string } } }];
    expect(withStore[0].$tensor.id).toBe('t1');
    expect(withStore[1].logits.$tensor.id).toBe('t1');
    expect(withStore[1].logits.$tensor.name).toBe('logits');
    expect(store.count).toBe(1);
    expect(store.has('t1')).toBe(true);
    roundTrips(withStore);
  });

  test('gpu tensors are summarised without reading their data', () => {
    const gpu = fakeTensor({ type: 'float32', dims: [1, 2, 4, 16], location: 'gpu-buffer' });
    const out = toCloneSafe({ 'present.0.key': gpu }) as { 'present.0.key': { $tensor: { head: unknown } } };
    expect(out['present.0.key'].$tensor.head).toBeNull();
    expect(gpu.dataReads).toBe(0);
  });

  test('typed arrays become plain number arrays, truncated to maxArray with $truncated', () => {
    expect(toCloneSafe(new Float32Array([1, 2, 3]))).toEqual([1, 2, 3]);
    const big = new Float32Array(1000).map((_, i) => i);
    const out = toCloneSafe(big) as { $truncated: number; values: number[] };
    expect(out.$truncated).toBe(1000);
    expect(out.values).toHaveLength(MAX_ARRAY);
    expect(out.values[63]).toBe(63);
    expect(toCloneSafe(new Uint8Array(10), undefined, { maxArray: 4 })).toEqual({ $truncated: 10, values: [0, 0, 0, 0] });
    roundTrips(out);
  });

  test('long plain arrays and strings are truncated too', () => {
    const arr = Array.from({ length: 100 }, (_, i) => i);
    expect(toCloneSafe(arr)).toEqual({ $truncated: 100, values: arr.slice(0, MAX_ARRAY) });
    expect(toCloneSafe('y'.repeat(3000))).toHaveLength(MAX_STRING);
    expect(toCloneSafe({ s: 'abcdef' }, undefined, { maxString: 3 })).toEqual({ s: 'abc' });
  });

  test('Error becomes { message }; Date becomes an ISO string', () => {
    expect(toCloneSafe(new TypeError('boom'))).toEqual({ message: 'boom' });
    expect(toCloneSafe({ err: new Error('x') })).toEqual({ err: { message: 'x' } });
    expect(toCloneSafe(new Date(0))).toBe('1970-01-01T00:00:00.000Z');
  });

  test('a realistic text-generation result is clone-safe', () => {
    const result = [{ generated_text: 'hi the quick brown', scores: new Float32Array([0.1, 0.2]), ids: [1996n, 7742n] }];
    const out = toCloneSafe(result);
    expect(out).toEqual([{ generated_text: 'hi the quick brown', scores: [0.10000000149011612, 0.20000000298023224], ids: [1996, 7742] }]);
    roundTrips(out);
  });
});
