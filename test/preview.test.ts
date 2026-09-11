import { describe, expect, test } from 'vitest';
import type { InputPreview } from '../src/events';
import { MAX_ARRAY, MAX_STRING, previewInput, toCloneSafe } from '../src/preview';
import { TensorStore } from '../src/store';
import { fakeTensor } from './fakes';

const roundTrips = (v: unknown) => {
  expect(structuredClone(v)).toEqual(v);
  expect(JSON.parse(JSON.stringify(v))).toEqual(v);
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

  test('image: RawImage-shaped objects carry width/height/channels', () => {
    expect(previewInput({ width: 224, height: 224, channels: 3, data: new Uint8ClampedArray(224 * 224 * 3) })).toEqual({
      kind: 'image',
      width: 224,
      height: 224,
      channels: 3,
    });
    expect(previewInput({ width: 8, height: 4 })).toEqual({ kind: 'image', width: 8, height: 4 });
  });

  test('image: http(s) strings, URL and Blob become src', () => {
    expect(previewInput('https://example.com/cat.png')).toEqual({ kind: 'image', src: 'https://example.com/cat.png' });
    expect(previewInput('http://example.com/cat.png')).toEqual({ kind: 'image', src: 'http://example.com/cat.png' });
    expect(previewInput(new URL('https://example.com/dog.jpg'))).toEqual({ kind: 'image', src: 'https://example.com/dog.jpg' });
    const p = previewInput(new Blob([new Uint8Array(10)], { type: 'image/png' }));
    expect(p.kind).toBe('image');
    expect((p as { src: string }).src).toMatch(/^blob:image\/png/);
  });

  test('audio: Float32Array / Float64Array samples and { audio, sampling_rate } objects', () => {
    expect(previewInput(new Float32Array(16000))).toEqual({ kind: 'audio', samples: 16000 });
    expect(previewInput(new Float64Array(8))).toEqual({ kind: 'audio', samples: 8 });
    expect(previewInput({ audio: new Float32Array(32000), sampling_rate: 16000 })).toEqual({ kind: 'audio', samples: 32000, sampleRate: 16000 });
    expect(previewInput({ audio: new Float32Array(5) })).toEqual({ kind: 'audio', samples: 5 });
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

  test('every branch survives structuredClone and JSON', () => {
    const previews: InputPreview[] = [
      previewInput('hi'),
      previewInput(['a', 'b']),
      previewInput({ width: 1, height: 1, channels: 1 }),
      previewInput('https://x.y/z.png'),
      previewInput(new Float32Array(3)),
      previewInput({ audio: new Float64Array(3), sampling_rate: 8000 }),
      previewInput({ a: 1n, b: [new Int32Array([1, 2])] }),
    ];
    for (const p of previews) roundTrips(p);
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
