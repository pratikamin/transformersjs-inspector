import { describe, expect, test, vi } from 'vitest';
import { InspectorBus } from '../src/bus';
import { WrapContext } from '../src/context';
import type { InspectorEvent } from '../src/events';
import { isInspectorEvent } from '../src/events';
import { TensorStore } from '../src/store';
import type { LogitsProcessorLike, StreamerLike, TensorLike } from '../src/types';
import { mergeProcessors, mergeStreamer, topKFromLogits, wrapGenerate } from '../src/wrap/generation';
import { isWrapped } from '../src/wrap/session';
import type { InspectorOptions } from '../src/context';
import { FAKE_PICKS, FAKE_VOCAB_SIZE, fakeGenerativeModel, fakeTensor, fakeTokenizer } from './fakes';

type LogitsEvent = Extract<InspectorEvent, { type: 'logits' }>;
type TokenEvent = Extract<InspectorEvent, { type: 'token' }>;

const logitsEvents = (bus: InspectorBus) => bus.history.filter((e): e is LogitsEvent => e.type === 'logits');
const tokenEvents = (bus: InspectorBus) => bus.history.filter((e): e is TokenEvent => e.type === 'token');

const PROMPT = [101, 7632, 102]; // [CLS] hi [SEP]
const PICK_TOKENS = ['the', 'quick', 'brown'];

function setup(opts: Partial<InspectorOptions> = {}) {
  const bus = new InspectorBus();
  const store = new TensorStore();
  const ctx = new WrapContext(bus, store, opts);
  ctx.tokenizer = fakeTokenizer();
  const model = fakeGenerativeModel();
  const generate = (extra: Record<string, unknown> = {}) =>
    model.generate({ input_ids: fakeTensor({ type: 'int64', dims: [1, PROMPT.length], data: PROMPT }), max_new_tokens: 3, ...extra });
  return { bus, store, ctx, model, generate };
}

/** Reference implementation: full softmax + full sort. */
function bruteTopK(data: ArrayLike<number>, k: number) {
  const arr = Array.from(data);
  const max = Math.max(...arr);
  const denom = arr.reduce((s, v) => s + Math.exp(v - max), 0);
  return arr
    .map((logit, id) => ({ id, logit, prob: Math.exp(logit - max) / denom }))
    .sort((a, b) => b.logit - a.logit || a.id - b.id)
    .slice(0, k);
}

describe('topKFromLogits', () => {
  test('hand-built row: correct ids, sorted desc, probs in (0,1] summing to <= 1', () => {
    const { entries, vocab } = topKFromLogits([1, 5, 3, 5, -2, 0], 3);
    expect(vocab).toBe(6);
    expect(entries.map((e) => e.id)).toEqual([1, 3, 2]); // tie on 5 keeps the lower id first
    expect(entries.map((e) => e.logit)).toEqual([5, 5, 3]);
    const denom = [1, 5, 3, 5, -2, 0].reduce((s, v) => s + Math.exp(v - 5), 0);
    expect(entries[0].prob).toBeCloseTo(1 / denom, 12);
    expect(entries[2].prob).toBeCloseTo(Math.exp(-2) / denom, 12);
    for (const e of entries) {
      expect(e.prob).toBeGreaterThan(0);
      expect(e.prob).toBeLessThanOrEqual(1);
    }
    for (let i = 1; i < entries.length; i++) expect(entries[i - 1].prob).toBeGreaterThanOrEqual(entries[i].prob);
    expect(entries.reduce((s, e) => s + e.prob, 0)).toBeLessThanOrEqual(1);
  });

  test('k >= vocab returns the whole vocab (probs sum to 1); k <= 0 and empty data return nothing', () => {
    const all = topKFromLogits(Float32Array.from([0.5, -1, 2]), 10);
    expect(all.vocab).toBe(3);
    expect(all.entries.map((e) => e.id)).toEqual([2, 0, 1]);
    expect(all.entries.reduce((s, e) => s + e.prob, 0)).toBeCloseTo(1, 12);
    expect(topKFromLogits([1, 2, 3], 0)).toEqual({ entries: [], vocab: 3 });
    expect(topKFromLogits([1, 2, 3], -4)).toEqual({ entries: [], vocab: 3 });
    expect(topKFromLogits([], 5)).toEqual({ entries: [], vocab: 0 });
    expect(topKFromLogits([7, 7, 7], 2.9).entries.map((e) => e.id)).toEqual([0, 1]);
  });

  test('partial selection matches a full sort on random rows, including ties', () => {
    let seed = 12345;
    const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 10 - 5;
    for (const k of [1, 7, 10, 64]) {
      const data = Float32Array.from({ length: 1000 }, () => Math.round(rand() * 4) / 4); // quarter steps: many ties
      const got = topKFromLogits(data, k).entries;
      const want = bruteTopK(data, k);
      expect(got.map((e) => e.id)).toEqual(want.map((e) => e.id));
      got.forEach((e, i) => expect(e.prob).toBeCloseTo(want[i].prob, 12));
    }
  });

  test('a 128k-vocab float32 row (the real shape) is handled with the peak on top', () => {
    const data = new Float32Array(FAKE_VOCAB_SIZE);
    for (let i = 0; i < data.length; i++) data[i] = -((i * 13) % 11) - 1;
    data[1996] = 8;
    const { entries, vocab } = topKFromLogits(data, 10);
    expect(vocab).toBe(FAKE_VOCAB_SIZE);
    expect(entries).toHaveLength(10);
    expect(entries[0]).toMatchObject({ id: 1996, logit: 8 });
    // ~11.7k entries sit at -1, so the peak's share is e^8 / (e^8 + sum of the rest), about 0.3
    let denom = 0;
    for (let i = 0; i < data.length; i++) denom += Math.exp(data[i] - 8);
    expect(entries[0].prob).toBeCloseTo(1 / denom, 10);
    expect(entries[1].prob).toBeCloseTo(Math.exp(-9) / denom, 10);
    expect(entries.slice(1).every((e) => e.logit === -1)).toBe(true);
    expect(entries.reduce((s, e) => s + e.prob, 0)).toBeLessThanOrEqual(1);
  });
});

describe('wrapGenerate', () => {
  test('3 steps: one logits and one token event per step with matching step numbers; prompt put skipped', async () => {
    const { bus, ctx, model, generate } = setup();
    ctx.currentCallId = 'c1';
    wrapGenerate(model, ctx);
    expect(isWrapped(model)).toBe(true);
    const out = await generate();
    expect(Array.from(out.data as ArrayLike<bigint>, Number)).toEqual([...PROMPT, ...FAKE_PICKS.slice(0, 3)]);
    expect(model.generations).toBe(1);

    const logits = logitsEvents(bus);
    const tokens = tokenEvents(bus);
    expect(logits).toHaveLength(3);
    expect(tokens).toHaveLength(3);
    expect(logits.map((e) => e.step)).toEqual([0, 1, 2]);
    expect(tokens.map((e) => e.step)).toEqual([0, 1, 2]);
    logits.forEach((e, step) => {
      expect(e.callId).toBe('c1');
      expect(e.vocab).toBe(FAKE_VOCAB_SIZE);
      expect(e.topK).toHaveLength(10);
      expect(e.topK[0]).toMatchObject({ id: FAKE_PICKS[step], token: PICK_TOKENS[step], logit: 8 });
      expect(e.topK[0].prob).toBeGreaterThan(0.25); // the fake's peak at 8 against ~11.7k entries at -1
      expect(e.topK[1].prob).toBeLessThan(0.001);
      expect(e.tensorId).toBeNull();
      for (let i = 1; i < e.topK.length; i++) expect(e.topK[i - 1].prob).toBeGreaterThanOrEqual(e.topK[i].prob);
    });
    tokens.forEach((e, step) => {
      expect(e.callId).toBe('c1');
      expect(e.ids).toEqual([FAKE_PICKS[step]]);
      expect(e.text).toBe(PICK_TOKENS[step]);
      expect(e.ids.every((id) => typeof id === 'number')).toBe(true);
    });
    // token n is emitted after logits n and before logits n+1
    const seq = bus.history.filter((e) => e.type === 'logits' || e.type === 'token').map((e) => e.type);
    expect(seq).toEqual(['logits', 'token', 'logits', 'token', 'logits', 'token']);
    for (const e of bus.history) {
      expect(isInspectorEvent(e)).toBe(true);
      expect(structuredClone(e)).toEqual(e);
      expect(JSON.parse(JSON.stringify(e))).toEqual(e);
    }
  });

  test('step counters reset on every generate call', async () => {
    const { bus, ctx, model, generate } = setup();
    wrapGenerate(model, ctx);
    await generate({ max_new_tokens: 2 });
    await generate({ max_new_tokens: 2 });
    expect(logitsEvents(bus).map((e) => e.step)).toEqual([0, 1, 0, 1]);
    expect(tokenEvents(bus).map((e) => e.step)).toEqual([0, 1, 0, 1]);
  });

  test("host's plain-array processor list and streamer are still called; ours runs first", async () => {
    const { bus, ctx, model, generate } = setup();
    wrapGenerate(model, ctx);
    const hostProc = vi.fn<LogitsProcessorLike>((_ids, logits) => logits);
    const tokensSeenAtPut: number[] = [];
    const hostStreamer: StreamerLike & { ended: number } = {
      ended: 0,
      put: vi.fn(() => {
        tokensSeenAtPut.push(tokenEvents(bus).length);
      }),
      end: vi.fn(() => {
        hostStreamer.ended++;
      }),
    };
    await generate({ logits_processor: [hostProc], streamer: hostStreamer });
    expect(hostProc).toHaveBeenCalledTimes(3);
    const [ids, logits] = hostProc.mock.calls[0];
    expect(Array.isArray(ids)).toBe(true);
    expect((logits as TensorLike).dims).toEqual([1, FAKE_VOCAB_SIZE]);
    expect(hostStreamer.put).toHaveBeenCalledTimes(4); // prompt + 3 steps
    expect(hostStreamer.ended).toBe(1);
    expect(tokensSeenAtPut).toEqual([0, 1, 2, 3]); // our token event precedes each host put
    expect(logitsEvents(bus)).toHaveLength(3);
    expect(tokenEvents(bus)).toHaveLength(3);
  });

  test("host's iterable LogitsProcessorList-like object and .processors holder are both spread", async () => {
    const { bus, ctx, model, generate } = setup();
    wrapGenerate(model, ctx);
    const a = vi.fn<LogitsProcessorLike>((_ids, logits) => logits);
    const b = vi.fn<LogitsProcessorLike>((_ids, logits) => logits);
    const iterable = {
      processors: [a, b],
      push: vi.fn(),
      [Symbol.iterator]() {
        return this.processors.values();
      },
    };
    await generate({ logits_processor: iterable });
    expect(a).toHaveBeenCalledTimes(3);
    expect(b).toHaveBeenCalledTimes(3);
    expect(iterable.push).not.toHaveBeenCalled(); // spread, not mutated
    expect(iterable.processors).toHaveLength(2);

    const c = vi.fn<LogitsProcessorLike>((_ids, logits) => logits);
    await generate({ logits_processor: { processors: [c] } });
    expect(c).toHaveBeenCalledTimes(3);
    expect(logitsEvents(bus)).toHaveLength(6);
  });

  test('mergeProcessors / mergeStreamer edge cases', () => {
    const { ctx } = setup();
    const ours: LogitsProcessorLike = (_ids, logits) => logits;
    expect(mergeProcessors(undefined, ours, ctx)).toEqual([ours]);
    expect(mergeProcessors(null, ours, ctx)).toEqual([ours]);
    expect(mergeProcessors(42, ours, ctx)).toEqual([ours]);
    expect(mergeProcessors({ processors: 'nope' }, ours, ctx)).toEqual([ours]);
    const host = [vi.fn()];
    const merged = mergeProcessors(host, ours, ctx) as unknown[];
    expect(merged).toEqual([host[0], ours]);
    expect(host).toHaveLength(1);

    const streamer: StreamerLike = { put: vi.fn(), end: vi.fn() };
    expect(mergeStreamer(undefined, streamer)).toBe(streamer);
    expect(mergeStreamer({ notAStreamer: true }, streamer)).toBe(streamer);
    const putOnly = { put: vi.fn() };
    const merged2 = mergeStreamer(putOnly, streamer);
    merged2.put([[1n]]);
    merged2.end();
    expect(streamer.put).toHaveBeenCalledWith([[1n]]);
    expect(putOnly.put).toHaveBeenCalledWith([[1n]]);
    expect(streamer.end).toHaveBeenCalledTimes(1);
  });

  test('retainLogits: true stores each step and the id reads back; default false leaves tensorId null', async () => {
    const { bus, store, ctx, model, generate } = setup({ retainLogits: true });
    wrapGenerate(model, ctx);
    await generate();
    const logits = logitsEvents(bus);
    expect(logits).toHaveLength(3);
    const ids = logits.map((e) => e.tensorId);
    expect(new Set(ids).size).toBe(3);
    for (const [step, id] of ids.entries()) {
      expect(id).toMatch(/^t\d+$/);
      expect(store.has(id!)).toBe(true);
      const res = await store.read(id!);
      expect('data' in res).toBe(true);
      if ('data' in res) {
        expect(res.dims).toEqual([1, FAKE_VOCAB_SIZE]);
        expect(res.dtype).toBe('float32');
        expect((res.data as Float32Array)[FAKE_PICKS[step]]).toBe(8);
      }
    }
    expect(store.count).toBe(3);

    const plain = setup();
    wrapGenerate(plain.model, plain.ctx);
    await plain.generate();
    expect(logitsEvents(plain.bus).map((e) => e.tensorId)).toEqual([null, null, null]);
    expect(plain.store.count).toBe(0);
  });

  test('detach restores the original generate identity and stops emitting; double wrap is idempotent', async () => {
    const { bus, ctx, model, generate } = setup();
    const original = model.generate;
    expect(Object.prototype.hasOwnProperty.call(model, 'generate')).toBe(true);
    const restore = wrapGenerate(model, ctx);
    const installed = model.generate;
    expect(installed).not.toBe(original);

    const again = wrapGenerate(model, ctx);
    expect(model.generate).toBe(installed);
    await generate();
    expect(logitsEvents(bus)).toHaveLength(3); // not 6
    again(); // the no-op restore from the second wrap must not undo the first
    expect(model.generate).toBe(installed);
    expect(isWrapped(model)).toBe(true);

    restore();
    expect(model.generate).toBe(original);
    expect(isWrapped(model)).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(model, 'generate')).toBe(true);
    await generate();
    expect(logitsEvents(bus)).toHaveLength(3);
    expect(model.generations).toBe(2);
    restore(); // twice is harmless
    expect(model.generate).toBe(original);
  });

  test('prototype generate: restore deletes the shadowing own property', () => {
    const { ctx } = setup();
    class Model {
      generate(): Promise<unknown> {
        return Promise.resolve(null);
      }
    }
    const m = new Model();
    const restore = wrapGenerate(m, ctx);
    expect(Object.prototype.hasOwnProperty.call(m, 'generate')).toBe(true);
    restore();
    expect(Object.prototype.hasOwnProperty.call(m, 'generate')).toBe(false);
    expect(m.generate).toBe(Model.prototype.generate);
  });

  test('escape hatch: opts.transformers.LogitsProcessorList builds a real list with push', async () => {
    class FakeList {
      static built = 0;
      processors: unknown[] = [];
      constructor() {
        FakeList.built++;
      }
      push(p: unknown): void {
        this.processors.push(p);
      }
      [Symbol.iterator]() {
        return this.processors.values();
      }
    }
    const { bus, ctx, model, generate } = setup({ transformers: { LogitsProcessorList: FakeList } });
    expect(ctx.opts.transformers?.LogitsProcessorList).toBe(FakeList);
    const real = model.generate;
    let seen: Record<string, unknown> | undefined;
    model.generate = (o) => {
      seen = o;
      return real.call(model, o);
    };
    wrapGenerate(model, ctx);
    const host = vi.fn<LogitsProcessorLike>((_ids, logits) => logits);
    await generate({ logits_processor: [host] });
    expect(FakeList.built).toBe(1);
    expect(seen?.logits_processor).toBeInstanceOf(FakeList);
    const list = seen?.logits_processor as FakeList;
    expect(list.processors).toHaveLength(2);
    expect(list.processors[0]).toBe(host);
    expect(typeof list.processors[1]).toBe('function');
    expect(host).toHaveBeenCalledTimes(3);
    expect(logitsEvents(bus)).toHaveLength(3);
    expect(seen?.max_new_tokens).toBe(3); // other options pass through untouched

    // our processor never reads a device-resident tensor and still returns it unchanged
    const ours = list.processors[1] as LogitsProcessorLike;
    const gpu = fakeTensor({ type: 'float32', dims: [1, 7], location: 'gpu-buffer' });
    expect(ours([[1n]], gpu)).toBe(gpu);
    expect(gpu.dataReads).toBe(0);
    const last = logitsEvents(bus).at(-1)!;
    expect(last).toMatchObject({ step: 3, vocab: 7, topK: [], tensorId: null });
  });
});
