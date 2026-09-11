import { describe, expect, test, vi } from 'vitest';
import { InspectorBus, InspectorError, loopbackPair } from '../src/bus';
import type { InspectorEvent, TensorData } from '../src/events';
import { isInspectorEvent } from '../src/events';
import type { TensorLike } from '../src/types';
import { fakePipeline, fakeTensor, fixtureEvents } from './fakes';

const ev = (n: number): InspectorEvent => ({ type: 'token', callId: 'c1', step: n, ids: [n], text: `t${n}`, t: n });
const flush = () => new Promise<void>((r) => setTimeout(r, 0));
const tensorData = (id: string): TensorData => ({ id, dtype: 'float32', dims: [2], data: new Float32Array([1, 2]) });

describe('InspectorBus local', () => {
  test('emit fans out to listeners; unsubscribe stops delivery', () => {
    const bus = new InspectorBus();
    const seen: InspectorEvent[] = [];
    const off = bus.on((e) => seen.push(e));
    bus.emit(ev(1));
    off();
    bus.emit(ev(2));
    expect(seen).toEqual([ev(1)]);
    expect(bus.history).toEqual([ev(1), ev(2)]);
  });

  test('history is capped at maxHistory and clear() empties it', () => {
    const bus = new InspectorBus({ maxHistory: 3 });
    for (let i = 1; i <= 5; i++) bus.emit(ev(i));
    expect(bus.history.map((e) => e.t)).toEqual([3, 4, 5]);
    bus.clear();
    expect(bus.history).toEqual([]);
    expect(new InspectorBus().maxHistory).toBe(500);
  });

  test('handle/request round trip; unregistered handler rejects', async () => {
    const bus = new InspectorBus();
    const off = bus.handle('tensor', async ({ id }) => tensorData(id));
    const res = await bus.request('tensor', { id: 't1' });
    expect(res).toMatchObject({ id: 't1', dims: [2] });
    off();
    await expect(bus.request('tensor', { id: 't1' })).rejects.toBeInstanceOf(InspectorError);
  });

  test('a throwing local handler rejects the request', async () => {
    const bus = new InspectorBus();
    bus.handle('tensor', () => {
      throw new Error('boom');
    });
    await expect(bus.request('tensor', { id: 'x' })).rejects.toThrow('boom');
  });
});

describe('InspectorBus over loopbackPair', () => {
  function pair(opts?: { maxHistory?: number }) {
    const [ta, tb] = loopbackPair();
    const a = new InspectorBus(opts);
    const b = new InspectorBus(opts);
    const offA = a.connect(ta);
    const offB = b.connect(tb);
    return { a, b, offA, offB };
  }

  test('relays each event exactly once with no echo', async () => {
    const { a, b } = pair();
    const seenA = vi.fn();
    const seenB = vi.fn();
    a.on(seenA);
    b.on(seenB);
    a.emit(ev(1));
    b.emit(ev(2));
    await flush();
    expect(seenA).toHaveBeenCalledTimes(2);
    expect(seenB).toHaveBeenCalledTimes(2);
    // local emits land synchronously, relayed ones a microtask later, so compare as sets
    expect(a.history.map((e) => e.t).sort()).toEqual([1, 2]);
    expect(b.history.map((e) => e.t).sort()).toEqual([1, 2]);
  });

  test('disconnect stops relaying', async () => {
    const { a, b, offA } = pair();
    offA();
    a.emit(ev(1));
    await flush();
    expect(b.history).toEqual([]);
  });

  test('a middle bus relays to its other transports but not back to the source', async () => {
    const [t1, t2] = loopbackPair();
    const [t3, t4] = loopbackPair();
    const a = new InspectorBus();
    const mid = new InspectorBus();
    const c = new InspectorBus();
    a.connect(t1);
    mid.connect(t2);
    mid.connect(t3);
    c.connect(t4);
    const seenA = vi.fn();
    a.on(seenA);
    a.emit(ev(1));
    await flush();
    await flush();
    expect(seenA).toHaveBeenCalledTimes(1);
    expect(mid.history).toEqual([ev(1)]);
    expect(c.history).toEqual([ev(1)]);
  });

  test('requests are forwarded to the peer that has a handler', async () => {
    const { a, b } = pair();
    b.handle('tensor', ({ id }) => tensorData(id));
    const res = await a.request('tensor', { id: 't9' });
    expect(res).toMatchObject({ id: 't9', dtype: 'float32' });
    expect('data' in res && res.data).toBeInstanceOf(Float32Array);
  });

  test('a local handler wins over the transport', async () => {
    const { a, b } = pair();
    b.handle('tensor', ({ id }) => ({ id, error: 'remote' }));
    a.handle('tensor', ({ id }) => ({ id, error: 'local' }));
    expect(await a.request('tensor', { id: 'x' })).toEqual({ id: 'x', error: 'local' });
  });

  test('remote handler errors and missing handlers reject with InspectorError', async () => {
    const { a, b } = pair();
    await expect(a.request('tensor', { id: 'x' })).rejects.toThrow(/no handler/);
    b.handle('tensor', () => {
      throw new Error('remote boom');
    });
    const err = await a.request('tensor', { id: 'x' }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(InspectorError);
    expect((err as Error).message).toBe('remote boom');
  });

  test('request times out when the peer never answers', async () => {
    const { a, b } = pair();
    b.handle('tensor', () => new Promise<TensorData>(() => {}));
    const err = await a.request('tensor', { id: 'x' }, { timeoutMs: 15 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(InspectorError);
    expect((err as Error).message).toMatch(/timed out/);
  });
});

describe('fixture events', () => {
  const events = fixtureEvents();

  test('cover one embedding call and one 3-step generation call', () => {
    const types = events.map((e) => e.type);
    expect(types.filter((t) => t === 'call:start')).toHaveLength(2);
    expect(types.filter((t) => t === 'result')).toHaveLength(2);
    expect(types.filter((t) => t === 'logits')).toHaveLength(3);
    expect(types.filter((t) => t === 'token')).toHaveLength(3);
    expect(types.filter((t) => t === 'run:start')).toHaveLength(4);
    expect(new Set(types)).toEqual(new Set(['call:start', 'tokenize', 'run:start', 'run:end', 'logits', 'token', 'result']));
  });

  test.each(events.map((e, i) => [i, e.type, e] as const))('event %i (%s) survives structuredClone and JSON round trip', (_i, _type, e) => {
    expect(structuredClone(e)).toEqual(e);
    expect(JSON.parse(JSON.stringify(e))).toEqual(e);
    expect(isInspectorEvent(e)).toBe(true);
  });

  test('isInspectorEvent rejects junk', () => {
    expect(isInspectorEvent(null)).toBe(false);
    expect(isInspectorEvent({ type: 'token' })).toBe(false);
    expect(isInspectorEvent({ type: 'nope', t: 1 })).toBe(false);
    expect(isInspectorEvent({ type: 'result', callId: 'c1', ms: 1, error: null, t: 1 })).toBe(false);
  });
});

describe('fakes', () => {
  test('gpu-buffer tensor: data getter throws, getData resolves', async () => {
    const t = fakeTensor({ type: 'float32', dims: [1, 2, 3, 16], location: 'gpu-buffer' });
    expect(() => t.data).toThrow(/getData/);
    expect(t.dataReads).toBe(1);
    expect((await t.getData()).length).toBe(96);
    const cpu = fakeTensor({ type: 'int64', dims: [1, 3], data: [101, 7632, 102] });
    expect(cpu.data).toBeInstanceOf(BigInt64Array);
    expect(Array.from(cpu.data as BigInt64Array, Number)).toEqual([101, 7632, 102]);
  });

  test('embedding pipeline runs tokenizer -> session -> tensor', async () => {
    const pipe = fakePipeline();
    const out = (await pipe('the quick brown fox.')) as TensorLike;
    expect(out.dims).toEqual([1, 7, 384]);
    const run = pipe.model.sessions.model.runs[0];
    expect(Array.from(run.input_ids.data as BigInt64Array, Number)).toEqual([101, 1996, 7742, 5927, 2673, 1012, 102]);
    expect(pipe.tokenizer._tokenizer.id_to_token(7742)).toBe('quick');
  });

  test('generation pipeline calls the logits processor and streamer once per step', async () => {
    const pipe = fakePipeline({ task: 'text-generation' });
    const steps: number[] = [];
    const puts: unknown[] = [];
    const processor = vi.fn((ids: unknown, logits: TensorLike) => {
      steps.push((ids as bigint[][])[0].length);
      expect(logits.dims).toEqual([1, 128256]);
      return logits;
    });
    const streamer = { put: (v: unknown) => puts.push(v), end: vi.fn() };
    const res = (await pipe('hi', { max_new_tokens: 3, logits_processor: [processor], streamer })) as { generated_text: string }[];
    expect(res[0].generated_text).toBe('hi the quick brown');
    expect(processor).toHaveBeenCalledTimes(3);
    expect(steps).toEqual([3, 4, 5]);
    expect(puts).toHaveLength(4);
    expect(streamer.end).toHaveBeenCalledTimes(1);
    const runs = pipe.model.sessions.model.runs;
    expect(runs).toHaveLength(3);
    expect(runs[1].input_ids.dims).toEqual([1, 1]);
    expect(runs[1]['past_key_values.0.key'].dims).toEqual([1, 2, 3, 16]);
    expect(runs[1]['past_key_values.0.key'].location).toBe('gpu-buffer');
  });
});
