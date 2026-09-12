/**
 * Replay (story 9): the per-bus `ReplayRegistry`, its `'replay'` handler, and the round trip
 * through `attach()`: a replayed call goes through the *wrapped* `_call`, so it produces the
 * ordinary event sequence under a fresh `callId` with `replayOf` on its `call:start`.
 */
import { describe, expect, test, vi } from 'vitest';
import { attach } from '../src/attach';
import { InspectorBus } from '../src/bus';
import { WrapContext } from '../src/context';
import type { InspectorEvent } from '../src/events';
import { isInspectorEvent } from '../src/events';
import { DEFAULT_REPLAY_HISTORY, ReplayRegistry, handleReplay, registryFor } from '../src/replay';
import type { ReplayEntry } from '../src/replay';
import { TensorStore } from '../src/store';
import type { PipelineLike } from '../src/types';
import { WRAPPED } from '../src/wrap/session';
import { fakePipeline } from './fakes';
import type { FakePipeline } from './fakes';

const TEXT = 'the quick brown fox.';

const types = (bus: InspectorBus) => bus.history.map((e) => e.type);
const find = <T extends InspectorEvent['type']>(bus: InspectorBus, type: T) =>
  bus.history.filter((e): e is Extract<InspectorEvent, { type: T }> => e.type === type);
const starts = (bus: InspectorBus) => find(bus, 'call:start');
const results = (bus: InspectorBus) => find(bus, 'result');
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

function assertCloneSafe(bus: InspectorBus): void {
  for (const e of bus.history) {
    expect(isInspectorEvent(e)).toBe(true);
    expect(structuredClone(e)).toEqual(e);
    expect(JSON.parse(JSON.stringify(e))).toEqual(e);
  }
}

type Deferred = { promise: Promise<void>; resolve: () => void; reject: (e: unknown) => void };
function deferred(): Deferred {
  let resolve!: () => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * A pipeline whose `_call` waits on the current gate before running the real fake `_call`,
 * recording every argument list it saw. Installed *before* `attach()`, so it is the
 * "original" the wrapper captures.
 */
function gatedPipeline(): { pipe: FakePipeline; gates: Deferred[]; seen: unknown[][] } {
  const pipe = fakePipeline();
  const inner = pipe._call;
  const gates: Deferred[] = [];
  const seen: unknown[][] = [];
  pipe._call = async (...args: unknown[]) => {
    seen.push(args);
    const gate = deferred();
    gates.push(gate);
    await gate.promise;
    return inner(...args);
  };
  return { pipe, gates, seen };
}

/** A minimal entry for exercising `handleReplay` without `attach()`. */
function entryFor(bus: InspectorBus, call: PipelineLike['_call'], wrapped = true): ReplayEntry {
  const pipe = { model: { sessions: {} }, _call: call } as PipelineLike;
  if (wrapped) (pipe as unknown as Record<symbol, unknown>)[WRAPPED] = call;
  return { pipe, args: ['x'], ctx: new WrapContext(bus, new TensorStore()), inFlight: false };
}

describe('ReplayRegistry', () => {
  test('defaults to 200 entries, records in insertion order and evicts the oldest past max', () => {
    expect(DEFAULT_REPLAY_HISTORY).toBe(200);
    expect(new ReplayRegistry().max).toBe(200);
    const bus = new InspectorBus();
    const r = new ReplayRegistry(2);
    const e = (): ReplayEntry => entryFor(bus, async () => null);
    r.record('c1', e());
    r.record('c2', e());
    expect(r.size).toBe(2);
    expect(r.has('c1')).toBe(true);
    r.record('c3', e());
    expect(r.size).toBe(2);
    expect(r.has('c1')).toBe(false);
    expect(r.has('c2')).toBe(true);
    expect(r.has('c3')).toBe(true);
    // Re-recording an id moves it to the newest slot.
    r.record('c2', e());
    r.record('c4', e());
    expect([...['c2', 'c3', 'c4']].map((id) => r.has(id))).toEqual([true, false, true]);
    r.clear();
    expect(r.size).toBe(0);
  });

  test('max is floored at 0 (recording disabled) and dropByPipe forgets one pipeline only', () => {
    const bus = new InspectorBus();
    const off = new ReplayRegistry(0);
    off.record('c1', entryFor(bus, async () => null));
    expect(off.size).toBe(0);
    expect(new ReplayRegistry(-3).max).toBe(0);
    expect(new ReplayRegistry(2.9).max).toBe(2);

    const r = new ReplayRegistry(10);
    const a = entryFor(bus, async () => 'a');
    const b = entryFor(bus, async () => 'b');
    r.record('c1', a);
    r.record('c2', b);
    r.record('c3', { ...a });
    r.dropByPipe(a.pipe);
    expect(r.has('c1')).toBe(false);
    expect(r.has('c3')).toBe(false);
    expect(r.get('c2')).toBe(b);
  });

  test('registryFor: one registry and one handler per bus, sized by the first call', async () => {
    const bus = new InspectorBus();
    const r = registryFor(bus, 5);
    expect(r.max).toBe(5);
    expect(registryFor(bus, 99)).toBe(r);
    expect(registryFor(new InspectorBus())).not.toBe(r);
    await expect(bus.request('replay', { callId: 'c1' })).resolves.toEqual({ ok: false, error: 'unknown call c1' });
  });
});

describe('handleReplay', () => {
  test('refuses a non-string callId, an unknown id, an in-flight entry and an unwrapped pipeline', () => {
    const bus = new InspectorBus();
    const r = new ReplayRegistry();
    expect(handleReplay(r, { callId: 3 as unknown as string })).toEqual({ ok: false, error: 'replay: callId must be a string' });
    expect(handleReplay(r, { callId: 'c1' })).toEqual({ ok: false, error: 'unknown call c1' });
    const busy = entryFor(bus, async () => null);
    busy.inFlight = true;
    r.record('c1', busy);
    expect(handleReplay(r, { callId: 'c1' })).toEqual({ ok: false, error: expect.stringMatching(/c1 .*in flight/) });
    r.record('c2', entryFor(bus, async () => null, false));
    expect(handleReplay(r, { callId: 'c2' })).toEqual({ ok: false, error: expect.stringMatching(/no longer wrapped/) });
    expect(handleReplay(new ReplayRegistry(0), { callId: 'c1' })).toEqual({ ok: false, error: expect.stringMatching(/disabled/) });
  });

  test('a synchronous throw from _call is an ok: false answer and leaves the entry replayable', () => {
    const bus = new InspectorBus();
    const r = new ReplayRegistry();
    const entry = entryFor(bus, (() => {
      throw new Error('sync boom');
    }) as unknown as PipelineLike['_call']);
    r.record('c1', entry);
    expect(handleReplay(r, { callId: 'c1' })).toEqual({ ok: false, error: 'sync boom' });
    expect(entry.inFlight).toBe(false);
    expect(entry.ctx.pendingReplayOf).toBeNull();
  });

  test('a wrapped _call that emits no call:start (broken wrapper) is reported, never a success', async () => {
    const bus = new InspectorBus();
    const r = new ReplayRegistry();
    const entry = entryFor(bus, async () => 'silent');
    r.record('c1', entry);
    expect(handleReplay(r, { callId: 'c1' })).toEqual({ ok: false, error: expect.stringMatching(/no call:start/) });
    await tick();
    expect(entry.inFlight).toBe(false);
  });
});

describe('replay through attach()', () => {
  test('answers { ok, callId: c2 } as soon as the new call starts; the replayed call re-emits the same events with replayOf', async () => {
    const bus = new InspectorBus();
    const { pipe, gates, seen } = gatedPipeline();
    attach(pipe, { panel: false, bus });

    const first = pipe(TEXT, { pooling: 'mean' });
    gates[0].resolve();
    await first;
    expect(types(bus)).toEqual(['call:start', 'tokenize', 'run:start', 'run:end', 'result']);

    const res = await bus.request('replay', { callId: `${bus.id}/c1` });
    expect(res).toEqual({ ok: true, callId: `${bus.id}/c2` });
    // Resolved at call start: c2 has begun but not finished.
    expect(starts(bus).map((e) => e.callId)).toEqual([`${bus.id}/c1`, `${bus.id}/c2`]);
    expect(starts(bus)[1]).toMatchObject({ callId: `${bus.id}/c2`, replayOf: `${bus.id}/c1`, label: 'feature-extraction · bert', task: 'feature-extraction' });
    expect('replayOf' in starts(bus)[0]).toBe(false);
    expect(results(bus)).toHaveLength(1);

    // The original arguments are handed over by identity, options object included.
    expect(seen).toHaveLength(2);
    expect(seen[1]).toEqual(seen[0]);
    expect(seen[1][1]).toBe(seen[0][1]);

    gates[1].resolve();
    await vi.waitFor(() => expect(results(bus)).toHaveLength(2));
    expect(types(bus)).toEqual(['call:start', 'tokenize', 'run:start', 'run:end', 'result', 'call:start', 'tokenize', 'run:start', 'run:end', 'result']);
    const tok = find(bus, 'tokenize');
    expect(tok[1].ids).toEqual(tok[0].ids);
    expect(tok[1].callId).toBe(`${bus.id}/c2`);
    expect(results(bus)[1]).toMatchObject({ callId: `${bus.id}/c2`, error: null });
    assertCloneSafe(bus);
  });

  test('a replay while the call (or its replay) is in flight is refused; it succeeds once settled', async () => {
    const bus = new InspectorBus();
    const { pipe, gates } = gatedPipeline();
    attach(pipe, { panel: false, bus });

    const first = pipe(TEXT);
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: false, error: expect.stringMatching(/c1 .*in flight/) });
    gates[0].resolve();
    await first;

    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c2` });
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: false, error: expect.stringMatching(/c1 .*in flight/) });
    // The replayed call itself is in flight too.
    await expect(bus.request('replay', { callId: `${bus.id}/c2` })).resolves.toEqual({ ok: false, error: expect.stringMatching(/c2 .*in flight/) });
    gates[1].resolve();
    await vi.waitFor(() => expect(results(bus)).toHaveLength(2));
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c3` });
    gates[2].resolve();
    await vi.waitFor(() => expect(results(bus)).toHaveLength(3));
    expect(starts(bus).map((e) => e.replayOf)).toEqual([undefined, `${bus.id}/c1`, `${bus.id}/c1`]);
  });

  test('a replayed call is itself replayable (replayOf chains) and unknown ids are refused', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    attach(pipe, { panel: false, bus });
    await pipe(TEXT);
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c2` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(2));
    await expect(bus.request('replay', { callId: `${bus.id}/c2` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c3` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(3));
    expect(starts(bus).map((e) => [e.callId, e.replayOf ?? null])).toEqual([
      [`${bus.id}/c1`, null],
      [`${bus.id}/c2`, `${bus.id}/c1`],
      [`${bus.id}/c3`, `${bus.id}/c2`],
    ]);
    await expect(bus.request('replay', { callId: `${bus.id}/c9` })).resolves.toEqual({ ok: false, error: `unknown call ${bus.id}/c9` });
  });

  test('a failing replay still answers ok (the call started); its rejection is swallowed and reported as result.error', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    let fail = false;
    const inner = pipe._call;
    pipe._call = async (...args: unknown[]) => {
      if (fail) throw new Error('model exploded');
      return inner(...args);
    };
    attach(pipe, { panel: false, bus });
    await pipe(TEXT);
    fail = true;
    const unhandled = vi.fn();
    process.on('unhandledRejection', unhandled);
    try {
      await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c2` });
      await vi.waitFor(() => expect(results(bus)).toHaveLength(2));
      await tick();
      await tick();
    } finally {
      process.off('unhandledRejection', unhandled);
    }
    expect(results(bus)[1]).toMatchObject({ callId: `${bus.id}/c2`, error: 'Error: model exploded' });
    expect(unhandled).not.toHaveBeenCalled();
    // Settled, so the original can be replayed again.
    fail = false;
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c3` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(3));
    assertCloneSafe(bus);
  });

  test('detach() forgets the pipeline: its ids become unknown, and replayHistory bounds the registry', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    const handle = attach(pipe, { panel: false, bus, replayHistory: 1 });
    await pipe('one');
    await pipe('two');
    expect(registryFor(bus).max).toBe(1);
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: false, error: `unknown call ${bus.id}/c1` });
    await expect(bus.request('replay', { callId: `${bus.id}/c2` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c3` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(3));
    expect(find(bus, 'tokenize')[2].text).toBe('two');

    handle.detach();
    expect(registryFor(bus).size).toBe(0);
    await expect(bus.request('replay', { callId: `${bus.id}/c3` })).resolves.toEqual({ ok: false, error: `unknown call ${bus.id}/c3` });
    // The handler stays on the bus (it is per bus, not per attach); a re-attach records again.
    attach(pipe, { panel: false, bus });
    await pipe('three');
    await expect(bus.request('replay', { callId: `${bus.id}/c4` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c5` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(5));
  });

  test('replayHistory: 0 disables recording for the bus and the handler says so', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    attach(pipe, { panel: false, bus, replayHistory: 0 });
    await pipe(TEXT);
    expect(registryFor(bus).size).toBe(0);
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: false, error: expect.stringMatching(/disabled/) });
  });

  test('two pipelines on one bus share the handler and each replays through its own wrapper', async () => {
    const bus = new InspectorBus();
    const a = fakePipeline();
    const b = fakePipeline({ task: 'text-generation' });
    attach(a, { panel: false, bus, label: 'A' });
    attach(b, { panel: false, bus, label: 'B' });
    await a('alpha');
    await b('beta', { max_new_tokens: 2 });
    await expect(bus.request('replay', { callId: `${bus.id}/c2` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c3` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(3));
    await expect(bus.request('replay', { callId: `${bus.id}/c1` })).resolves.toEqual({ ok: true, callId: `${bus.id}/c4` });
    await vi.waitFor(() => expect(results(bus)).toHaveLength(4));
    expect(starts(bus).map((e) => [e.callId, e.label, e.replayOf ?? null])).toEqual([
      [`${bus.id}/c1`, 'A', null],
      [`${bus.id}/c2`, 'B', null],
      [`${bus.id}/c3`, 'B', `${bus.id}/c2`],
      [`${bus.id}/c4`, 'A', `${bus.id}/c1`],
    ]);
    const gen = find(bus, 'token');
    expect(gen.filter((e) => e.callId === `${bus.id}/c3`).map((e) => e.ids)).toEqual(gen.filter((e) => e.callId === `${bus.id}/c2`).map((e) => e.ids));
    assertCloneSafe(bus);
  });
});
