import { describe, expect, test } from 'vitest';
import { InspectorBus } from '../src/bus';
import { DEFAULT_OPTIONS, WrapContext, resolveOptions } from '../src/context';
import type { InspectorEvent } from '../src/events';
import { isInspectorEvent } from '../src/events';
import { TensorStore } from '../src/store';
import type { SessionLike, TensorLike } from '../src/types';
import { WRAPPED, isWrapped, wrapSession, wrapSessions } from '../src/wrap/session';
import type { FakeTensor } from './fakes';
import { fakeEncoderSession, fakeSession, fakeTensor } from './fakes';

function setup() {
  const bus = new InspectorBus();
  const store = new TensorStore();
  const ctx = new WrapContext(bus, store);
  return { bus, store, ctx };
}

const IDS = [101, 1996, 7742, 5927, 2673, 1012, 102];

function encoderFeeds(): Record<string, FakeTensor> {
  const dims = [1, IDS.length];
  return {
    input_ids: fakeTensor({ type: 'int64', dims, data: IDS }),
    attention_mask: fakeTensor({ type: 'int64', dims, data: IDS.map(() => 1) }),
    token_type_ids: fakeTensor({ type: 'int64', dims, data: IDS.map(() => 0) }),
  };
}

const types = (bus: InspectorBus) => bus.history.map((e) => e.type);
const find = <T extends InspectorEvent['type']>(bus: InspectorBus, type: T) =>
  bus.history.filter((e): e is Extract<InspectorEvent, { type: T }> => e.type === type);

describe('InspectorOptions / WrapContext', () => {
  test('defaults: head 8, topK 10, retainBytes 64 MiB, retainLogits false, label', () => {
    expect(DEFAULT_OPTIONS).toEqual({ head: 8, topK: 10, retainBytes: 64 * 1024 * 1024, retainLogits: false, label: 'pipeline' });
    expect(resolveOptions()).toEqual(DEFAULT_OPTIONS);
    expect(resolveOptions({ head: 4, topK: undefined, label: 'x' })).toEqual({ ...DEFAULT_OPTIONS, head: 4, label: 'x' });
    const { bus, store, ctx } = setup();
    expect(ctx.bus).toBe(bus);
    expect(ctx.store).toBe(store);
    expect(ctx.opts).toEqual(DEFAULT_OPTIONS);
    expect(new WrapContext(bus, store, { retainLogits: true }).opts.retainLogits).toBe(true);
  });

  test('ids are namespaced counters and now() is a finite number', () => {
    const { bus, ctx } = setup();
    expect(ctx.currentCallId).toBeNull();
    expect(ctx.tokenizer).toBeNull();
    expect([ctx.nextCallId(), ctx.nextCallId()]).toEqual([`${bus.id}/c1`, `${bus.id}/c2`]);
    expect([ctx.nextRunId(), ctx.nextRunId(), ctx.nextRunId()]).toEqual([`${bus.id}/r1`, `${bus.id}/r2`, `${bus.id}/r3`]);
    expect(Number.isFinite(ctx.now())).toBe(true);
    expect(ctx.tokenToString(1996)).toBeNull(); // no tokenizer yet
  });
});

describe('wrapSession', () => {
  test('WRAPPED is the registered symbol', () => {
    expect(WRAPPED).toBe(Symbol.for('transformersjs-inspector.wrapped'));
  });

  test('inputs are summarised and run:start emitted before the original run is entered', async () => {
    const { bus, store, ctx } = setup();
    let seenAtRun: string[] | null = null;
    let readsAtRun = -1;
    const produced = fakeTensor({ type: 'float32', dims: [1, 7, 384] });
    const session = fakeSession({
      name: 'model',
      inputNames: ['input_ids', 'attention_mask', 'token_type_ids'],
      outputNames: ['last_hidden_state'],
      produce: (feeds) => {
        seenAtRun = types(bus);
        readsAtRun = (feeds.input_ids as FakeTensor).dataReads;
        // proxy mode: the feeds are transferred and unreadable once run() owns them
        for (const t of Object.values(feeds)) (t as FakeTensor).dispose();
        return { last_hidden_state: produced };
      },
    });
    const restore = wrapSession(session, 'model', ctx);
    const feeds = encoderFeeds();
    const out = await session.run(feeds);

    expect(seenAtRun).toEqual(['run:start']);
    expect(readsAtRun).toBe(1);
    expect(out.last_hidden_state).toBe(produced);
    expect(session.runs).toEqual([feeds]);
    expect(types(bus)).toEqual(['run:start', 'run:end']);

    const [start] = find(bus, 'run:start');
    const [end] = find(bus, 'run:end');
    expect(start).toMatchObject({ callId: null, runId: `${bus.id}/r1`, session: 'model' });
    expect(start.inputs.map((s) => s.name)).toEqual(['input_ids', 'attention_mask', 'token_type_ids']);
    expect(start.inputs[0]).toMatchObject({ dtype: 'int64', dims: [1, 7], location: 'cpu', size: 7, bytes: 56, head: IDS });
    expect(end).toMatchObject({ callId: null, runId: `${bus.id}/r1`, session: 'model', error: null });
    expect(end.outputs).toHaveLength(1);
    expect(end.outputs[0]).toMatchObject({ name: 'last_hidden_state', dtype: 'float32', dims: [1, 7, 384], size: 7 * 384 });
    expect(end.outputs[0].head).toHaveLength(8);
    expect(end.ms).toBeGreaterThanOrEqual(0);
    expect(end.t).toBeGreaterThanOrEqual(start.t);
    for (const s of [...start.inputs, ...end.outputs]) expect(store.has(s.id)).toBe(true);
    restore();
  });

  test('events carry ctx.currentCallId and survive structuredClone / JSON', async () => {
    const { bus, ctx } = setup();
    const session = fakeEncoderSession();
    wrapSession(session, 'model', ctx);
    ctx.currentCallId = 'c1';
    await session.run(encoderFeeds());
    ctx.currentCallId = null;
    await session.run(encoderFeeds());
    expect(bus.history.map((e) => ('callId' in e ? e.callId : undefined))).toEqual(['c1', 'c1', null, null]);
    expect(find(bus, 'run:start').map((e) => e.runId)).toEqual([`${bus.id}/r1`, `${bus.id}/r2`]);
    expect(find(bus, 'run:end').map((e) => e.runId)).toEqual([`${bus.id}/r1`, `${bus.id}/r2`]);
    for (const ev of bus.history) {
      expect(isInspectorEvent(ev)).toBe(true);
      expect(structuredClone(ev)).toEqual(ev);
      expect(JSON.parse(JSON.stringify(ev))).toEqual(ev);
    }
  });

  test('gpu-buffer feeds and outputs get head: null and their data getter is never touched', async () => {
    const { bus, ctx } = setup();
    const present = fakeTensor({ type: 'float32', dims: [1, 2, 4, 16], location: 'gpu-buffer' });
    const logits = fakeTensor({ type: 'float32', dims: [1, 1, 32] });
    const session = fakeSession({
      name: 'model',
      inputNames: ['input_ids', 'past_key_values.0.key'],
      outputNames: ['logits', 'present.0.key'],
      produce: () => ({ logits, 'present.0.key': present }),
    });
    wrapSession(session, 'model', ctx);
    const past = fakeTensor({ type: 'float32', dims: [1, 2, 3, 16], location: 'gpu-buffer' });
    await session.run({ input_ids: fakeTensor({ type: 'int64', dims: [1, 1], data: [7] }), 'past_key_values.0.key': past });

    const [start] = find(bus, 'run:start');
    const [end] = find(bus, 'run:end');
    expect(start.inputs[1]).toMatchObject({ name: 'past_key_values.0.key', location: 'gpu-buffer', head: null, bytes: 1 * 2 * 3 * 16 * 4 });
    expect(end.outputs.map((s) => [s.name, s.location, s.head === null])).toEqual([
      ['logits', 'cpu', false],
      ['present.0.key', 'gpu-buffer', true],
    ]);
    expect(past.dataReads).toBe(0);
    expect(present.dataReads).toBe(0);
    expect(logits.dataReads).toBe(1);
  });

  test('a throwing run emits run:end with error and rethrows the same error', async () => {
    const { bus, ctx } = setup();
    const boom = new Error('boom');
    const session = fakeSession({
      name: 'model',
      inputNames: ['input_ids'],
      outputNames: ['logits'],
      produce: () => {
        throw boom;
      },
    });
    wrapSession(session, 'model', ctx);
    await expect(session.run({ input_ids: fakeTensor({ type: 'int64', dims: [1, 2], data: [1, 2] }) })).rejects.toBe(boom);
    expect(types(bus)).toEqual(['run:start', 'run:end']);
    const [end] = find(bus, 'run:end');
    expect(end).toMatchObject({ runId: `${bus.id}/r1`, session: 'model', outputs: [], error: 'Error: boom' });
    expect(end.ms).toBeGreaterThanOrEqual(0);
    expect(structuredClone(end)).toEqual(end);
  });

  test('wrapping twice is a no-op and the second restore does nothing', async () => {
    const { bus, ctx } = setup();
    const session = fakeEncoderSession();
    const original = session.run;
    const restore1 = wrapSession(session, 'model', ctx);
    const wrapped = session.run;
    expect(wrapped).not.toBe(original);
    expect(isWrapped(session)).toBe(true);

    const restore2 = wrapSession(session, 'model', ctx);
    expect(session.run).toBe(wrapped);
    await session.run(encoderFeeds());
    expect(types(bus)).toEqual(['run:start', 'run:end']);

    restore2();
    expect(session.run).toBe(wrapped);
    expect(isWrapped(session)).toBe(true);

    restore1();
    expect(session.run).toBe(original);
    expect(isWrapped(session)).toBe(false);
    expect((session as unknown as Record<symbol, unknown>)[WRAPPED]).toBeUndefined();
    await session.run(encoderFeeds());
    expect(bus.history).toHaveLength(2);
    restore1(); // idempotent
    expect(session.run).toBe(original);

    // can be wrapped again after restore
    const restore3 = wrapSession(session, 'model', ctx);
    await session.run(encoderFeeds());
    expect(bus.history).toHaveLength(4);
    restore3();
    expect(session.run).toBe(original);
  });

  test('restore reinstates a prototype run without leaving an own property behind', async () => {
    const { bus, ctx } = setup();
    class Session implements SessionLike {
      inputNames = ['x'];
      outputNames = ['y'];
      args: unknown[] = [];
      async run(feeds: Record<string, TensorLike>, ...rest: unknown[]): Promise<Record<string, TensorLike>> {
        this.args = rest;
        return { y: feeds.x };
      }
    }
    const session = new Session();
    const restore = wrapSession(session, 'proto', ctx);
    expect(Object.prototype.hasOwnProperty.call(session, 'run')).toBe(true);
    const x = fakeTensor({ type: 'int32', dims: [2], data: [1, 2] });
    const out = await session.run({ x }, ['y'], { logSeverityLevel: 3 });
    expect(out.y).toBe(x); // `this` and the extra run() arguments pass through
    expect(session.args).toEqual([['y'], { logSeverityLevel: 3 }]);
    expect(find(bus, 'run:end')[0].outputs[0]).toMatchObject({ name: 'y', head: [1, 2] });
    restore();
    expect(Object.prototype.hasOwnProperty.call(session, 'run')).toBe(false);
    expect(session.run).toBe(Session.prototype.run);
  });
});

describe('wrapSessions', () => {
  test('wraps every session with its name, shares one runId counter, restores all', async () => {
    const { bus, ctx } = setup();
    const encoder = fakeEncoderSession();
    const decoder = fakeSession({
      name: 'decoder_model_merged',
      inputNames: ['input_ids'],
      outputNames: ['logits'],
      produce: () => ({ logits: fakeTensor({ type: 'float32', dims: [1, 1, 16] }) }),
    });
    const originals = { encoder: encoder.run, decoder: decoder.run };
    const sessions = { encoder_model: encoder, decoder_model_merged: decoder, not_a_session: { inputNames: [], outputNames: [] } } as unknown as Record<string, SessionLike>;
    const restore = wrapSessions(sessions, ctx);
    expect(isWrapped(encoder)).toBe(true);
    expect(isWrapped(decoder)).toBe(true);
    expect(isWrapped(sessions.not_a_session)).toBe(false);

    await encoder.run(encoderFeeds());
    await decoder.run({ input_ids: fakeTensor({ type: 'int64', dims: [1, 1], data: [5] }) });
    expect(find(bus, 'run:start').map((e) => [e.runId, e.session])).toEqual([
      [`${bus.id}/r1`, 'encoder_model'],
      [`${bus.id}/r2`, 'decoder_model_merged'],
    ]);
    expect(find(bus, 'run:end').map((e) => [e.runId, e.session])).toEqual([
      [`${bus.id}/r1`, 'encoder_model'],
      [`${bus.id}/r2`, 'decoder_model_merged'],
    ]);

    restore();
    expect(encoder.run).toBe(originals.encoder);
    expect(decoder.run).toBe(originals.decoder);
    expect(isWrapped(encoder)).toBe(false);
    expect(isWrapped(decoder)).toBe(false);
  });
});
