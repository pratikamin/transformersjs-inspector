import { describe, expect, test, vi } from 'vitest';
import { attach, defaultLabel } from '../src/attach';
import { InspectorBus, InspectorError } from '../src/bus';
import { GLOBAL_KEY, ensurePanel, getDefaultBus, getDefaultStore } from '../src/default';
import type { InspectorEvent } from '../src/events';
import { isInspectorEvent } from '../src/events';
import { TensorStore } from '../src/store';
import type { TensorLike } from '../src/types';
import { WrapContext } from '../src/context';
import { isWrapped } from '../src/wrap/session';
import { wrapPipelineCall } from '../src/wrap/pipeline';
import { FAKE_PICKS, fakePipeline } from './fakes';

const TEXT = 'the quick brown fox.';
const IDS = [101, 1996, 7742, 5927, 2673, 1012, 102];

const types = (bus: InspectorBus) => bus.history.map((e) => e.type);
const find = <T extends InspectorEvent['type']>(bus: InspectorBus, type: T) =>
  bus.history.filter((e): e is Extract<InspectorEvent, { type: T }> => e.type === type);

function assertCloneSafe(bus: InspectorBus): void {
  for (const e of bus.history) {
    expect(isInspectorEvent(e)).toBe(true);
    expect(structuredClone(e)).toEqual(e);
    expect(JSON.parse(JSON.stringify(e))).toEqual(e);
  }
}

describe('attach: encoder pipeline', () => {
  test('one call emits call:start, tokenize, run:start, run:end, result in order with one callId', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    const handle = attach(pipe, { panel: false, bus });
    expect(handle.bus).toBe(bus);
    expect(handle.store).toBe(getDefaultStore());

    const out = (await pipe(TEXT)) as { dims: number[] };
    expect(out.dims).toEqual([1, 7, 384]);
    expect(types(bus)).toEqual(['call:start', 'tokenize', 'run:start', 'run:end', 'result']);
    const ids = new Set(bus.history.map((e) => e.callId));
    expect(ids.size).toBe(1);
    expect([...ids][0]).toMatch(/^c\d+$/);

    const [start] = find(bus, 'call:start');
    expect(start).toMatchObject({ label: 'feature-extraction · bert', task: 'feature-extraction', input: { kind: 'text', text: TEXT } });
    const [tok] = find(bus, 'tokenize');
    expect(tok.text).toBe(TEXT);
    expect(tok.ids).toEqual([IDS]);
    expect(tok.tokens[0]).toEqual(['[CLS]', 'the', 'quick', 'brown', 'fox', '.', '[SEP]']);
    const [runStart] = find(bus, 'run:start');
    expect(runStart.session).toBe('model');
    expect(runStart.inputs.map((s) => s.name)).toEqual(['input_ids', 'attention_mask', 'token_type_ids']);
    const [runEnd] = find(bus, 'run:end');
    expect(runEnd.runId).toBe(runStart.runId);
    expect(runEnd.outputs[0]).toMatchObject({ name: 'last_hidden_state', dtype: 'float32', dims: [1, 7, 384] });
    const [result] = find(bus, 'result');
    expect(result.error).toBeNull();
    expect(result.ms).toBeGreaterThanOrEqual(0);
    expect(result.ms).toBeLessThanOrEqual(runEnd.t - start.t + 1);
    const tensor = (result.result as { $tensor: { id: string; dims: number[] } }).$tensor;
    expect(tensor).toMatchObject({ dims: [1, 7, 384], dtype: 'float32', location: 'cpu' });
    expect(tensor.id).toBe(runEnd.outputs[0].id); // same tensor object → same store id

    assertCloneSafe(bus);
    handle.detach();
  });

  test('the result tensor is readable through bus.request via the store (attachTo registered once per pair)', async () => {
    const bus = new InspectorBus();
    const store = new TensorStore();
    const attachTo = vi.spyOn(store, 'attachTo');
    const a = attach(fakePipeline(), { panel: false, bus, store });
    const b = attach(fakePipeline(), { panel: false, bus, store });
    expect(attachTo).toHaveBeenCalledTimes(1);
    expect(a.store).toBe(store);
    expect(b.store).toBe(store);
    const pipe = fakePipeline();
    const c = attach(pipe, { panel: false, bus, store });
    await pipe(TEXT);
    const [result] = find(bus, 'result');
    const id = (result.result as { $tensor: { id: string } }).$tensor.id;
    const res = await bus.request('tensor', { id });
    expect('data' in res).toBe(true);
    if ('data' in res) expect((res.data as Float32Array).length).toBe(7 * 384);
    a.detach();
    b.detach();
    c.detach();
    // detaching does not unregister the shared store handler
    expect('data' in (await bus.request('tensor', { id }))).toBe(true);
  });

  test('detach() restores _call, tokenizer._call and every session.run by identity; a second detach is harmless', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    const originals = { call: pipe._call, tok: pipe.tokenizer._call, run: pipe.model.sessions.model.run };
    const handle = attach(pipe, { panel: false, bus });
    expect(pipe._call).not.toBe(originals.call);
    expect(pipe.tokenizer._call).not.toBe(originals.tok);
    expect(pipe.model.sessions.model.run).not.toBe(originals.run);
    expect(isWrapped(pipe)).toBe(true);

    handle.detach();
    expect(pipe._call).toBe(originals.call);
    expect(pipe.tokenizer._call).toBe(originals.tok);
    expect(pipe.model.sessions.model.run).toBe(originals.run);
    expect(isWrapped(pipe)).toBe(false);
    expect(isWrapped(pipe.tokenizer)).toBe(false);
    expect(isWrapped(pipe.model.sessions.model)).toBe(false);

    await pipe(TEXT);
    expect(bus.history).toHaveLength(0);
    handle.detach();
    expect(pipe._call).toBe(originals.call);
  });

  test('attaching twice does not double-wrap; the second handle cannot undo the first', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    const originals = { call: pipe._call, run: pipe.model.sessions.model.run };
    const first = attach(pipe, { panel: false, bus });
    const installed = { call: pipe._call, run: pipe.model.sessions.model.run };
    const second = attach(pipe, { panel: false, bus });
    expect(pipe._call).toBe(installed.call);
    expect(pipe.model.sessions.model.run).toBe(installed.run);

    await pipe(TEXT);
    expect(types(bus)).toEqual(['call:start', 'tokenize', 'run:start', 'run:end', 'result']);

    second.detach();
    expect(pipe._call).toBe(installed.call);
    expect(pipe.model.sessions.model.run).toBe(installed.run);
    first.detach();
    expect(pipe._call).toBe(originals.call);
    expect(pipe.model.sessions.model.run).toBe(originals.run);
  });

  test('a throwing pipeline emits result with error, rethrows and restores currentCallId', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    pipe._call = async () => {
      throw new Error('boom');
    };
    attach(pipe, { panel: false, bus });
    await expect(pipe(TEXT)).rejects.toThrow('boom');
    expect(types(bus)).toEqual(['call:start', 'result']);
    const [result] = find(bus, 'result');
    expect(result).toMatchObject({ error: 'Error: boom', result: null });
    assertCloneSafe(bus);

    // a later call still gets a fresh, correctly scoped callId
    pipe._call = async () => 'ok';
    const ctx = new WrapContext(bus, new TensorStore());
    wrapPipelineCall(pipe, ctx);
    expect(ctx.currentCallId).toBeNull();
    await pipe('x');
    expect(ctx.currentCallId).toBeNull();
  });

  test('label option, task-less pipeline and a pipeline without a tokenizer', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline();
    delete (pipe as { tokenizer?: unknown }).tokenizer;
    delete (pipe as { task?: unknown }).task;
    const tokenizer = fakePipeline().tokenizer; // the fake's _call still needs one to encode
    pipe._call = async (text: unknown) => {
      const enc = tokenizer._call(text) as Record<string, TensorLike>;
      const out = await pipe.model.sessions.model.run({ input_ids: enc.input_ids, attention_mask: enc.attention_mask, token_type_ids: enc.token_type_ids });
      return out.last_hidden_state;
    };
    expect(defaultLabel(pipe)).toBe('pipeline · bert');
    attach(pipe, { panel: false, bus, label: 'custom' });
    await pipe(TEXT);
    expect(types(bus)).toEqual(['call:start', 'run:start', 'run:end', 'result']);
    expect(find(bus, 'call:start')[0]).toMatchObject({ label: 'custom', task: null });
  });

  test('attaching a non-pipeline throws InspectorError', () => {
    const msg = 'attach(): expected a Transformers.js pipeline with model.sessions';
    for (const bad of [undefined, null, 42, 'pipe', {}, { model: {} }, { model: { sessions: null } }, () => {}]) {
      expect(() => attach(bad, { panel: false })).toThrow(InspectorError);
      expect(() => attach(bad, { panel: false })).toThrow(msg);
    }
    const pipe = fakePipeline();
    expect(() => attach(pipe, { panel: false, bus: new InspectorBus() }).detach()).not.toThrow();
  });
});

describe('attach: generative pipeline', () => {
  test('logits and token events are tagged with the callId and interleave with the runs', async () => {
    const bus = new InspectorBus();
    const pipe = fakePipeline({ task: 'text-generation' });
    const originalGenerate = pipe.model.generate;
    const handle = attach(pipe, { panel: false, bus });
    expect(pipe.model.generate).not.toBe(originalGenerate);

    const out = (await pipe('hi', { max_new_tokens: 3 })) as { generated_text: string }[];
    expect(out).toEqual([{ generated_text: 'hi the quick brown' }]);

    expect(types(bus)).toEqual([
      'call:start',
      'tokenize',
      ...['run:start', 'run:end', 'logits', 'token'],
      ...['run:start', 'run:end', 'logits', 'token'],
      ...['run:start', 'run:end', 'logits', 'token'],
      'result',
    ]);
    const callId = find(bus, 'call:start')[0].callId;
    expect(bus.history.every((e) => e.callId === callId)).toBe(true);
    expect(find(bus, 'call:start')[0].label).toBe('text-generation · llama');
    expect(find(bus, 'logits').map((e) => e.step)).toEqual([0, 1, 2]);
    expect(find(bus, 'logits').map((e) => e.topK[0].id)).toEqual(FAKE_PICKS.slice(0, 3));
    expect(find(bus, 'token').map((e) => e.ids)).toEqual(FAKE_PICKS.slice(0, 3).map((id) => [id]));
    expect(find(bus, 'token').map((e) => e.text)).toEqual(['the', 'quick', 'brown']);
    expect(find(bus, 'result')[0].result).toEqual([{ generated_text: 'hi the quick brown' }]);
    const runIds = find(bus, 'run:start').map((e) => e.runId);
    expect(new Set(runIds).size).toBe(3);
    assertCloneSafe(bus);

    handle.detach();
    expect(pipe.model.generate).toBe(originalGenerate);
    expect(isWrapped(pipe.model)).toBe(false);
    handle.detach();
    expect(pipe.model.generate).toBe(originalGenerate);
  });

  test('retainLogits flows through to the generation wrapper and the given store', async () => {
    const bus = new InspectorBus();
    const store = new TensorStore();
    const pipe = fakePipeline({ task: 'text-generation' });
    attach(pipe, { panel: false, bus, store, retainLogits: true, topK: 3 });
    await pipe('hi', { max_new_tokens: 2 });
    const logits = find(bus, 'logits');
    expect(logits).toHaveLength(2);
    for (const e of logits) {
      expect(e.topK).toHaveLength(3);
      expect(e.tensorId).toMatch(/^t\d+$/);
      expect(store.has(e.tensorId!)).toBe(true);
    }
  });
});

describe('defaults', () => {
  test('default bus and store are singletons shared through the global symbol', () => {
    const bus = getDefaultBus();
    const store = getDefaultStore();
    expect(bus).toBeInstanceOf(InspectorBus);
    expect(store).toBeInstanceOf(TensorStore);
    expect(getDefaultBus()).toBe(bus);
    expect(getDefaultStore()).toBe(store);
    expect(GLOBAL_KEY).toBe(Symbol.for('transformersjs-inspector'));
    const g = (globalThis as unknown as Record<symbol, { bus: unknown; store: unknown }>)[Symbol.for('transformersjs-inspector')];
    expect(g.bus).toBe(bus);
    expect(g.store).toBe(store);
  });

  test('attach without a bus uses the defaults and its events land on the default bus', async () => {
    const bus = getDefaultBus();
    const before = bus.history.length;
    const pipe = fakePipeline();
    const handle = attach(pipe, { panel: false });
    expect(handle.bus).toBe(bus);
    expect(handle.store).toBe(getDefaultStore());
    await pipe(TEXT);
    expect(bus.history.length - before).toBe(5);
    handle.detach();
  });

  test('ensurePanel is a no-op without a document', () => {
    expect(typeof document).toBe('undefined');
    expect(ensurePanel(getDefaultBus())).toBeNull();
    expect(ensurePanel(new InspectorBus(), { open: true })).toBeNull();
    // attach with the default panel setting must not throw in node either
    const pipe = fakePipeline();
    expect(() => attach(pipe, { bus: new InspectorBus() }).detach()).not.toThrow();
  });
});
