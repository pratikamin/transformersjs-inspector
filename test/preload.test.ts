import { afterEach, describe, expect, test } from 'vitest';
import { InspectorBus } from '../src/bus';
import { WrapContext } from '../src/context';
import type { InspectorEvent } from '../src/events';
import type { OrtLike } from '../src/preload-core';
import { ORT_SYMBOL, installPreload, isInstalled, nameFor } from '../src/preload-core';
import { TensorStore } from '../src/store';
import type { SessionLike } from '../src/types';
import { isWrapped } from '../src/wrap/session';
import type { FakeSession } from './fakes';
import { fakeEncoderSession, fakeTensor } from './fakes';

type GlobalWithOrt = Record<typeof ORT_SYMBOL, unknown>;
const g = globalThis as unknown as GlobalWithOrt;

/** A module-shaped stand-in for `import * as ort from 'onnxruntime-web/webgpu'`. */
function fakeOrt(): OrtLike & { calls: unknown[][]; sessions: FakeSession[] } {
  const calls: unknown[][] = [];
  const sessions: FakeSession[] = [];
  class Real {
    static readonly tag = 'real-factory';
    static async create(...args: unknown[]): Promise<SessionLike> {
      calls.push(args);
      const s = fakeEncoderSession();
      sessions.push(s);
      return s;
    }
  }
  class Tensor {}
  return { InferenceSession: Real, env: { wasm: { proxy: false }, versions: { web: 'fake' } }, Tensor, registerBackend: () => {}, calls, sessions };
}

function setup() {
  const bus = new InspectorBus();
  const store = new TensorStore();
  const ctx = new WrapContext(bus, store);
  return { bus, store, ctx };
}

const IDS = [101, 1996, 7742, 5927, 2673, 1012, 102];

function encoderFeeds() {
  const dims = [1, IDS.length];
  return {
    input_ids: fakeTensor({ type: 'int64', dims, data: IDS }),
    attention_mask: fakeTensor({ type: 'int64', dims, data: IDS.map(() => 1) }),
    token_type_ids: fakeTensor({ type: 'int64', dims, data: IDS.map(() => 0) }),
  };
}

const find = <T extends InspectorEvent['type']>(bus: InspectorBus, type: T) =>
  bus.history.filter((e): e is Extract<InspectorEvent, { type: T }> => e.type === type);

afterEach(() => {
  delete g[ORT_SYMBOL];
});

describe('installPreload', () => {
  test('ORT_SYMBOL is the key Transformers.js checks', () => {
    expect(ORT_SYMBOL).toBe(Symbol.for('onnxruntime'));
  });

  test('sets globalThis[Symbol.for("onnxruntime")] to the returned shim; env and Tensor keep their identity', () => {
    const ort = fakeOrt();
    expect(isInstalled()).toBe(false);
    expect(ORT_SYMBOL in globalThis).toBe(false);

    const shim = installPreload(ort, setup().ctx);
    expect(isInstalled()).toBe(true);
    expect(g[ORT_SYMBOL]).toBe(shim);
    expect(shim).not.toBe(ort);
    expect(shim.env).toBe(ort.env);
    expect(shim.Tensor).toBe(ort.Tensor);
    expect(shim.registerBackend).toBe(ort.registerBackend); // every other export is carried over
    // The factory is a wrapper inheriting from the real one, so other statics still resolve.
    expect(shim.InferenceSession).not.toBe(ort.InferenceSession);
    expect(Object.getPrototypeOf(shim.InferenceSession)).toBe(ort.InferenceSession);
    expect((shim.InferenceSession as unknown as { tag: string }).tag).toBe('real-factory');
    expect(shim.InferenceSession.create).not.toBe(ort.InferenceSession.create);
  });

  test('create passes its arguments through, returns the real session (wrapped) and its run emits run:start/run:end', async () => {
    const ort = fakeOrt();
    const { bus, ctx } = setup();
    const shim = installPreload(ort, ctx);

    const buffer = new Uint8Array([0x08, 0x07]);
    const options = { executionProviders: [] as string[], logSeverityLevel: 2 };
    const session = await shim.InferenceSession.create(buffer, options);
    expect(ort.calls).toEqual([[buffer, options]]);
    expect(session).toBe(ort.sessions[0]);
    expect(isWrapped(session)).toBe(true);
    expect(bus.history).toHaveLength(0); // creating a session emits nothing

    const out = await session.run(encoderFeeds());
    expect(out.last_hidden_state.dims).toEqual([1, IDS.length, 384]);
    expect(ort.sessions[0].runs).toHaveLength(1);
    expect(bus.history.map((e) => e.type)).toEqual(['run:start', 'run:end']);

    const [start] = find(bus, 'run:start');
    expect(start.callId).toBeNull();
    expect(start.runId).toBe(`${bus.id}/r1`);
    expect(start.session).toBe('session#1 → last_hidden_state');
    expect(start.inputs.map((t) => t.name)).toEqual(['input_ids', 'attention_mask', 'token_type_ids']);
    const [end] = find(bus, 'run:end');
    expect(end.runId).toBe(`${bus.id}/r1`);
    expect(end.session).toBe('session#1 → last_hidden_state');
    expect(end.error).toBeNull();
    expect(end.outputs.map((t) => [t.name, t.dims])).toEqual([['last_hidden_state', [1, IDS.length, 384]]]);
    expect(end.ms).toBeGreaterThanOrEqual(0);
  });

  test('session names: file name of a string or URL argument without .onnx, else session#<n> → <first output>', async () => {
    const ort = fakeOrt();
    const { bus, ctx } = setup();
    const shim = installPreload(ort, ctx);
    const buffer = new Uint8Array(2);
    const enc = fakeEncoderSession();
    expect(nameFor([buffer], enc, 3)).toBe('session#3 → last_hidden_state');
    expect(nameFor(['https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx'], enc, 1)).toBe('model_quantized');
    expect(nameFor(['./models/decoder_model_merged.ONNX'], enc, 1)).toBe('decoder_model_merged');
    expect(nameFor(['C:\\models\\model.onnx'], enc, 1)).toBe('model');
    expect(nameFor(['model'], enc, 1)).toBe('model');
    expect(nameFor([new URL('https://example.test/a/b/encoder_model.onnx?x=1')], enc, 1)).toBe('encoder_model');
    expect(nameFor([''], enc, 4)).toBe('session#4 → last_hidden_state');
    expect(nameFor([buffer], { inputNames: [], outputNames: [], run: enc.run }, 2)).toBe('session#2 → ?');

    // The counter advances per created session, buffers and paths alike.
    for (const arg of [buffer, 'x/model.onnx', buffer]) {
      const session = await shim.InferenceSession.create(arg);
      await session.run(encoderFeeds());
    }
    expect(find(bus, 'run:start').map((e) => e.session)).toEqual(['session#1 → last_hidden_state', 'model', 'session#3 → last_hidden_state']);
  });

  test('a failing run emits run:end with the error and rethrows', async () => {
    const ort = fakeOrt();
    const { bus, ctx } = setup();
    const shim = installPreload(ort, ctx);
    const session = await shim.InferenceSession.create(new Uint8Array(1));
    await expect(session.run({})).rejects.toThrow(/input 'input_ids' is missing/);
    const [end] = find(bus, 'run:end');
    expect(end.error).toMatch(/input_ids/);
    expect(end.outputs).toEqual([]);
  });

  test('double install is a no-op that returns the existing shim, even with a different ort', async () => {
    const ort = fakeOrt();
    const { bus, ctx } = setup();
    const shim = installPreload(ort, ctx);
    const other = fakeOrt();
    expect(installPreload(other, setup().ctx)).toBe(shim);
    expect(installPreload(ort, ctx)).toBe(shim);
    expect(g[ORT_SYMBOL]).toBe(shim);

    // Still wrapped exactly once: one run:start per run, on the first context's bus.
    const session = await shim.InferenceSession.create(new Uint8Array(1));
    await session.run(encoderFeeds());
    expect(find(bus, 'run:start')).toHaveLength(1);
    expect(other.calls).toHaveLength(0);
  });

  test('isInstalled ignores a foreign value under the symbol; installPreload replaces it', () => {
    g[ORT_SYMBOL] = { InferenceSession: {}, env: {}, Tensor: {} };
    expect(isInstalled()).toBe(false);
    const shim = installPreload(fakeOrt(), setup().ctx);
    expect(g[ORT_SYMBOL]).toBe(shim);
    expect(isInstalled()).toBe(true);
  });
});
