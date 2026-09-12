import { afterEach, describe, expect, test, vi } from 'vitest';
import { attach } from '../src/attach';
import { InspectorBus } from '../src/bus';
import { getDefaultStore } from '../src/default';
import { isInspectorEvent } from '../src/events';
import { createState, reduce } from '../src/panel/model';
import { TensorStore } from '../src/store';
import { messagePortTransport } from '../src/worker';
import { fakePipeline, fakeTensor } from './fakes';

const channels: MessageChannel[] = [];
const cleanups: (() => void)[] = [];
function connect(page: InspectorBus, worker: InspectorBus): void {
  const channel = new MessageChannel();
  channels.push(channel);
  cleanups.push(page.connect(messagePortTransport(channel.port1)));
  cleanups.push(worker.connect(messagePortTransport(channel.port2)));
}
afterEach(() => {
  cleanups.splice(0).forEach((off) => off());
  for (const channel of channels.splice(0)) {
    channel.port1.close();
    channel.port2.close();
  }
});

const scalar = (value: number) => fakeTensor({ type: 'float32', dims: [1], data: Float32Array.of(value) });

describe('ownership across page and worker buses', () => {
  test('separates calls/runs, reads the owning tensor and replays the owning pipeline with two workers', async () => {
    const page = new InspectorBus();
    const worker1 = new InspectorBus();
    const worker2 = new InspectorBus();
    connect(page, worker1);
    connect(page, worker2);
    const buses = [page, worker1, worker2];
    const stores = buses.map(() => new TensorStore());
    const pipes = buses.map(() => fakePipeline());
    for (let i = 0; i < buses.length; i++) {
      cleanups.push(attach(pipes[i], { bus: buses[i], store: stores[i], label: `model ${i}`, panel: false }).detach);
      await pipes[i](`input ${i}`);
    }
    await vi.waitFor(() => expect(page.history.filter((e) => e.type === 'result')).toHaveLength(3));
    const starts = page.history.filter((e) => e.type === 'call:start');
    expect(new Set(starts.map((e) => e.callId)).size).toBe(3);
    expect(new Set(page.history.filter((e) => e.type === 'run:start').map((e) => e.runId)).size).toBe(3);
    const state = createState();
    page.history.forEach((e) => reduce(state, e));
    expect(state.calls).toHaveLength(3);
    expect(state.calls.every((call) => call.runs.length === 1 && call.runs[0].done)).toBe(true);
    for (const e of page.history) {
      expect(isInspectorEvent(e)).toBe(true);
      expect(JSON.parse(JSON.stringify(e))).toEqual(e);
    }

    // Make the owner slower than the non-owning worker's rejection.
    worker1.handle('tensor', async ({ id }) => {
      await new Promise((r) => setTimeout(r, 30));
      return stores[1].read(id);
    }, { scope: stores[1].id });
    const ids = stores.map((store, i) => store.put(scalar(11 + i), 'value').id);
    expect(new Set(ids).size).toBe(3);
    for (let i = 0; i < ids.length; i++) {
      const result = await page.request('tensor', { id: ids[i] });
      expect(result).toMatchObject({ id: ids[i], data: Float32Array.of(11 + i) });
    }
    stores[1].clear();
    await expect(page.request('tensor', { id: ids[1] })).resolves.toEqual({ id: ids[1], error: 'evicted' });
    await expect(page.request('tensor', { id: `${stores[2].id}/missing` })).resolves.toEqual({ id: `${stores[2].id}/missing`, error: 'unknown' });

    for (const start of starts) {
      const replay = await page.request('replay', { callId: start.callId });
      expect(replay.ok).toBe(true);
      if (!replay.ok) throw new Error(replay.error);
      await vi.waitFor(() => expect(page.history.some((e) => e.type === 'result' && e.callId === replay.callId)).toBe(true));
      expect(page.history.find((e) => e.type === 'call:start' && e.callId === replay.callId))
        .toMatchObject({ replayOf: start.callId, label: start.label, input: start.input });
    }
    expect(new Set(page.history.filter((e) => e.type === 'call:start').map((e) => e.callId)).size).toBe(6);
  });

  test('routes an owned request through an intermediate bus with its own store', async () => {
    const page = new InspectorBus(), middle = new InspectorBus(), worker = new InspectorBus();
    connect(page, middle);
    connect(middle, worker);
    new TensorStore().attachTo(page);
    new TensorStore().attachTo(middle);
    const store = new TensorStore();
    store.attachTo(worker);
    const { id } = store.put(scalar(42), 'answer');
    await expect(page.request('tensor', { id })).resolves.toMatchObject({ id, data: Float32Array.of(42) });
    await expect(page.request('tensor', { id: 'absent/t1' })).rejects.toThrow(/no handler/);
  });
});

describe('attach tensor budget', () => {
  test('zero retains no tensors, a small budget evicts, and independent stores remain readable on one bus', async () => {
    const bus = new InspectorBus();
    const shared = getDefaultStore();
    const originalBudget = shared.maxBytes;
    const zero = attach(fakePipeline(), { bus, panel: false, retainBytes: 0 });
    const small = attach(fakePipeline(), { bus, panel: false, retainBytes: 4 });
    const normal = attach(fakePipeline(), { bus, panel: false });
    cleanups.push(zero.detach, small.detach, normal.detach);
    expect(zero.store.maxBytes).toBe(0);
    expect(small.store.maxBytes).toBe(4);
    expect(normal.store).toBe(shared);
    expect(shared.maxBytes).toBe(originalBudget);
    const empty = zero.store.put(scalar(1), 'zero');
    const old = small.store.put(scalar(2), 'old');
    const recent = small.store.put(scalar(3), 'recent');
    const defaultTensor = normal.store.put(scalar(4), 'default');
    expect(zero.store.count).toBe(0);
    expect(small.store.bytes).toBe(4);
    for (const id of [empty.id, old.id]) {
      await expect(bus.request('tensor', { id })).resolves.toEqual({ id, error: 'evicted' });
    }
    await expect(bus.request('tensor', { id: recent.id })).resolves.toMatchObject({ data: Float32Array.of(3) });
    await expect(bus.request('tensor', { id: defaultTensor.id })).resolves.toMatchObject({ data: Float32Array.of(4) });
  });

  test('an explicit store takes precedence over retainBytes', () => {
    const store = new TensorStore({ maxBytes: 8 });
    const handle = attach(fakePipeline(), { bus: new InspectorBus(), panel: false, store, retainBytes: 0 });
    cleanups.push(handle.detach);
    expect(handle.store).toBe(store);
    expect(store.maxBytes).toBe(8);
  });
});
