/**
 * The worker bridge over a real `MessageChannel` (Node's global): bus A on `port1` plays the
 * page, bus B on `port2` the worker. Delivery through a MessagePort is a macrotask, so every
 * assertion on the peer waits with `vi.waitFor` and then a settle tick to prove "exactly once".
 * Ports are closed in `afterEach` or the open handles keep Vitest alive.
 */
import { afterEach, describe, expect, test, vi } from 'vitest';
import { attach } from '../src/attach';
import { InspectorBus } from '../src/bus';
import type { InspectorEvent, TensorData } from '../src/events';
import { TensorStore } from '../src/store';
import { connectWorker, exposeToPage, messagePortTransport } from '../src/worker';
import type { PortLike } from '../src/worker';
import { fakePipeline, fakeTensor } from './fakes';

const ev = (n: number): InspectorEvent => ({ type: 'token', callId: 'c1', step: n, ids: [n], text: `t${n}`, t: n });
const settle = () => new Promise<void>((r) => setTimeout(r, 20));
const dataOf = (res: TensorData) => ('data' in res ? res.data : undefined);

let channel: MessageChannel | null = null;
const cleanups: (() => void)[] = [];

function openChannel(): { page: InspectorBus; worker: InspectorBus; port1: MessagePort; port2: MessagePort } {
  channel = new MessageChannel();
  const { port1, port2 } = channel;
  const page = new InspectorBus();
  const worker = new InspectorBus();
  cleanups.push(page.connect(messagePortTransport(port1)));
  cleanups.push(worker.connect(messagePortTransport(port2)));
  return { page, worker, port1, port2 };
}

afterEach(() => {
  for (const off of cleanups.splice(0)) off();
  channel?.port1.close();
  channel?.port2.close();
  channel = null;
});

describe('messagePortTransport over MessageChannel', () => {
  test('an event emitted on the worker bus arrives on the page bus exactly once and is not echoed back', async () => {
    const { page, worker } = openChannel();
    const seenPage = vi.fn();
    const seenWorker = vi.fn();
    page.on(seenPage);
    worker.on(seenWorker);

    worker.emit(ev(1));
    await vi.waitFor(() => expect(seenPage).toHaveBeenCalledTimes(1));
    await settle();
    expect(seenPage).toHaveBeenCalledTimes(1);
    expect(seenPage).toHaveBeenCalledWith(ev(1));
    expect(seenWorker).toHaveBeenCalledTimes(1); // the local delivery only; no echo from the page
    expect(page.history).toEqual([ev(1)]);
    expect(worker.history).toEqual([ev(1)]);
  });

  test("page.request('tensor') is answered by the worker's TensorStore with a Float32Array", async () => {
    const { page, worker } = openChannel();
    const store = new TensorStore();
    store.attachTo(worker);
    const values = [0.5, -1.25, 3, 4.5];
    const { id } = store.put(fakeTensor({ type: 'float32', dims: [1, 4], data: Float32Array.from(values) }), 'last_hidden_state');

    const res = await page.request('tensor', { id });
    expect(res).toMatchObject({ id, dtype: 'float32', dims: [1, 4] });
    const data = dataOf(res);
    expect(data).toBeInstanceOf(Float32Array); // structured clone keeps the typed array
    expect(Array.from(data as Float32Array)).toEqual(values);

    // An unknown id is an ordinary (ok) response carrying the store's error, not a rejection.
    await expect(page.request('tensor', { id: 'nope' })).resolves.toEqual({ id: 'nope', error: 'unknown' });
  });

  test("page.request('replay') is answered by the registry attach() put on the worker bus; the new call's events reach the page", async () => {
    const { page, worker } = openChannel();
    const pipe = fakePipeline();
    const handle = attach(pipe, { bus: worker, panel: false });
    await pipe('over the bridge');
    const resultsOn = (bus: InspectorBus) => bus.history.filter((e) => e.type === 'result');
    await vi.waitFor(() => expect(resultsOn(page)).toHaveLength(1));

    // The page bus has no 'replay' handler of its own: the request crosses the port and is
    // answered as soon as the replayed call has started on the worker side.
    await expect(page.request('replay', { callId: 'c1' })).resolves.toEqual({ ok: true, callId: 'c2' });
    await vi.waitFor(() => expect(resultsOn(page)).toHaveLength(2));
    const starts = page.history.filter((e) => e.type === 'call:start');
    expect(starts).toHaveLength(2);
    expect(starts[1]).toMatchObject({ callId: 'c2', replayOf: 'c1', label: 'feature-extraction · bert' });
    expect(page.history.map((e) => e.type)).toEqual(worker.history.map((e) => e.type));
    for (const e of page.history) expect(structuredClone(e)).toEqual(e);

    await expect(page.request('replay', { callId: 'nope' })).resolves.toEqual({ ok: false, error: 'unknown call nope' });
    handle.detach();
    await expect(page.request('replay', { callId: 'c2' })).resolves.toEqual({ ok: false, error: 'unknown call c2' });
  });

  test('a handler that throws on the worker side rejects the page request with its message', async () => {
    const { page, worker } = openChannel();
    worker.handle('tensor', () => {
      throw new Error('boom');
    });
    await expect(page.request('tensor', { id: 'x' })).rejects.toThrow('boom');
  });

  test('untagged messages on the port are ignored by the transport; tagged ones stay visible to host listeners', async () => {
    const { page, worker, port2 } = openChannel();
    const seenPage = vi.fn();
    page.on(seenPage);
    const raw = vi.fn();
    port2.addEventListener('message', (e) => raw(e.data));

    // Host protocol messages travel on the same port without disturbing either bus.
    port2.postMessage({ text: 'hello' });
    port2.postMessage({ __tjsi: 2, kind: 'event', event: ev(9) }); // wrong tag value
    port2.postMessage('plain string');
    port2.postMessage(null);
    await settle();
    expect(seenPage).not.toHaveBeenCalled();
    expect(page.history).toEqual([]);

    // And the wire messages a bus posts are ordinary messages a host listener can see and skip.
    page.emit(ev(1));
    await vi.waitFor(() => expect(raw).toHaveBeenCalledTimes(1));
    expect(raw.mock.calls[0][0]).toMatchObject({ __tjsi: 1, kind: 'event', event: ev(1) });
    expect(worker.history).toEqual([ev(1)]);
  });

  test('close() removes the listener but leaves the port usable; start?.() is called when present', () => {
    const listeners: unknown[] = [];
    const posted: unknown[] = [];
    const start = vi.fn();
    const port: PortLike = {
      postMessage: (m) => posted.push(m),
      addEventListener: (_type, cb) => listeners.push(cb),
      removeEventListener: (_type, cb) => listeners.splice(listeners.indexOf(cb), 1),
      start,
    };
    const t = messagePortTransport(port);
    expect(start).toHaveBeenCalledTimes(1);
    expect(listeners).toHaveLength(1);
    const cb = vi.fn();
    t.subscribe(cb);
    (listeners[0] as (e: { data: unknown }) => void)({ data: { __tjsi: 1, kind: 'event', event: ev(1) } });
    (listeners[0] as (e: { data: unknown }) => void)({ data: { text: 'x' } });
    expect(cb).toHaveBeenCalledTimes(1);
    t.post({ __tjsi: 1, kind: 'event', event: ev(2) });
    expect(posted).toEqual([{ __tjsi: 1, kind: 'event', event: ev(2) }]);
    t.close?.();
    expect(listeners).toHaveLength(0);

    // A port without start() (a Worker, a worker's self) is fine too.
    const bare: PortLike = { postMessage: () => {}, addEventListener: () => {}, removeEventListener: () => {} };
    expect(() => messagePortTransport(bare)).not.toThrow();
  });
});

describe('connectWorker / exposeToPage', () => {
  test('wire a page bus to a worker bus over the two ends of a channel', async () => {
    channel = new MessageChannel();
    const workerBus = new InspectorBus();
    cleanups.push(exposeToPage(workerBus, channel.port2));
    const pageBus = connectWorker(channel.port1);
    expect(pageBus).toBeInstanceOf(InspectorBus);

    const seen = vi.fn();
    pageBus.on(seen);
    workerBus.emit(ev(3));
    await vi.waitFor(() => expect(seen).toHaveBeenCalledWith(ev(3)));

    const store = new TensorStore();
    store.attachTo(workerBus);
    const { id } = store.put(fakeTensor({ type: 'float32', dims: [2], data: Float32Array.from([1, 2]) }), 'x');
    expect(dataOf(await pageBus.request('tensor', { id }))).toBeInstanceOf(Float32Array);
  });

  test('connectWorker reuses a caller-supplied bus; the disconnect from exposeToPage stops the relay', async () => {
    channel = new MessageChannel();
    const workerBus = new InspectorBus();
    const off = exposeToPage(workerBus, channel.port2);
    const mine = new InspectorBus();
    expect(connectWorker(channel.port1, mine)).toBe(mine);

    workerBus.emit(ev(1));
    await vi.waitFor(() => expect(mine.history).toEqual([ev(1)]));
    off();
    workerBus.emit(ev(2));
    await settle();
    expect(mine.history).toEqual([ev(1)]);
    await expect(mine.request('tensor', { id: 't1' }, { timeoutMs: 50 })).rejects.toThrow(/timed out/);
  });
});
