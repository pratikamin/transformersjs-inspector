/**
 * Web Worker bridge: the same `InspectorBus` on both sides of a `postMessage` boundary.
 * Inside the worker, `attach(pipe, { bus, panel: false })` emits onto a local bus that
 * `exposeToPage` relays to the page; on the page, `connectWorker(worker)` yields a bus that
 * sees those events and forwards `request('tensor')` back to the worker's `TensorStore`, so
 * `mountPanel(bus)` works unchanged. Every wire message carries `__tjsi: 1`; anything else
 * on the port (the host's own protocol) is ignored here and must be ignored by the host in
 * turn (`isWireMessage`).
 */
import type { Transport, WireMessage } from './bus';
import { InspectorBus, isWireMessage } from './bus';

/** The subset of `Worker`, `MessagePort` and a worker's `self` the transport needs. */
export interface PortLike {
  postMessage(msg: unknown): void;
  addEventListener(type: 'message', cb: (e: { data: unknown }) => void): void;
  removeEventListener(type: 'message', cb: (e: { data: unknown }) => void): void;
  /** `MessagePort` only delivers once started when listeners are attached via `addEventListener`. */
  start?(): void;
}

/**
 * A `Transport` over any `PortLike`. Only messages whose `data` is a `WireMessage` are
 * delivered to subscribers; `close()` removes the listener but leaves the port open (it is
 * the host's, not ours).
 */
export function messagePortTransport(port: PortLike): Transport {
  const subs = new Set<(msg: WireMessage) => void>();
  const onMessage = (e: { data: unknown }): void => {
    if (!isWireMessage(e.data)) return;
    for (const cb of subs) cb(e.data);
  };
  port.addEventListener('message', onMessage);
  port.start?.();
  return {
    post(msg) {
      port.postMessage(msg);
    },
    subscribe(cb) {
      subs.add(cb);
      return () => {
        subs.delete(cb);
      };
    },
    close() {
      subs.clear();
      port.removeEventListener('message', onMessage);
    },
  };
}

/** Page side: connects `bus` (a fresh one by default) to the worker and returns it. */
export function connectWorker(worker: PortLike, bus: InspectorBus = new InspectorBus()): InspectorBus {
  bus.connect(messagePortTransport(worker));
  return bus;
}

/** Worker side: relays `bus` over `port` (the worker's own `self` by default); returns the disconnect function. */
export function exposeToPage(bus: InspectorBus, port: PortLike = self as unknown as PortLike): () => void {
  return bus.connect(messagePortTransport(port));
}
