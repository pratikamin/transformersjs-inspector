import type { InspectorEvent, RequestMap } from './events';
import { isInspectorEvent } from './events';

/** One message on a `Transport`. `__tjsi` marks it so postMessage listeners can ignore the rest. */
export type WireMessage = { __tjsi: 1 } & (
  | { kind: 'event'; event: InspectorEvent }
  | { kind: 'request'; reqId: string; name: string; payload: unknown }
  | { kind: 'response'; reqId: string; ok: boolean; payload?: unknown; error?: string }
);

export interface Transport {
  post(msg: WireMessage): void;
  subscribe(cb: (msg: WireMessage) => void): () => void;
  close?(): void;
}

export class InspectorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InspectorError';
  }
}

type Listener = (ev: InspectorEvent) => void;
type Handler = (req: unknown) => unknown;
type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout>; waiting: Set<Transport> };

export const DEFAULT_MAX_HISTORY = 500;
export const DEFAULT_REQUEST_TIMEOUT_MS = 10_000;

export function isWireMessage(x: unknown): x is WireMessage {
  return typeof x === 'object' && x !== null && (x as { __tjsi?: unknown }).__tjsi === 1;
}

/**
 * Transport-agnostic event bus. Local `emit` fans out to listeners and every
 * connected transport; events arriving on a transport are delivered locally and
 * relayed to the *other* transports only. Requests are answered by a local
 * handler for the requested owner when one exists, otherwise forwarded to the
 * connected transports. Transports must form a tree (cycles are unsupported).
 */
export class InspectorBus {
  /** Namespace for call/run ids, unique across pages and workers. Treat ids as opaque. */
  readonly id = `b${crypto.randomUUID()}`;
  readonly maxHistory: number;
  private readonly _history: InspectorEvent[] = [];
  private readonly listeners = new Set<Listener>();
  private readonly handlers = new Map<string, Handler>();
  private readonly transports = new Set<Transport>();
  private readonly pending = new Map<string, Pending>();
  private seq = 0;

  constructor(opts: { maxHistory?: number } = {}) {
    this.maxHistory = Math.max(0, opts.maxHistory ?? DEFAULT_MAX_HISTORY);
  }

  get history(): readonly InspectorEvent[] {
    return this._history;
  }

  emit(ev: InspectorEvent): void {
    this.deliver(ev, null);
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  handle<K extends keyof RequestMap>(
    name: K,
    h: (req: RequestMap[K]['req']) => RequestMap[K]['res'] | Promise<RequestMap[K]['res']>,
    opts: { scope?: string } = {},
  ): () => void {
    const key = opts.scope === undefined ? name : `${name}/${opts.scope}`;
    this.handlers.set(key, h as Handler);
    return () => {
      if (this.handlers.get(key) === h) this.handlers.delete(key);
    };
  }

  request<K extends keyof RequestMap>(
    name: K,
    req: RequestMap[K]['req'],
    opts: { timeoutMs?: number } = {},
  ): Promise<RequestMap[K]['res']> {
    return this.dispatch(name, req, null, opts.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS) as Promise<RequestMap[K]['res']>;
  }

  connect(t: Transport): () => void {
    this.transports.add(t);
    const unsubscribe = t.subscribe((msg) => this.receive(msg, t));
    return () => {
      unsubscribe();
      this.transports.delete(t);
      t.close?.();
    };
  }

  clear(): void {
    this._history.length = 0;
  }

  // ---- internals -----------------------------------------------------------

  private deliver(ev: InspectorEvent, from: Transport | null): void {
    if (this.maxHistory > 0) {
      this._history.push(ev);
      if (this._history.length > this.maxHistory) this._history.splice(0, this._history.length - this.maxHistory);
    }
    for (const l of this.listeners) l(ev);
    this.broadcast({ __tjsi: 1, kind: 'event', event: ev }, from);
  }

  private broadcast(msg: WireMessage, except: Transport | null): number {
    let n = 0;
    for (const t of this.transports) {
      if (t === except) continue;
      t.post(msg);
      n++;
    }
    return n;
  }

  private dispatch(name: string, payload: unknown, from: Transport | null, timeoutMs: number): Promise<unknown> {
    // A qualified id must reach its owner; an unrelated local store/registry
    // must never answer it with its own tensor or an "unknown" verdict.
    const record = payload as { id?: unknown; callId?: unknown } | null;
    const id = name === 'tensor' ? record?.id : name === 'replay' ? record?.callId : undefined;
    const slash = typeof id === 'string' ? id.indexOf('/') : -1;
    const scope = slash >= 0 ? (id as string).slice(0, slash) : null;
    let h = this.handlers.get(scope === null ? name : `${name}/${scope}`);
    // Keep bare-id requests useful for callers with a single local handler.
    if (!h && scope === null) {
      const matches = [...this.handlers].filter(([key]) => key.startsWith(`${name}/`));
      if (matches.length === 1) h = matches[0][1];
    }
    if (h) {
      return new Promise((resolve) => resolve(h(payload)));
    }
    return new Promise((resolve, reject) => {
      const reqId = `${this.id}-${++this.seq}`;
      const timer = setTimeout(() => {
        this.pending.delete(reqId);
        reject(new InspectorError(`request '${name}' timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      const waiting = new Set([...this.transports].filter((t) => t !== from));
      this.pending.set(reqId, { resolve, reject, timer, waiting });
      for (const t of waiting) t.post({ __tjsi: 1, kind: 'request', reqId, name, payload });
      if (waiting.size === 0) {
        clearTimeout(timer);
        this.pending.delete(reqId);
        reject(new InspectorError(`no handler for request '${name}'`));
      }
    });
  }

  private receive(msg: WireMessage, from: Transport): void {
    if (!isWireMessage(msg)) return;
    switch (msg.kind) {
      case 'event':
        if (isInspectorEvent(msg.event)) this.deliver(msg.event, from);
        return;
      case 'request': {
        const { reqId } = msg;
        this.dispatch(msg.name, msg.payload, from, DEFAULT_REQUEST_TIMEOUT_MS).then(
          (payload) => from.post({ __tjsi: 1, kind: 'response', reqId, ok: true, payload }),
          (e: unknown) => from.post({ __tjsi: 1, kind: 'response', reqId, ok: false, error: errorMessage(e) }),
        );
        return;
      }
      case 'response': {
        const p = this.pending.get(msg.reqId);
        if (!p || !p.waiting.delete(from)) return;
        // Several workers can be connected. A non-owner can reject before the
        // owner answers; only fail after every branch has rejected.
        if (!msg.ok && p.waiting.size > 0) return;
        this.pending.delete(msg.reqId);
        clearTimeout(p.timer);
        if (msg.ok) p.resolve(msg.payload);
        else p.reject(new InspectorError(msg.error ?? 'request failed'));
        return;
      }
    }
  }
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Two in-memory transports wired to each other; delivery is asynchronous (`queueMicrotask`). */
export function loopbackPair(): [Transport, Transport] {
  type Cb = (msg: WireMessage) => void;
  type End = Transport & { subs: Set<Cb>; peer: End | null; open: boolean };
  const make = (): End => {
    const self: End = {
      subs: new Set<Cb>(),
      peer: null,
      open: true,
      post(msg) {
        const peer = self.peer;
        if (!self.open || !peer) return;
        queueMicrotask(() => {
          for (const cb of peer.subs) cb(msg);
        });
      },
      subscribe(cb) {
        self.subs.add(cb);
        return () => {
          self.subs.delete(cb);
        };
      },
      close() {
        self.open = false;
        self.subs.clear();
      },
    };
    return self;
  };
  const a = make();
  const b = make();
  a.peer = b;
  b.peer = a;
  return [a, b];
}
