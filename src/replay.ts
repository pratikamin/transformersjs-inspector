/**
 * Replay: re-run a captured pipeline call with the arguments it was captured with.
 *
 * The registry lives where `attach()` ran (the pipeline's realm), never in the panel: the
 * panel only ever sees clone-safe previews, and in the worker case it is in another realm
 * altogether. It keeps the raw `args` by *reference* (a `Float32Array` of audio, a
 * `RawImage`), bounded FIFO by `replayHistory` (200 by default), and is dropped per pipeline
 * on `detach()`.
 *
 * One registry and one `'replay'` handler per bus (`registryFor`), whatever the number of
 * `attach()` calls sharing it: `bus.handle` keeps one handler per name, so a per-attach
 * handler would silently replace the previous pipeline's. The handler answers **as soon as
 * the new call has started** (its id is known synchronously, because the pipeline wrapper
 * emits `call:start` before its first `await`); the outcome arrives as ordinary events, so a
 * long generation never runs into the request timeout.
 */
import type { InspectorBus } from './bus';
import type { WrapContext } from './context';
import type { ReplayResult } from './events';
import type { PipelineLike } from './types';
import { DEFAULT_MAX_CALLS } from './panel/model';
import { isWrapped } from './wrap/session';

/** Entries kept per bus when `replayHistory` is not given; matches the panel's row cap. */
export const DEFAULT_REPLAY_HISTORY = DEFAULT_MAX_CALLS;

export interface ReplayEntry {
  pipe: PipelineLike;
  /** The original arguments, by reference. */
  args: unknown[];
  /** The context that recorded the call: its wrapper reads `pendingReplayOf` and writes `lastCallId`. */
  ctx: WrapContext;
  /** True while the call itself, or a replay of it, is running. */
  inFlight: boolean;
}

export class ReplayRegistry {
  /** Insertion-ordered; the oldest entry is evicted past `max`. `0` disables recording. */
  readonly max: number;
  private readonly entries = new Map<string, ReplayEntry>();

  constructor(max = DEFAULT_REPLAY_HISTORY) {
    this.max = Math.max(0, Math.floor(max));
  }

  record(callId: string, entry: ReplayEntry): void {
    if (this.max === 0) return;
    this.entries.delete(callId);
    this.entries.set(callId, entry);
    while (this.entries.size > this.max) {
      const oldest = this.entries.keys().next();
      if (oldest.done) break;
      this.entries.delete(oldest.value);
    }
  }

  get(callId: string): ReplayEntry | undefined {
    return this.entries.get(callId);
  }

  has(callId: string): boolean {
    return this.entries.has(callId);
  }

  /** Forgets every call recorded for `pipe` (called by `detach()`). */
  dropByPipe(pipe: PipelineLike): void {
    for (const [id, entry] of this.entries) {
      if (entry.pipe === pipe) this.entries.delete(id);
    }
  }

  get size(): number {
    return this.entries.size;
  }

  clear(): void {
    this.entries.clear();
  }
}

const registries = new WeakMap<InspectorBus, ReplayRegistry>();

/**
 * The registry answering `'replay'` on `bus`, created (with `max`) and registered on first
 * use; later calls return the existing one and ignore `max`. Mirrors `registerStore` in
 * `attach.ts`.
 */
export function registryFor(bus: InspectorBus, max?: number): ReplayRegistry {
  let registry = registries.get(bus);
  if (!registry) {
    registry = new ReplayRegistry(max);
    registries.set(bus, registry);
    const r = registry;
    bus.handle('replay', (req) => handleReplay(r, req));
  }
  return registry;
}

/**
 * Re-runs the recorded call through the pipeline's *wrapped* `_call` (never the original),
 * so the wrapper emits the usual events with a fresh `callId` and `replayOf` set. Resolves
 * once that `call:start` is out; the pipeline promise runs on and its rejection is
 * swallowed here because the wrapper already reported it in `result`.
 */
export function handleReplay(registry: ReplayRegistry, req: { callId: string }): ReplayResult {
  const callId = typeof req === 'object' && req !== null ? req.callId : undefined;
  if (typeof callId !== 'string') return { ok: false, error: 'replay: callId must be a string' };
  const entry = registry.get(callId);
  if (!entry) return { ok: false, error: registry.max === 0 ? 'replay disabled (replayHistory: 0)' : `unknown call ${callId}` };
  if (entry.inFlight) return { ok: false, error: `replay of ${callId} refused: the call is still in flight` };
  const { ctx, pipe, args } = entry;
  // Detached by another handle: the original `_call` would run silently, so refuse instead.
  if (!isWrapped(pipe)) return { ok: false, error: `replay of ${callId} refused: the pipeline is no longer wrapped` };

  entry.inFlight = true;
  ctx.pendingReplayOf = callId;
  ctx.lastCallId = null;
  let p: Promise<unknown>;
  try {
    p = Promise.resolve(pipe._call(...args));
  } catch (e: unknown) {
    ctx.pendingReplayOf = null;
    entry.inFlight = false;
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  const newId = ctx.lastCallId;
  const settle = (): void => {
    entry.inFlight = false;
  };
  p.then(settle, settle);
  if (newId === null) {
    // Cannot happen while `isWrapped(pipe)` holds; kept so a broken wrapper never reports success.
    ctx.pendingReplayOf = null;
    return { ok: false, error: `replay of ${callId} failed: no call:start was emitted` };
  }
  return { ok: true, callId: newId };
}
