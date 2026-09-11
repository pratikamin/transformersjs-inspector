/**
 * Session boundary: replaces `session.run` on the instance the host holds (verified in
 * the feasibility prototype, docs/01-research.md) with a wrapper that summarises every feed *before* calling through
 * (in proxy mode the feeds are transferred to a worker and unreadable afterwards),
 * emits `run:start`, awaits the original, summarises the outputs and emits `run:end`.
 */
import type { WrapContext } from '../context';
import type { TensorSummary } from '../events';
import type { SessionLike, TensorLike } from '../types';
import { isTensorLike } from '../summarize';

/** Marker set on every wrapped instance; its value is the installed wrapper function. */
export const WRAPPED = Symbol.for('transformersjs-inspector.wrapped');

type Marked = Record<symbol, unknown>;

export function isWrapped(target: object): boolean {
  return (target as Marked)[WRAPPED] !== undefined;
}

export const noop = (): void => {};

/**
 * Installs `wrapper` as `target[key]`, marks the target and returns a restore function.
 * Restoring reinstates the exact original: the own property is put back when there was
 * one, otherwise it is deleted so the prototype's method shows through again. Restoring
 * twice, or after somebody else replaced the method, is a no-op.
 */
export function replaceMethod<T extends object, K extends keyof T>(target: T, key: K, wrapper: T[K]): () => void {
  const hadOwn = Object.prototype.hasOwnProperty.call(target, key);
  const original = target[key];
  target[key] = wrapper;
  (target as Marked)[WRAPPED] = wrapper;
  return () => {
    if ((target as Marked)[WRAPPED] !== wrapper) return;
    delete (target as Marked)[WRAPPED];
    if (target[key] !== wrapper) return;
    if (hadOwn) target[key] = original;
    else delete target[key];
  };
}

/** Summaries (with store ids) for every tensor-valued entry of a feeds / outputs object. */
function summarizeAll(values: unknown, ctx: WrapContext): TensorSummary[] {
  if (typeof values !== 'object' || values === null) return [];
  const out: TensorSummary[] = [];
  for (const [name, v] of Object.entries(values as Record<string, unknown>)) {
    if (isTensorLike(v)) out.push(ctx.store.put(v, name));
  }
  return out;
}

export function wrapSession(session: SessionLike, name: string, ctx: WrapContext): () => void {
  if (isWrapped(session)) return noop;
  const original = session.run;
  const wrapped: SessionLike['run'] = async (feeds, ...rest) => {
    const runId = ctx.nextRunId();
    const callId = ctx.currentCallId;
    const inputs = summarizeAll(feeds, ctx); // before calling through: proxy mode moves the feeds
    const t0 = ctx.now();
    ctx.bus.emit({ type: 'run:start', callId, runId, session: name, inputs, t: t0 });
    let out: Record<string, TensorLike>;
    try {
      out = await original.call(session, feeds, ...rest);
    } catch (e: unknown) {
      const t = ctx.now();
      ctx.bus.emit({ type: 'run:end', callId, runId, session: name, outputs: [], ms: t - t0, error: String(e), t });
      throw e;
    }
    const outputs = summarizeAll(out, ctx);
    const t = ctx.now();
    ctx.bus.emit({ type: 'run:end', callId, runId, session: name, outputs, ms: t - t0, error: null, t });
    return out;
  };
  return replaceMethod(session, 'run', wrapped);
}

/** Wraps every session in `pipe.model.sessions`; the returned function restores them in reverse order. */
export function wrapSessions(sessions: Record<string, SessionLike>, ctx: WrapContext): () => void {
  const restores: (() => void)[] = [];
  for (const [name, session] of Object.entries(sessions)) {
    if (typeof session?.run !== 'function') continue;
    restores.push(wrapSession(session, name, ctx));
  }
  return () => {
    for (let i = restores.length - 1; i >= 0; i--) restores[i]();
  };
}
