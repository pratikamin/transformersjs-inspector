/**
 * Pipeline boundary: wraps `pipe._call` (the `Callable` hook, exactly as the tokenizer
 * wrapper does) so one pipeline call becomes one correlated row. The wrapper allocates a
 * `callId`, publishes it on `ctx.currentCallId` for the duration of the call (every
 * tokenize / run / logits / token event emitted meanwhile carries it), emits `call:start`
 * with a preview of the first argument and, once the original settles, `result` with a
 * clone-safe copy of the value (tensors become `{ $tensor }` summaries in the store) or the
 * error string.
 */
import type { WrapContext } from '../context';
import type { PipelineLike } from '../types';
import { previewInput, toCloneSafe } from '../preview';
import { isWrapped, noop, replaceMethod } from './session';

const warnedFor = new WeakSet<object>();

export function wrapPipelineCall(pipe: PipelineLike, ctx: WrapContext): () => void {
  if (isWrapped(pipe)) return noop;
  if (typeof pipe._call !== 'function') {
    if (!warnedFor.has(pipe)) {
      warnedFor.add(pipe);
      console.warn('transformersjs-inspector: pipeline has no _call method; call rows will be missing (unsupported Transformers.js version?)');
    }
    return noop;
  }
  const original = pipe._call;
  const task = typeof pipe.task === 'string' ? pipe.task : null;
  const wrapped: PipelineLike['_call'] = async (...args) => {
    const callId = ctx.nextCallId();
    const previous = ctx.currentCallId;
    ctx.currentCallId = callId;
    const t0 = ctx.now();
    ctx.bus.emit({ type: 'call:start', callId, label: ctx.opts.label, task, input: previewInput(args[0]), t: t0 });
    try {
      const result = await original.apply(pipe, args);
      const t = ctx.now();
      ctx.bus.emit({ type: 'result', callId, result: toCloneSafe(result, ctx.store) ?? null, ms: t - t0, error: null, t });
      return result;
    } catch (e: unknown) {
      const t = ctx.now();
      ctx.bus.emit({ type: 'result', callId, result: null, ms: t - t0, error: String(e), t });
      throw e;
    } finally {
      ctx.currentCallId = previous;
    }
  };
  return replaceMethod(pipe, '_call', wrapped);
}
