/**
 * Tokenizer boundary: wraps `tokenizer._call` (the `Callable` hook, verified in the feasibility prototype)
 * on the instance so the raw text pairs up with the ids and token strings it produced,
 * and emits one `tokenize` event per call. `_call` and `_tokenizer` are underscore-private
 * in Transformers.js 4.x (research risk 1): when `_call` is not a function the wrapper
 * warns once per instance and degrades to a no-op instead of throwing.
 */
import type { WrapContext } from '../context';
import type { TensorLike, TokenizerLike } from '../types';
import { isCpuResident, isTensorLike } from '../summarize';
import { isWrapped, noop, replaceMethod } from './session';

/**
 * `input_ids` as plain rows: `dims [b, s]` → `b` rows of `s` numbers (bigint → Number).
 * This is the one place besides `src/summarize.ts` that reads `tensor.data`; it only does
 * so for CPU-resident tensors (tokenizer output always is) and returns `[]` otherwise.
 */
export function idsFrom(input_ids: TensorLike): number[][] {
  if (!isCpuResident(input_ids)) return [];
  let data: ArrayLike<unknown>;
  try {
    data = input_ids.data;
  } catch {
    return [];
  }
  const dims = input_ids.dims;
  const rows = dims.length >= 2 ? Math.max(0, Number(dims[0])) : 1;
  if (rows === 0 || data.length === 0) return Array.from({ length: rows }, () => []);
  const cols = Math.floor(data.length / rows);
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row = new Array<number>(cols);
    for (let c = 0; c < cols; c++) row[c] = Number(data[r * cols + c]);
    out.push(row);
  }
  return out;
}

function textOf(input: unknown): string | string[] {
  if (typeof input === 'string') return input;
  if (Array.isArray(input)) return input.map((x) => (typeof x === 'string' ? x : String(x)));
  return String(input);
}

const warnedFor = new WeakSet<object>();

export function wrapTokenizer(tok: TokenizerLike, ctx: WrapContext): () => void {
  if (isWrapped(tok)) return noop;
  if (typeof tok._call !== 'function') {
    if (!warnedFor.has(tok)) {
      warnedFor.add(tok);
      console.warn('transformersjs-inspector: tokenizer has no _call method; tokenizer rows will be missing (unsupported Transformers.js version?)');
    }
    return noop;
  }
  if (ctx.tokenizer === null) ctx.tokenizer = tok;
  const original = tok._call;
  const wrapped: TokenizerLike['_call'] = (text, opts) => {
    const t0 = ctx.now();
    const result = original.call(tok, text, opts);
    const ms = ctx.now() - t0;
    const ids = typeof result === 'object' && result !== null && isTensorLike(result.input_ids) ? idsFrom(result.input_ids) : [];
    const strs = ids.map((row) => row.map((id) => ctx.tokenStrings(id)));
    const tokens = strs.map((row) => row.map((s) => s.text));
    const raw = strs.map((row) => row.map((s) => s.raw));
    ctx.bus.emit({ type: 'tokenize', callId: ctx.currentCallId, text: textOf(text), ids, tokens, raw, ms, t: ctx.now() });
    return result;
  };
  const restore = replaceMethod(tok, '_call', wrapped);
  return () => {
    restore();
    if (ctx.tokenizer === tok) ctx.tokenizer = null;
  };
}
