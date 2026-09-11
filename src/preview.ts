/**
 * Turns arbitrary pipeline inputs and results into small, structured-clone-safe values
 * for `call:start` / `result` events. Tensors are reduced to summaries (registered in
 * the `TensorStore` when one is given), typed arrays and long strings are truncated,
 * bigint becomes number, and anything unclonable is dropped.
 */
import type { InputPreview } from './events';
import type { TensorStore } from './store';
import { isTensorLike, summarizeTensor } from './summarize';

export const MAX_STRING = 2000;
export const MAX_ARRAY = 64;
export const MAX_DEPTH = 6;

export type CloneSafeOptions = { maxString?: number; maxArray?: number; maxDepth?: number };

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null;
const isNum = (x: unknown): x is number => typeof x === 'number';
const isStr = (x: unknown): x is string => typeof x === 'string';
const isHttp = (s: string): boolean => /^https?:\/\//i.test(s);
const isSamples = (x: unknown): x is Float32Array | Float64Array => x instanceof Float32Array || x instanceof Float64Array;
const isBlob = (x: unknown): x is Blob => typeof Blob !== 'undefined' && x instanceof Blob;
const isUrl = (x: unknown): x is URL => typeof URL !== 'undefined' && x instanceof URL;
const clip = (s: string, max: number): string => (s.length > max ? s.slice(0, max) : s);

export function previewInput(x: unknown): InputPreview {
  if (isStr(x)) return isHttp(x) ? { kind: 'image', src: x } : { kind: 'text', text: clip(x, MAX_STRING) };
  if (Array.isArray(x) && x.every(isStr)) return { kind: 'texts', texts: x.map((s) => clip(s, MAX_STRING)) };
  if (isSamples(x)) return { kind: 'audio', samples: x.length };
  if (isUrl(x)) return { kind: 'image', src: x.href };
  if (isBlob(x)) return { kind: 'image', src: `blob:${x.type || 'application/octet-stream'};${x.size}` };
  if (isObj(x) && !isTensorLike(x)) {
    if ('audio' in x && (isSamples(x.audio) || isObj(x.audio) || Array.isArray(x.audio))) {
      const audio = x.audio as ArrayLike<unknown>;
      const rate = x.sampling_rate;
      return { kind: 'audio', samples: isNum(audio.length) ? audio.length : 0, ...(isNum(rate) ? { sampleRate: rate } : {}) };
    }
    if (isNum(x.width) && isNum(x.height)) {
      return { kind: 'image', width: x.width, height: x.height, ...(isNum(x.channels) ? { channels: x.channels } : {}) };
    }
  }
  return { kind: 'other', json: toCloneSafe(x) };
}

/** Sentinel for values that are dropped from their container (undefined, functions, symbols). */
const DROP: unique symbol = Symbol('drop');

type Ctx = { store: TensorStore | undefined; maxString: number; maxArray: number; maxDepth: number; stack: Set<object> };

export function toCloneSafe(v: unknown, store?: TensorStore, opts: CloneSafeOptions = {}): unknown {
  const ctx: Ctx = {
    store,
    maxString: opts.maxString ?? MAX_STRING,
    maxArray: opts.maxArray ?? MAX_ARRAY,
    maxDepth: opts.maxDepth ?? MAX_DEPTH,
    stack: new Set(),
  };
  const out = walk(v, '', 0, ctx);
  return out === DROP ? undefined : out;
}

function walk(v: unknown, name: string, depth: number, ctx: Ctx): unknown {
  switch (typeof v) {
    case 'string':
      return clip(v, ctx.maxString);
    case 'number':
    case 'boolean':
      return v;
    case 'bigint':
      return Number(v);
    case 'undefined':
    case 'function':
    case 'symbol':
      return DROP;
    case 'object':
      break;
  }
  if (v === null) return null;
  const o = v as object;

  if (isTensorLike(o)) {
    return { $tensor: ctx.store ? ctx.store.put(o, name) : summarizeTensor(o, name, '') };
  }
  if (o instanceof Error) return { message: o.message };
  if (o instanceof Date) return o.toISOString();
  if (ArrayBuffer.isView(o)) {
    if (o instanceof DataView) return { $truncated: o.byteLength, values: [] };
    return truncatedArray(o as unknown as ArrayLike<unknown>, ctx, (x) => Number(x));
  }

  if (ctx.stack.has(o)) return '[cycle]';
  if (depth >= ctx.maxDepth) return '[deep]';
  ctx.stack.add(o);
  try {
    if (Array.isArray(o)) {
      return truncatedArray(o, ctx, (x, i) => walk(x, `${name}[${i}]`, depth + 1, ctx));
    }
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(o)) {
      const val = walk((o as Record<string, unknown>)[key], key, depth + 1, ctx);
      if (val !== DROP) out[key] = val;
    }
    return out;
  } finally {
    ctx.stack.delete(o);
  }
}

/** First `maxArray` converted items; longer inputs become `{ $truncated: total, values }`. */
function truncatedArray(src: ArrayLike<unknown>, ctx: Ctx, conv: (x: unknown, i: number) => unknown): unknown {
  const total = src.length;
  const n = Math.min(total, ctx.maxArray);
  const values: unknown[] = [];
  for (let i = 0; i < n; i++) {
    const c = conv(src[i], i);
    if (c !== DROP) values.push(c);
  }
  return total > ctx.maxArray ? { $truncated: total, values } : values;
}
