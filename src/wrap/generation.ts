/**
 * Generation boundary: replaces `model.generate` on the instance so every call runs with a
 * duck-typed logits processor and streamer of ours merged in front of the host's. Neither
 * needs the `LogitsProcessor` / `TextStreamer` classes: `LogitsProcessorList._call` invokes
 * each entry as `processor(input_ids, logits)` and `generate` only ever calls
 * `streamer.put()` / `streamer.end()` (verified in spike/b.html and
 * `modeling_utils.js` 4.2.0, lines 944-1031), so plain callables satisfy both contracts.
 *
 * The processor emits `logits` (top-k over batch row 0, softmax-normalised) and the streamer
 * emits `token` (the ids of the step, prompt `put` skipped); both count steps from 0 per
 * `generate` call. Logits are ~512 KB per step on a 128k vocab, so top-k runs on the hot
 * path and the full tensor is only kept when `opts.retainLogits` is set.
 */
import type { WrapContext } from '../context';
import type { TopKEntry } from '../events';
import type { GenerateLike, LogitsProcessorLike, StreamerLike, TensorLike } from '../types';
import { cpuData, isTensorLike } from '../summarize';
import { isWrapped, noop, replaceMethod } from './session';

export interface TopKResult {
  /** `k` entries (fewer when the vocab is smaller), by `prob` descending; ties keep the lower id first. */
  entries: { id: number; logit: number; prob: number }[];
  /** `data.length`. */
  vocab: number;
}

/**
 * Top-`k` entries of one logits row with softmax probabilities. One pass for the max, one
 * for the denominator, then a partial selection that keeps only `k` candidates (an insertion
 * into a small sorted buffer, with the buffer's minimum as the rejection threshold) instead
 * of sorting the whole vocab.
 */
export function topKFromLogits(data: ArrayLike<number>, k: number): TopKResult {
  const vocab = data.length;
  const n = Math.min(Math.max(0, Math.floor(k)), vocab);
  if (n === 0) return { entries: [], vocab };

  let max = -Infinity;
  for (let i = 0; i < vocab; i++) if (data[i] > max) max = data[i];
  let denom = 0;
  for (let i = 0; i < vocab; i++) denom += Math.exp(data[i] - max);

  const ids = new Array<number>(n);
  const vals = new Array<number>(n);
  let size = 0;
  for (let i = 0; i < vocab; i++) {
    const v = data[i];
    if (size === n && !(v > vals[size - 1])) continue; // below (or tied with) the current k-th: reject
    let pos = size < n ? size : n - 1; // slot to fill: append, or overwrite the k-th
    while (pos > 0 && vals[pos - 1] < v) {
      vals[pos] = vals[pos - 1];
      ids[pos] = ids[pos - 1];
      pos--;
    }
    vals[pos] = v;
    ids[pos] = i;
    if (size < n) size++;
  }

  const entries = new Array<{ id: number; logit: number; prob: number }>(size);
  for (let j = 0; j < size; j++) entries[j] = { id: ids[j], logit: vals[j], prob: Math.exp(vals[j] - max) / denom };
  return { entries, vocab };
}

/** Per-`generate`-call state shared by the processor and the streamer. */
class GenerationCapture {
  private logitsStep = 0;
  private tokenStep = 0;
  private sawPrompt = false;
  readonly processor: LogitsProcessorLike;
  readonly streamer: StreamerLike;

  constructor(private readonly ctx: WrapContext) {
    this.processor = (_input_ids, logits) => {
      this.emitLogits(logits, this.logitsStep++);
      return logits;
    };
    this.streamer = {
      put: (value) => {
        if (!this.sawPrompt) {
          this.sawPrompt = true; // the first `put` is the prompt (`skip_prompt` in TextStreamer terms)
          return;
        }
        this.emitToken(value, this.tokenStep++);
      },
      end: () => {
        this.sawPrompt = false;
      },
    };
  }

  private emitLogits(logits: TensorLike, step: number): void {
    const ctx = this.ctx;
    let topK: TopKEntry[] = [];
    let vocab = 0;
    let tensorId: string | null = null;
    if (isTensorLike(logits)) {
      const dims = logits.dims;
      const data = cpuData(logits);
      vocab = dims.length > 0 ? Number(dims[dims.length - 1]) : (data?.length ?? 0);
      if (data !== null && vocab > 0) {
        const row = rowZero(data, vocab);
        topK = topKFromLogits(row, ctx.opts.topK).entries.map((e) => ({ ...e, token: ctx.tokenToString(e.id) }));
      }
      if (ctx.opts.retainLogits) tensorId = ctx.store.put(logits, 'logits').id;
    }
    ctx.bus.emit({ type: 'logits', callId: ctx.currentCallId, step, vocab, topK, tensorId, t: ctx.now() });
  }

  private emitToken(value: unknown, step: number): void {
    const ctx = this.ctx;
    const ids = idsOfRowZero(value);
    const strs = ids.map((id) => ctx.tokenToString(id));
    const text = strs.some((s) => s !== null) ? strs.map((s) => s ?? '').join('') : null;
    ctx.bus.emit({ type: 'token', callId: ctx.currentCallId, step, ids, text, t: ctx.now() });
  }
}

/** Batch row 0 of a `[batch, vocab]` buffer without copying when it is a typed array. */
function rowZero(data: ArrayLike<unknown>, vocab: number): ArrayLike<number> {
  if (data.length <= vocab) return data as ArrayLike<number>;
  if (ArrayBuffer.isView(data) && 'subarray' in data) return (data as unknown as { subarray(a: number, b: number): ArrayLike<number> }).subarray(0, vocab);
  return Array.prototype.slice.call(data, 0, vocab) as number[];
}

/** `value[0]` as numbers: `generate` passes `bigint[][]` (one row per batch entry). */
function idsOfRowZero(value: unknown): number[] {
  if (value === null || typeof value !== 'object') return [];
  const row = Array.isArray(value) ? (value as unknown[])[0] : isTensorLike(value) ? cpuData(value) : null;
  if (row === null || row === undefined || typeof row !== 'object') return [];
  return Array.from(row as ArrayLike<unknown>, Number);
}

function isIterable(x: object): x is Iterable<unknown> {
  return typeof (x as { [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function';
}

/** The host's processors as an array: an iterable (`LogitsProcessorList`, array), a `.processors` holder, or nothing. */
function hostProcessors(host: unknown): unknown[] {
  if (host === null || host === undefined) return [];
  if (typeof host !== 'object' && typeof host !== 'function') return [];
  if (isIterable(host)) return [...host];
  const inner = (host as { processors?: unknown }).processors;
  return Array.isArray(inner) ? [...inner] : [];
}

/**
 * The caller's list plus `ours` at the end (so we see the logits the sampler sees). A plain
 * array by default — `LogitsProcessorList.extend` spreads it — or a real list built with
 * `.push` when `ctx.opts.transformers.LogitsProcessorList` is provided.
 */
export function mergeProcessors(host: unknown, ours: LogitsProcessorLike, ctx: WrapContext): unknown {
  const all = [...hostProcessors(host), ours];
  const List = ctx.opts.transformers?.LogitsProcessorList;
  if (!List) return all;
  const list = new List();
  for (const p of all) list.push(p);
  return list;
}

/** `ours` first, then the host's streamer, for both `put` and `end`; just `ours` when the host has none. */
export function mergeStreamer(host: unknown, ours: StreamerLike): StreamerLike {
  if (host === null || typeof host !== 'object' || typeof (host as StreamerLike).put !== 'function') return ours;
  const h = host as Partial<StreamerLike>;
  return {
    put(value) {
      ours.put(value);
      h.put?.(value);
    },
    end() {
      ours.end();
      h.end?.();
    },
  };
}

export function wrapGenerate(model: GenerateLike, ctx: WrapContext): () => void {
  if (isWrapped(model)) return noop;
  if (typeof model.generate !== 'function') return noop;
  const original = model.generate;
  const wrapped: GenerateLike['generate'] = (opts) => {
    const capture = new GenerationCapture(ctx);
    const o: Record<string, unknown> = opts ?? {};
    return original.call(model, {
      ...o,
      logits_processor: mergeProcessors(o.logits_processor, capture.processor, ctx),
      streamer: mergeStreamer(o.streamer, capture.streamer),
    });
  };
  return replaceMethod(model, 'generate', wrapped);
}
