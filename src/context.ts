/**
 * Shared state for the instance wrappers (`src/wrap/*`): the bus they emit on, the store
 * that hands out tensor ids, the resolved options, id counters and the current call
 * correlation. One `WrapContext` per `attach()`.
 */
import type { InspectorBus } from './bus';
import type { TensorStore } from './store';
import type { TokenizerLike } from './types';
import { DEFAULT_MAX_BYTES } from './store';
import { DEFAULT_HEAD } from './summarize';

export interface InspectorOptions {
  /** Values summarised eagerly per tensor. */
  head: number;
  /** Entries per `logits` event. */
  topK: number;
  /** Nominal byte budget of the tensor store. */
  retainBytes: number;
  /** Keep every step's full logits tensor in the store (512 KB per step on a 128k vocab). */
  retainLogits: boolean;
  /** Row label for `call:start`. */
  label: string;
  /**
   * Escape hatch for the generation wrapper: when the host's Transformers.js rejects a plain
   * array as `logits_processor`, pass its `LogitsProcessorList` class and a real list is
   * built with `.push` instead.
   */
  transformers?: { LogitsProcessorList: new () => { push(p: unknown): void } };
}

export const DEFAULT_TOP_K = 10;
export const DEFAULT_LABEL = 'pipeline';

export const DEFAULT_OPTIONS: Readonly<InspectorOptions> = Object.freeze({
  head: DEFAULT_HEAD,
  topK: DEFAULT_TOP_K,
  retainBytes: DEFAULT_MAX_BYTES,
  retainLogits: false,
  label: DEFAULT_LABEL,
});

/** Fills in the defaults; `undefined` entries in `partial` do not override them. */
export function resolveOptions(partial: Partial<InspectorOptions> = {}): InspectorOptions {
  const out: InspectorOptions = { ...DEFAULT_OPTIONS };
  for (const key of Object.keys(DEFAULT_OPTIONS) as (keyof InspectorOptions)[]) {
    const v = partial[key];
    if (v !== undefined) (out as Record<keyof InspectorOptions, unknown>)[key] = v;
  }
  if (partial.transformers !== undefined) out.transformers = partial.transformers;
  return out;
}

/**
 * Call and run counters live on the *bus*, not the context: every `attach()` makes a new
 * `WrapContext`, and two contexts emitting on one bus (a second pipeline, or a re-attach
 * after `detach()`) must never both issue `c1`, because the panel reducer folds a repeated
 * `call:start` id into the existing row. A `Symbol.for` key keeps the counters shared even
 * when a preload bundle and an `attach()` bundle meet on the same global bus.
 */
const ID_COUNTERS: unique symbol = Symbol.for('transformersjs-inspector.ids');

interface IdCounters {
  calls: number;
  runs: number;
}

function countersOf(bus: InspectorBus): IdCounters {
  const holder = bus as unknown as Record<typeof ID_COUNTERS, IdCounters | undefined>;
  return (holder[ID_COUNTERS] ??= { calls: 0, runs: 0 });
}

export class WrapContext {
  readonly bus: InspectorBus;
  readonly store: TensorStore;
  readonly opts: InspectorOptions;
  /** Set by the pipeline wrapper for the duration of one `pipe._call`; events emitted meanwhile carry it. */
  currentCallId: string | null = null;
  /** Used by `tokenToString`; the tokenizer wrapper sets it, `attach()` may set it earlier. */
  tokenizer: TokenizerLike | null = null;
  private readonly counters: IdCounters;

  constructor(bus: InspectorBus, store: TensorStore, opts: Partial<InspectorOptions> = {}) {
    this.bus = bus;
    this.store = store;
    this.opts = resolveOptions(opts);
    this.counters = countersOf(bus);
  }

  /** Unique per bus: `c1`, `c2`, … across every context that emits on it. */
  nextCallId(): string {
    return `c${++this.counters.calls}`;
  }

  /** Unique per bus, like `nextCallId`. */
  nextRunId(): string {
    return `r${++this.counters.runs}`;
  }

  /**
   * Token string for one id: `_tokenizer.id_to_token(id)` (the raw vocab entry, as the
   * spike verified), else `decode([id])`, else `null`. Never throws.
   */
  tokenToString(id: number): string | null {
    const tok = this.tokenizer;
    if (!tok) return null;
    try {
      const inner = tok._tokenizer;
      if (inner && typeof inner.id_to_token === 'function') {
        const s = inner.id_to_token(id);
        if (typeof s === 'string') return s;
      }
      if (typeof tok.decode === 'function') {
        const s = tok.decode([id]);
        if (typeof s === 'string') return s;
      }
    } catch {
      // fall through: an unknown id or a tokenizer whose internals moved
    }
    return null;
  }

  now(): number {
    return performance.now();
  }
}
