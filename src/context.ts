/**
 * Shared state for the instance wrappers (`src/wrap/*`): the bus they emit on, the store
 * that hands out tensor ids, the resolved options, id counters and the current call
 * correlation. One `WrapContext` per `attach()`.
 */
import type { InspectorBus } from './bus';
import type { ReplayRegistry } from './replay';
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

export interface TokenStrings {
  /** The vocab string (`id_to_token`), e.g. `▁word`, `##ing`, `Ġfilm`; `null` when unknown. */
  raw: string | null;
  /** The decoded display text, e.g. `word`, `ing`, ` film`; `null` when neither lookup worked. */
  text: string | null;
}

/** `tokenStrings` memo entries before it is cleared. */
export const TOKEN_CACHE_MAX = 4096;

const DECODE_ONE = Object.freeze({ skip_special_tokens: false, clean_up_tokenization_spaces: false });

function lookupTokenStrings(tok: TokenizerLike, id: number): TokenStrings {
  let raw: string | null = null;
  try {
    const inner = tok._tokenizer;
    if (inner && typeof inner.id_to_token === 'function') {
      const s = inner.id_to_token(id);
      if (typeof s === 'string') raw = s;
    }
  } catch {
    // an unknown id or a tokenizer whose internals moved: no vocab string
  }
  let text: string | null = raw;
  try {
    if (typeof tok.decode === 'function') {
      const s = tok.decode([id], DECODE_ONE);
      if (typeof s === 'string') text = s;
    }
  } catch {
    // decode failed: show the vocab string instead
  }
  return { raw, text };
}

export class WrapContext {
  readonly bus: InspectorBus;
  readonly store: TensorStore;
  readonly opts: InspectorOptions;
  /** Set by the pipeline wrapper for the duration of one `pipe._call`; events emitted meanwhile carry it. */
  currentCallId: string | null = null;
  /** Used by `tokenStrings`; the tokenizer wrapper sets it, `attach()` may set it earlier. */
  tokenizer: TokenizerLike | null = null;
  /** Set by the replay handler right before it invokes the pipeline; the wrapper moves it onto the next `call:start.replayOf`. */
  pendingReplayOf: string | null = null;
  /** The id the pipeline wrapper allocated most recently (read synchronously by the replay handler). */
  lastCallId: string | null = null;
  /** Where the pipeline wrapper records `{ pipe, args }` per call; `null` when replay is off. */
  replays: ReplayRegistry | null = null;
  private readonly counters: IdCounters;
  /** `tokenStrings` memo, valid for `tokenCacheFor` only (the tokenizer may be swapped between calls). */
  private readonly tokenCache = new Map<number, TokenStrings>();
  private tokenCacheFor: TokenizerLike | null = null;

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
   * Both strings for one id: `raw` is the vocab entry (`_tokenizer.id_to_token(id)`, as the
   * spike verified; `null` when unknown), `text` is the decoded display text
   * (`decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false })`, falling
   * back to `raw` when `decode` is missing, throws or returns a non-string). Never throws.
   * Memoised per tokenizer: generation asks `topK` times per step.
   */
  tokenStrings(id: number): TokenStrings {
    const tok = this.tokenizer;
    if (!tok) return { raw: null, text: null };
    if (this.tokenCacheFor !== tok) {
      this.tokenCache.clear();
      this.tokenCacheFor = tok;
    }
    const hit = this.tokenCache.get(id);
    if (hit) return hit;
    const out = lookupTokenStrings(tok, id);
    if (this.tokenCache.size >= TOKEN_CACHE_MAX) this.tokenCache.clear();
    this.tokenCache.set(id, out);
    return out;
  }

  /** `tokenStrings(id).text`, for callers that only need the display text. */
  tokenToString(id: number): string | null {
    return this.tokenStrings(id).text;
  }

  now(): number {
    return performance.now();
  }
}
