/**
 * `attach(pipe, opts?)`: the headline API. Wraps, on the instances the host already holds,
 * every `pipe.model.sessions[name].run`, `pipe.tokenizer._call` (when present),
 * `pipe.model.generate` (when a function) and finally `pipe._call`, all sharing one
 * `WrapContext` so the events of one pipeline call carry one `callId`. `detach()` restores
 * them in reverse order and is idempotent. Without an explicit bus/store the process-wide
 * defaults are used, and the panel is mounted once per bus wherever a DOM exists.
 */
import type { InspectorOptions } from './context';
import type { PanelOptions } from './panel/panel';
import type { InspectorBus } from './bus';
import type { TensorStore } from './store';
import type { GenerateLike, PipelineLike } from './types';
import { InspectorError } from './bus';
import { WrapContext } from './context';
import { ensurePanel, getDefaultBus, getDefaultStore } from './default';
import { wrapGenerate } from './wrap/generation';
import { wrapPipelineCall } from './wrap/pipeline';
import { wrapSessions } from './wrap/session';
import { wrapTokenizer } from './wrap/tokenizer';

export interface AttachOptions extends Partial<InspectorOptions> {
  /** Bus to emit on; the process-wide default bus otherwise. */
  bus?: InspectorBus;
  /** Store that hands out tensor ids; the process-wide default store otherwise. */
  store?: TensorStore;
  /**
   * `false` mounts no panel; an object is passed to `mountPanel`. Default: mount (a
   * no-op where there is no DOM, e.g. inside a worker).
   */
  panel?: boolean | PanelOptions;
}

export interface AttachHandle {
  bus: InspectorBus;
  store: TensorStore;
  /** Restores every wrapped method to its original; safe to call more than once. */
  detach(): void;
}

const isRecord = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null;

/**
 * Pipelines, models and tokenizers all `extend Callable` in Transformers.js 4.x, whose
 * constructor returns a *function* carrying the instance properties; `typeof` is `'function'`
 * for every one of them, so an object-only guard rejects real instances.
 */
const isInstanceLike = (x: unknown): x is Record<string, unknown> => typeof x === 'function' || isRecord(x);

function assertPipeline(pipe: unknown): asserts pipe is PipelineLike {
  const candidate = isInstanceLike(pipe) ? (pipe as { model?: unknown }) : null;
  const model = candidate?.model;
  const modelSessions = isInstanceLike(model) ? (model as { sessions?: unknown }).sessions : undefined;
  if (!isRecord(modelSessions)) {
    throw new InspectorError('attach(): expected a Transformers.js pipeline with model.sessions');
  }
}

/** `${task} · ${model_type}`, e.g. `feature-extraction · bert`; `pipeline · ?` when neither is known. */
export function defaultLabel(pipe: PipelineLike): string {
  const task = typeof pipe.task === 'string' ? pipe.task : 'pipeline';
  const modelType = pipe.model.config?.model_type;
  return `${task} · ${typeof modelType === 'string' ? modelType : '?'}`;
}

/** Stores already answering `'tensor'` requests on a bus; `attachTo` runs once per pair. */
const registered = new WeakMap<InspectorBus, WeakSet<TensorStore>>();

function registerStore(bus: InspectorBus, store: TensorStore): void {
  let stores = registered.get(bus);
  if (!stores) {
    stores = new WeakSet();
    registered.set(bus, stores);
  }
  if (stores.has(store)) return;
  stores.add(store);
  store.attachTo(bus);
}

export function attach(pipe: unknown, opts: AttachOptions = {}): AttachHandle {
  assertPipeline(pipe);
  const bus = opts.bus ?? getDefaultBus();
  const store = opts.store ?? getDefaultStore();
  registerStore(bus, store);

  const ctx = new WrapContext(bus, store, { ...opts, label: opts.label ?? defaultLabel(pipe) });
  const { model, tokenizer } = pipe;
  if (isInstanceLike(tokenizer)) ctx.tokenizer = tokenizer;

  const restores: (() => void)[] = [wrapSessions(model.sessions, ctx)];
  if (isInstanceLike(tokenizer)) restores.push(wrapTokenizer(tokenizer, ctx));
  if (typeof model.generate === 'function') restores.push(wrapGenerate(model as GenerateLike, ctx));
  restores.push(wrapPipelineCall(pipe, ctx));

  if (opts.panel !== false) ensurePanel(bus, typeof opts.panel === 'object' ? opts.panel : undefined);

  let detached = false;
  return {
    bus,
    store,
    detach() {
      if (detached) return;
      detached = true;
      for (let i = restores.length - 1; i >= 0; i--) restores[i]();
    },
  };
}
