/**
 * The inference side of the worker demo: a module worker that owns the Transformers.js
 * pipeline. `./tf.ts` does its top-level `await import(CDN)` here, inside the worker, so the
 * page never loads the library. The inspector wiring is three lines: a local bus,
 * `exposeToPage(bus)` to relay it over `self`, and `attach(pipe, { bus, panel: false })`,
 * which also registers the (default) `TensorStore` as that bus's `'tensor'` handler — so the
 * page panel's "Load values" is answered from this worker's memory.
 *
 * The host protocol below shares the port with the inspector's tagged wire messages: the
 * transport ignores anything without `__tjsi`, and this file ignores anything that has it.
 */
import type { ProgressInfo } from '@huggingface/transformers';
import { InspectorBus, attach, exposeToPage } from '../src/index';
import type { WireMessage } from '../src/index';
import { tf } from './tf';

/** Page → worker. */
export type HostMessage = { text: string };

/** Worker → page. */
export type WorkerMessage =
  | { ready: true }
  | { progress: string }
  | { done: true; dims: number[]; head: number[] }
  | { error: string };

type EmbeddingPipeline = Awaited<ReturnType<typeof loadPipeline>>;

const post = (msg: WorkerMessage): void => self.postMessage(msg);

const bus = new InspectorBus();
exposeToPage(bus);

function loadPipeline(progress_callback: (p: ProgressInfo) => void) {
  return tf.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { progress_callback });
}

function onProgress(p: ProgressInfo): void {
  if (p.status === 'progress') post({ progress: `${p.file} ${p.progress.toFixed(0)}%` });
  else if (p.status === 'ready') post({ progress: `${p.task} · ${p.model} ready` });
}

let loading: Promise<EmbeddingPipeline> | null = null;

function pipeline(): Promise<EmbeddingPipeline> {
  loading ??= loadPipeline(onProgress).then((pipe) => {
    attach(pipe, { bus, panel: false, label: 'feature-extraction · all-MiniLM-L6-v2 (worker)' });
    return pipe;
  });
  return loading;
}

const isHostMessage = (x: unknown): x is HostMessage =>
  typeof x === 'object' && x !== null && (x as Partial<WireMessage>).__tjsi !== 1 && typeof (x as HostMessage).text === 'string';

self.addEventListener('message', (e: MessageEvent<unknown>) => {
  if (!isHostMessage(e.data)) return; // inspector traffic, handled by the transport
  const { text } = e.data;
  void (async () => {
    try {
      const pipe = await pipeline();
      const out = await pipe(text, { pooling: 'mean', normalize: true });
      const head = Array.from(out.data.slice(0, 8), (v) => Number(Number(v).toFixed(4)));
      post({ done: true, dims: [...out.dims], head });
    } catch (err: unknown) {
      post({ error: err instanceof Error ? (err.stack ?? err.message) : String(err) });
    }
  })();
});

post({ ready: true });
