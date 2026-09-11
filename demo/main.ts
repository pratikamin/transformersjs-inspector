/**
 * Demo page driver. Every `<section data-task>` in `index.html` is wired to an entry of the
 * task registry below: the pipeline is built lazily on the first Run, `attach(pipe)` is
 * called the moment `pipeline()` resolves (default bus, auto-mounted panel), and the
 * section's `data-status` cycles `idle → loading → running → done | error`. Adding a task is
 * one `defineTask` entry plus one section in the HTML. `bench()` times the feature-extraction
 * pipeline detached and attached (see the Benchmark button) and publishes `window.__bench`.
 */
import type { ProgressInfo, Tensor, TextClassificationOutput, TextGenerationOutput } from '@huggingface/transformers';
import type { AttachHandle } from '../src/index';
import { VERSION, attach, ensurePanel, getDefaultBus } from '../src/index';
import { tf } from './tf';

type Status = 'idle' | 'loading' | 'running' | 'done' | 'error';

interface TaskSpec<P, R> {
  label: string;
  load(progress_callback: (p: ProgressInfo) => void): Promise<P>;
  run(pipe: P, text: string): Promise<R>;
  format(result: R): string;
}

interface Task {
  label: string;
  loaded(): boolean;
  /** Builds (once) and attaches the pipeline. */
  load(progress_callback: (p: ProgressInfo) => void): Promise<void>;
  /** Runs `text` through the loaded pipeline; returns the text for `[data-output]`. */
  run(text: string): Promise<string>;
  /** Runs `text` without formatting; what the benchmark times. */
  exec(text: string): Promise<unknown>;
  /** Restores the pipeline's original methods; `reattach` wraps them again. */
  detach(): void;
  reattach(): void;
}

function defineTask<P, R>(spec: TaskSpec<P, R>): Task {
  let pipe: P | null = null;
  let loading: Promise<P> | null = null;
  let handle: AttachHandle | null = null;
  const wrap = (p: P): void => {
    // The one-line integration: wrap the instance the page already holds.
    handle = attach(p, { label: spec.label });
  };
  return {
    label: spec.label,
    loaded: () => pipe !== null,
    async load(progress_callback) {
      if (pipe !== null) return;
      loading ??= spec.load(progress_callback).then((p) => {
        wrap(p);
        pipe = p;
        return p;
      });
      try {
        await loading;
      } finally {
        loading = null;
      }
    },
    async run(text) {
      return spec.format(await spec.run(requirePipe(), text));
    },
    exec(text) {
      return spec.run(requirePipe(), text);
    },
    detach() {
      handle?.detach();
      handle = null;
    },
    reattach() {
      if (pipe !== null && handle === null) wrap(pipe);
    },
  };
  function requirePipe(): P {
    if (pipe === null) throw new Error(`${spec.label}: pipeline not loaded`);
    return pipe;
  }
}

function formatTensor(t: Tensor): string {
  const head = Array.from(t.data.slice(0, 8), (v) => Number(Number(v).toFixed(4)));
  return `dims [${t.dims.join(', ')}] · ${t.type}\nfirst ${head.length} values: ${head.join(', ')}${t.data.length > head.length ? ', …' : ''}`;
}

/** First entry of a pipeline output, which is `Single[]` for one input and `Single[][]` for a batch. */
function first<T>(r: T[] | T[][]): T {
  const head: T | T[] | undefined = r[0];
  const single = Array.isArray(head) ? head[0] : head;
  if (single === undefined) throw new Error('empty pipeline output');
  return single;
}

/** The `generated_text` of the first sequence, verbatim (the e2e compares it with the panel's Result). */
function formatGeneration(r: TextGenerationOutput | TextGenerationOutput[]): string {
  const text = first(r).generated_text;
  return typeof text === 'string' ? text : JSON.stringify(text);
}

function formatClassification(r: TextClassificationOutput | TextClassificationOutput[]): string {
  const { label, score } = first(r);
  return `${label} · ${score.toFixed(4)}`;
}

/** One entry per `[data-task]` section. */
export const TASKS: Record<string, Task> = {
  'feature-extraction': defineTask({
    label: 'feature-extraction · all-MiniLM-L6-v2',
    load: (progress_callback) => tf.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { progress_callback }),
    run: (pipe, text) => pipe(text, { pooling: 'mean', normalize: true }),
    format: formatTensor,
  }),
  'text-generation': defineTask({
    label: 'text-generation · tiny-random-Llama',
    // `device: 'auto'`, as in the feasibility prototype's verified generation run.
    load: (progress_callback) => tf.pipeline('text-generation', 'onnx-community/tiny-random-LlamaForCausalLM-ONNX', { device: 'auto', progress_callback }),
    run: (pipe, text) => pipe(text, { max_new_tokens: 3 }),
    format: formatGeneration,
  }),
  'text-classification': defineTask({
    label: 'text-classification · distilbert-sst-2',
    load: (progress_callback) => tf.pipeline('text-classification', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', { progress_callback }),
    run: (pipe, text) => pipe(text),
    format: formatClassification,
  }),
};

// ---- benchmark ---------------------------------------------------------------------

export interface BenchResult {
  /** Median wall time of one embedding with the pipeline detached. */
  detachedMs: number;
  /** Median with the pipeline attached and the panel closed. */
  attachedMs: number;
  /** `attachedMs / detachedMs`. */
  ratio: number;
}

declare global {
  interface Window {
    __bench?: BenchResult;
  }
}

const BENCH_TEXT = 'The inspector sees everything.';
const BENCH_WARMUP = 5;
const BENCH_RUNS = 30;

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

async function timeRuns(task: Task, n: number): Promise<number[]> {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    await task.exec(BENCH_TEXT);
    out.push(performance.now() - t0);
  }
  return out;
}

/**
 * Warm-up, then `BENCH_RUNS` embeddings with the feature-extraction pipeline detached and
 * `BENCH_RUNS` more re-attached with the panel closed (only the reducer and badge run). The
 * result lands on `window.__bench` for `e2e/overhead.spec.ts`.
 */
export async function bench(progress_callback: (p: ProgressInfo) => void): Promise<BenchResult> {
  const task = TASKS['feature-extraction'];
  if (!task.loaded()) await task.load(progress_callback);
  await timeRuns(task, BENCH_WARMUP);
  task.detach();
  const detachedMs = median(await timeRuns(task, BENCH_RUNS));
  ensurePanel(getDefaultBus())?.close();
  task.reattach();
  const attachedMs = median(await timeRuns(task, BENCH_RUNS));
  const result: BenchResult = { detachedMs, attachedMs, ratio: attachedMs / detachedMs };
  window.__bench = result;
  return result;
}

// ---- page wiring -------------------------------------------------------------------

const progressLine = document.querySelector<HTMLElement>('[data-progress]');

function onProgress(p: ProgressInfo): void {
  if (!progressLine) return;
  switch (p.status) {
    case 'initiate':
      progressLine.textContent = `fetching ${p.file}…`;
      break;
    case 'progress':
      progressLine.textContent = `${p.file} ${p.progress.toFixed(0)}%`;
      break;
    case 'done':
      progressLine.textContent = `${p.file} done`;
      break;
    case 'ready':
      progressLine.textContent = `${p.task} · ${p.model} ready`;
      break;
    default:
      break;
  }
}

function wireSection(section: HTMLElement): void {
  const name = section.dataset.task ?? '';
  const task = TASKS[name];
  const textarea = section.querySelector<HTMLTextAreaElement>('textarea');
  const button = section.querySelector<HTMLButtonElement>('[data-run]');
  const output = section.querySelector<HTMLElement>('[data-output]');
  if (!task || !textarea || !button || !output) {
    console.warn(`demo: section "${name}" is incomplete or has no task entry`);
    return;
  }
  const setStatus = (s: Status): void => {
    section.dataset.status = s;
  };
  // The HTML ships the button disabled: this module only runs once `tf.ts` has finished its
  // top-level await on the CDN import, so a click before that would go nowhere.
  button.disabled = false;
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      if (!task.loaded()) {
        setStatus('loading');
        await task.load(onProgress);
      }
      setStatus('running');
      output.textContent = await task.run(textarea.value);
      setStatus('done');
    } catch (e: unknown) {
      output.textContent = e instanceof Error ? (e.stack ?? e.message) : String(e);
      setStatus('error');
    } finally {
      button.disabled = false;
    }
  });
  setStatus('idle');
}

function wireBench(): void {
  const button = document.querySelector<HTMLButtonElement>('[data-benchmark]');
  const output = document.querySelector<HTMLOutputElement>('[data-bench]');
  const holder = button?.closest<HTMLElement>('[data-bench-status]');
  if (!button || !output || !holder) return;
  button.disabled = false; // shipped disabled, same reason as the Run buttons
  button.addEventListener('click', async () => {
    button.disabled = true;
    holder.dataset.benchStatus = 'running';
    output.textContent = 'running…';
    try {
      const { detachedMs, attachedMs, ratio } = await bench(onProgress);
      output.textContent = `detached median ${detachedMs.toFixed(2)} ms · attached median ${attachedMs.toFixed(2)} ms · ratio ${ratio.toFixed(3)}`;
      holder.dataset.benchStatus = 'done';
    } catch (e: unknown) {
      output.textContent = e instanceof Error ? (e.stack ?? e.message) : String(e);
      holder.dataset.benchStatus = 'error';
    } finally {
      button.disabled = false;
    }
  });
}

const versionOut = document.querySelector<HTMLOutputElement>('[data-version]');
if (versionOut) versionOut.textContent = VERSION;
for (const section of document.querySelectorAll<HTMLElement>('section[data-task]')) wireSection(section);
wireBench();
