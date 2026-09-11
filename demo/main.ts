/**
 * Demo page driver. Every `<section data-task>` in `index.html` is wired to an entry of the
 * task registry below: the pipeline is built lazily on the first Run, `attach(pipe)` is
 * called the moment `pipeline()` resolves (default bus, auto-mounted panel), and the
 * section's `data-status` cycles `idle → loading → running → done | error`. Adding a task is
 * one `defineTask` entry plus one section in the HTML.
 */
import type { ProgressInfo, Tensor } from '@huggingface/transformers';
import { VERSION, attach } from '../src/index';
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
}

function defineTask<P, R>(spec: TaskSpec<P, R>): Task {
  let pipe: P | null = null;
  let loading: Promise<P> | null = null;
  return {
    label: spec.label,
    loaded: () => pipe !== null,
    async load(progress_callback) {
      if (pipe !== null) return;
      loading ??= spec.load(progress_callback).then((p) => {
        // The one-line integration: wrap the instance the page already holds.
        attach(p, { label: spec.label });
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
      if (pipe === null) throw new Error(`${spec.label}: pipeline not loaded`);
      return spec.format(await spec.run(pipe, text));
    },
  };
}

function formatTensor(t: Tensor): string {
  const head = Array.from(t.data.slice(0, 8), (v) => Number(Number(v).toFixed(4)));
  return `dims [${t.dims.join(', ')}] · ${t.type}\nfirst ${head.length} values: ${head.join(', ')}${t.data.length > head.length ? ', …' : ''}`;
}

/** One entry per `[data-task]` section; story 11 adds text-generation and text-classification. */
export const TASKS: Record<string, Task> = {
  'feature-extraction': defineTask({
    label: 'feature-extraction · all-MiniLM-L6-v2',
    load: (progress_callback) => tf.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { progress_callback }),
    run: (pipe, text) => pipe(text, { pooling: 'mean', normalize: true }),
    format: formatTensor,
  }),
};

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

const versionOut = document.querySelector<HTMLOutputElement>('[data-version]');
if (versionOut) versionOut.textContent = VERSION;
for (const section of document.querySelectorAll<HTMLElement>('section[data-task]')) wireSection(section);
