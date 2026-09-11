/**
 * An unmodified Transformers.js host: no import of the inspector, no `attach()`. The only
 * concession is `device: 'auto'` — with the ORT symbol injected, Transformers.js's own
 * device list is empty and its default `'wasm'` throws (docs/01-research.md).
 *
 * This is the second module script on `preload.html`. On the dev server the two scripts
 * stay separate and execute in document order. `build:demo` merges them into one chunk, and
 * a bundler hoists a *shared* chunk's import above that chunk's own body: importing
 * `./tf.ts` here (shared with `index.html`) put its top-level `await import(Transformers)`
 * in a chunk that evaluated before the preload installed the shim. So this host loads
 * Transformers.js itself, exactly as a CDN-based page would, and the dynamic import runs in
 * body order — after `preload-entry.ts` has set the symbol.
 */
import type * as Transformers from '@huggingface/transformers';

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
const SENTENCE = 'The inspector sees everything.';

const main = document.querySelector<HTMLElement>('main[data-status]');
const output = document.querySelector<HTMLElement>('[data-output]');
const progressLine = document.querySelector<HTMLElement>('[data-progress]');

function onProgress(p: Transformers.ProgressInfo): void {
  if (!progressLine) return;
  if (p.status === 'progress') progressLine.textContent = `${p.file} ${p.progress.toFixed(0)}%`;
  else if (p.status === 'ready') progressLine.textContent = `${p.task} · ${p.model} ready`;
}

try {
  const tf = (await import(/* @vite-ignore */ TRANSFORMERS_URL)) as typeof Transformers;
  const pipe = await tf.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { device: 'auto', progress_callback: onProgress });
  const out = await pipe(SENTENCE, { pooling: 'mean', normalize: true });
  const head = Array.from(out.data.slice(0, 8), (v) => Number(Number(v).toFixed(4)));
  if (output) output.textContent = `dims [${out.dims.join(', ')}] · ${out.type}\nfirst ${head.length} values: ${head.join(', ')}, …`;
  if (main) main.dataset.status = 'done';
} catch (e: unknown) {
  if (output) output.textContent = e instanceof Error ? (e.stack ?? e.message) : String(e);
  if (main) main.dataset.status = 'error';
}
