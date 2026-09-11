/**
 * Page side of the worker demo. No Transformers.js here: the module worker owns the
 * pipeline (`./inference.worker.ts`), `connectWorker(worker)` returns a bus mirroring the
 * worker's inspector bus, and `mountPanel` renders it exactly as it would a local one. The
 * page's own `{ text }` / `{ done }` protocol shares the worker port with the inspector's
 * tagged messages; each side ignores the other's.
 */
import { connectWorker, mountPanel } from '../src/index';
import type { WireMessage } from '../src/index';
import type { HostMessage, WorkerMessage } from './inference.worker';

type Status = 'starting' | 'idle' | 'running' | 'done' | 'error';

const main = document.querySelector<HTMLElement>('main[data-status]');
const textarea = document.querySelector<HTMLTextAreaElement>('textarea');
const button = document.querySelector<HTMLButtonElement>('[data-run]');
const output = document.querySelector<HTMLElement>('[data-output]');
const progressLine = document.querySelector<HTMLElement>('[data-progress]');

const setStatus = (s: Status): void => {
  if (main) main.dataset.status = s;
};
const fail = (message: string): void => {
  if (output) output.textContent = message;
  setStatus('error');
};

const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' });
const bus = connectWorker(worker);
mountPanel(bus, { open: true });

const send = (msg: HostMessage): void => worker.postMessage(msg);
const isWorkerMessage = (x: unknown): x is WorkerMessage =>
  typeof x === 'object' && x !== null && (x as Partial<WireMessage>).__tjsi !== 1;

worker.addEventListener('error', (e) => fail(`worker error: ${e.message}`));
worker.addEventListener('message', (e: MessageEvent<unknown>) => {
  if (!isWorkerMessage(e.data)) return; // inspector traffic, handled by the transport
  const msg = e.data;
  if ('ready' in msg) {
    // The HTML ships the button disabled: the worker has only now finished its CDN import.
    if (button) button.disabled = false;
    setStatus('idle');
  } else if ('progress' in msg) {
    if (progressLine) progressLine.textContent = msg.progress;
  } else if ('done' in msg) {
    if (output) output.textContent = `dims [${msg.dims.join(', ')}] · float32\nfirst ${msg.head.length} values: ${msg.head.join(', ')}, …`;
    if (button) button.disabled = false;
    setStatus('done');
  } else if ('error' in msg) {
    if (button) button.disabled = false;
    fail(msg.error);
  }
});

button?.addEventListener('click', () => {
  button.disabled = true;
  setStatus('running');
  send({ text: textarea?.value ?? '' });
});
