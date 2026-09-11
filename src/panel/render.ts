/**
 * Pure `CallView` → DOM rendering. Nothing here subscribes or handles clicks: the panel
 * owns the state and delegates clicks by `data-action`; this module only builds elements.
 *
 * Lazy tensor values: the panel resolves a `Load values` click through the bus and hands the
 * response to `renderTensorValues(cell, data)`, where `cell` comes from `valuesCellFor(row)`
 * (a `tr.values` inserted right after the tensor's row). The `Generation` section sets its
 * `.bar` widths through the CSSOM property, never a markup attribute, so a strict CSP is fine.
 */
import type { InspectorBus } from '../bus';
import type { InputPreview, TensorData, TensorSummary, TopKEntry } from '../events';
import { empty, fmtBytes, fmtDims, fmtMs, fmtNum, h } from './dom';
import type { Child } from './dom';
import type { CallView, RunView, StepView, TokenizeEvent } from './model';

/** What sections may need beyond the call itself (story 6 issues `ctx.bus.request('tensor')`). */
export interface RenderContext {
  bus: InspectorBus;
}

const EXCERPT_CHARS = 80;

function oneLine(s: string, max = EXCERPT_CHARS): string {
  const flat = s.replace(/\s+/g, ' ').trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

function safeJson(v: unknown): string {
  try {
    const s = JSON.stringify(v, null, 2);
    return s === undefined ? String(v) : s;
  } catch (e: unknown) {
    return `[unserialisable: ${e instanceof Error ? e.message : String(e)}]`;
  }
}

export function inputExcerpt(input: InputPreview | null): string {
  if (!input) return '';
  switch (input.kind) {
    case 'text':
      return oneLine(input.text);
    case 'texts':
      return input.texts.length === 0 ? '[]' : `${oneLine(input.texts[0])}${input.texts.length > 1 ? ` (+${input.texts.length - 1})` : ''}`;
    case 'image':
      return `image${input.width !== undefined && input.height !== undefined ? ` ${input.width}×${input.height}` : ''}`;
    case 'audio':
      return `audio ${input.samples} samples${input.sampleRate ? ` @ ${input.sampleRate} Hz` : ''}`;
    case 'other':
      return oneLine(safeJson(input.json));
  }
}

function statusOf(call: CallView): 'pending' | 'ok' | 'err' {
  if (call.error !== null) return 'err';
  return call.done ? 'ok' : 'pending';
}

// ---- summary row ----------------------------------------------------------------

/** One clickable line per call: `#n`, label, input excerpt, wall time and a status dot. */
export function renderRowSummary(call: CallView): HTMLElement {
  const status = statusOf(call);
  return h(
    'div',
    { class: 'summary', data: { call: call.id, action: 'expand' }, title: call.task ? `task: ${call.task}` : undefined },
    h('span', { class: 'n' }, `#${call.n}`),
    h('span', { class: 'label' }, call.label),
    h('span', { class: 'excerpt' }, inputExcerpt(call.input)),
    h('span', { class: 'ms' }, fmtMs(call.ms)),
    h('span', { class: `dot ${status}`, title: status }),
  );
}

// ---- details ----------------------------------------------------------------------

function section(title: string, ...body: Child[]): HTMLElement {
  return h('section', { class: 'section' }, h('h3', null, title), ...body);
}

function renderInput(call: CallView): HTMLElement {
  const input = call.input;
  if (!input) return section('Input', h('div', { class: 'muted' }, call.synthetic ? 'direct call, no pipeline input' : '—'));
  let body: Child;
  switch (input.kind) {
    case 'text':
      body = h('pre', null, input.text);
      break;
    case 'texts':
      body = h('ol', { class: 'texts' }, input.texts.map((t) => h('li', null, h('pre', null, t))));
      break;
    case 'image':
      body = h(
        'div',
        null,
        h('div', { class: 'meta' }, `image ${input.width ?? '?'}×${input.height ?? '?'}${input.channels !== undefined ? `×${input.channels}` : ''}`),
        input.src ? h('div', { class: 'muted' }, input.src) : null,
      );
      break;
    case 'audio':
      body = h('div', { class: 'meta' }, `audio · ${input.samples} samples${input.sampleRate ? ` @ ${input.sampleRate} Hz` : ''}`);
      break;
    case 'other':
      body = h('pre', null, safeJson(input.json));
      break;
  }
  return section('Input', h('div', { class: 'meta' }, `${input.kind}${call.task ? ` · task ${call.task}` : ''}`), body);
}

/**
 * A decoded token string as children: `'∅'` for null; otherwise the text with any leading and
 * trailing whitespace run wrapped in `span.ws` (a tinted background under the real space
 * character, so copy/paste stays intact).
 */
export function tokenText(s: string | null): Child[] {
  if (s === null) return ['∅'];
  const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(s);
  if (!m) return [s];
  const [, lead, mid, trail] = m;
  const out: Child[] = [];
  if (lead) out.push(h('span', { class: 'ws' }, lead));
  if (mid) out.push(mid);
  if (trail) out.push(h('span', { class: 'ws' }, trail));
  return out;
}

function renderTokenize(ev: TokenizeEvent, index: number): HTMLElement {
  const rows = ev.ids.map((ids, row) =>
    h(
      'div',
      { class: 'tokens', data: { row } },
      ids.map((id, i) => {
        const str = ev.tokens[row]?.[i] ?? null;
        const raw = ev.raw?.[row]?.[i] ?? null;
        return h('span', { class: 'chip', title: `id ${id} · raw ${raw ?? '∅'}` }, h('span', { class: 'chip-id' }, String(id)), h('span', { class: 'chip-str' }, tokenText(str)));
      }),
    ),
  );
  const total = ev.ids.reduce((a, r) => a + r.length, 0);
  return h('div', { class: 'tokenize', data: { tokenize: index } }, h('div', { class: 'meta' }, `${total} tokens · ${ev.ids.length} row${ev.ids.length === 1 ? '' : 's'} · ${fmtMs(ev.ms)}`), rows);
}

function fmtHead(t: TensorSummary): string {
  if (t.head === null) return '—';
  const shown = t.head.map(fmtNum).join(', ');
  return t.size > t.head.length ? `${shown}, …` : shown;
}

const TENSOR_COLUMNS = ['name', 'dtype', 'dims', 'location', 'bytes', 'head', ''];

/**
 * One `<tr data-tensor=id>` per summary. The `Load values` button carries both
 * `data-action="load"` and `data-tensor="<id>"`; the panel's click delegation resolves it.
 */
export function renderTensorTable(tensors: TensorSummary[]): HTMLElement {
  return h(
    'table',
    { class: 'tensors' },
    h('thead', null, h('tr', null, TENSOR_COLUMNS.map((c) => h('th', null, c)))),
    h(
      'tbody',
      null,
      tensors.map((t) =>
        h(
          'tr',
          { data: { tensor: t.id } },
          h('td', { class: 'name' }, t.name),
          h('td', null, t.dtype),
          h('td', null, fmtDims(t.dims)),
          h('td', null, t.location),
          h('td', { class: 'num', title: `${t.size} elements` }, fmtBytes(t.bytes)),
          h('td', { class: 'head' }, fmtHead(t)),
          h('td', null, t.id ? h('button', { class: 'btn', data: { action: 'load', tensor: t.id } }, 'Load values') : null),
        ),
      ),
    ),
  );
}

function renderRun(run: RunView, index: number): HTMLElement {
  const status = run.error !== null ? ` · error` : run.done ? '' : ' · running';
  return h(
    'div',
    { class: 'run', data: { run: run.runId } },
    h('div', { class: 'meta' }, `${index + 1}. ${run.session} · ${run.runId} · ${fmtMs(run.ms)}${status}`),
    run.error !== null ? h('div', { class: 'error' }, run.error) : null,
    h('h4', null, `Inputs (${run.inputs.length})`),
    run.inputs.length ? renderTensorTable(run.inputs) : h('div', { class: 'muted' }, 'none'),
    h('h4', null, `Outputs (${run.outputs.length})`),
    run.done ? (run.outputs.length ? renderTensorTable(run.outputs) : h('div', { class: 'muted' }, 'none')) : h('div', { class: 'muted' }, 'pending…'),
  );
}

// ---- lazy tensor values ----------------------------------------------------------

/** Values rendered before the `… N more` note; keeps a 128k-logit readback from freezing the tab. */
export const MAX_VALUES = 4096;

/**
 * Finds or creates the `tr.values` that sits directly under a `[data-tensor]` table row and
 * returns its single cell, the element `renderTensorValues` fills. The same tensor id may
 * appear in several tables (Session runs and Result); each row owns its own cell.
 */
export function valuesCellFor(row: HTMLElement): HTMLElement {
  const next = row.nextElementSibling;
  if (next instanceof HTMLElement && next.classList.contains('values')) {
    const existing = next.querySelector<HTMLElement>('td');
    if (existing) return existing;
  }
  const cell = h('td', { class: 'values-cell', colSpan: TENSOR_COLUMNS.length });
  const tr = h('tr', { class: 'values', data: { values: row.dataset.tensor ?? '' } }, cell);
  row.parentNode?.insertBefore(tr, row.nextSibling);
  return cell;
}

function valuesOf(data: ArrayBufferView | string[]): { shown: string[]; total: number } {
  if (Array.isArray(data)) return { shown: data.slice(0, MAX_VALUES), total: data.length };
  if (!('length' in data)) return { shown: [], total: 0 };
  const arr = data as unknown as ArrayLike<number | bigint | boolean>;
  const n = Math.min(arr.length, MAX_VALUES);
  const shown: string[] = new Array<string>(n);
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    shown[i] = fmtNum(typeof v === 'bigint' ? Number(v) : v);
  }
  return { shown, total: arr.length };
}

/**
 * Fills `el` with a `TensorData` response: up to `MAX_VALUES` values (bigint arrays through
 * `Number`, string arrays verbatim) followed by a `… N more` note when truncated, or the
 * error string (`unknown`, `evicted`, `disposed`, a timeout message) for `{ error }` responses.
 */
export function renderTensorValues(el: HTMLElement, data: TensorData): void {
  empty(el);
  if ('error' in data) {
    el.appendChild(h('div', { class: 'error', data: { valuesError: '' } }, data.error));
    return;
  }
  const { shown, total } = valuesOf(data.data);
  const more = total - shown.length;
  el.appendChild(h('div', { class: 'meta' }, `${data.dtype} ${fmtDims(data.dims)} · ${total} value${total === 1 ? '' : 's'}`));
  el.appendChild(h('div', { class: 'values-list' }, shown.length ? shown.join(', ') : h('span', { class: 'muted' }, 'empty')));
  if (more > 0) el.appendChild(h('div', { class: 'muted', data: { more } }, `… ${more} more`));
}

// ---- generation ----------------------------------------------------------------

const TOPK_COLUMNS = ['token', 'id', 'logit', 'prob'];

function clampPct(prob: number): string {
  const pct = Number.isFinite(prob) ? Math.min(100, Math.max(0, prob * 100)) : 0;
  return `${pct.toFixed(1)}%`;
}

function renderTopK(entries: TopKEntry[], picked: number[]): HTMLElement {
  return h(
    'table',
    { class: 'topk' },
    h('thead', null, h('tr', null, TOPK_COLUMNS.map((c) => h('th', null, c)))),
    h(
      'tbody',
      null,
      entries.map((e) => {
        const bar = h('div', { class: 'bar' });
        // CSSOM property write; the stylesheet's `.bar { width: 0 }` is the fallback.
        bar.style.width = clampPct(e.prob);
        return h(
          'tr',
          { class: picked.includes(e.id) ? 'picked' : undefined, data: { token: e.id } },
          h('td', { class: 'tok', title: e.raw ?? '' }, tokenText(e.token)),
          h('td', { class: 'num' }, String(e.id)),
          h('td', { class: 'num' }, fmtNum(e.logit)),
          h('td', { class: 'prob' }, h('span', { class: 'prob-text' }, fmtNum(e.prob)), bar),
        );
      }),
    ),
  );
}

function renderStep(step: StepView): HTMLElement {
  const ids = step.ids ?? [];
  const tokenLabel = ids.length ? `token ${ids.join(', ')}${step.text !== undefined && step.text !== null ? ` "${step.text}"` : ''}` : 'no token yet';
  const logitsText = step.tensorId ? ` · logits ${step.tensorId}` : '';
  return h(
    'div',
    { class: 'step', data: { step: step.step } },
    h('div', { class: 'meta step-head' }, `step ${step.step} · ${tokenLabel}${logitsText}`),
    step.topK ? (step.topK.length ? renderTopK(step.topK, ids) : h('div', { class: 'muted' }, 'empty top-k')) : h('div', { class: 'muted' }, 'no logits captured'),
  );
}

/** Per-step token and top-k table; only called when `call.steps.length > 0`. */
export function renderGeneration(call: CallView): HTMLElement {
  return section('Generation', h('div', { class: 'meta' }, `${call.steps.length} step${call.steps.length === 1 ? '' : 's'}`), call.steps.map(renderStep));
}

// ---- result ---------------------------------------------------------------------

function isTensorSummary(x: unknown): x is TensorSummary {
  return typeof x === 'object' && x !== null && typeof (x as TensorSummary).id === 'string' && Array.isArray((x as TensorSummary).dims) && typeof (x as TensorSummary).dtype === 'string';
}

/** Collects `{ $tensor: TensorSummary }` markers left by `toCloneSafe` anywhere in a result. */
export function collectResultTensors(value: unknown, out: TensorSummary[] = [], depth = 0): TensorSummary[] {
  if (depth > 8 || typeof value !== 'object' || value === null) return out;
  const marker = (value as { $tensor?: unknown }).$tensor;
  if (isTensorSummary(marker)) {
    out.push(marker);
    return out;
  }
  const children = Array.isArray(value) ? value : Object.values(value as Record<string, unknown>);
  for (const c of children) collectResultTensors(c, out, depth + 1);
  return out;
}

function renderResult(call: CallView): HTMLElement {
  if (!call.done) return section('Result', h('div', { class: 'muted' }, 'pending…'));
  if (call.error !== null) return section('Result', h('div', { class: 'error' }, call.error));
  if (call.synthetic) return section('Result', h('div', { class: 'muted' }, 'direct call, no pipeline result'));
  const tensors = collectResultTensors(call.result);
  return section('Result', h('div', { class: 'meta' }, fmtMs(call.ms)), tensors.length ? renderTensorTable(tensors) : null, h('pre', null, safeJson(call.result)));
}

/**
 * Everything below a summary row: Input, Tokenizer, Session runs, Generation and Result.
 * Re-rendered wholesale when the call updates while expanded, which drops any values already
 * loaded into it; the panel owns the bus, so `ctx` is reserved for sections that request.
 */
export function renderRowDetails(call: CallView, ctx: RenderContext): HTMLElement {
  void ctx;
  const sections: Child[] = [renderInput(call)];
  if (call.tokenize.length > 0) sections.push(section('Tokenizer', call.tokenize.map(renderTokenize)));
  if (call.runs.length > 0) sections.push(section('Session runs', call.runs.map(renderRun)));
  if (call.steps.length > 0) sections.push(renderGeneration(call));
  sections.push(renderResult(call));
  return h('div', { class: 'details', data: { details: call.id } }, sections);
}
