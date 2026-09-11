/**
 * Pure `CallView` → DOM rendering. Nothing here subscribes or handles clicks: the panel
 * owns the state and delegates clicks by `data-action`; this module only builds elements.
 *
 * Extension points reserved for story 6 (lazy tensor values, generation section):
 *   - `renderTensorValues(el, data)` fills a `tr.values` inserted after a `[data-tensor]` row.
 *   - `renderGeneration(call)` becomes a section between "Session runs" and "Result".
 */
import type { InspectorBus } from '../bus';
import type { InputPreview, TensorSummary } from '../events';
import { fmtBytes, fmtDims, fmtMs, fmtNum, h } from './dom';
import type { Child } from './dom';
import type { CallView, RunView, TokenizeEvent } from './model';

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

function renderTokenize(ev: TokenizeEvent, index: number): HTMLElement {
  const rows = ev.ids.map((ids, row) =>
    h(
      'div',
      { class: 'tokens', data: { row } },
      ids.map((id, i) => {
        const str = ev.tokens[row]?.[i] ?? null;
        return h('span', { class: 'chip', title: `id ${id}` }, h('span', { class: 'chip-id' }, String(id)), h('span', { class: 'chip-str' }, str ?? '∅'));
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

/** One `<tr data-tensor=id>` per summary; the `Load values` button is wired by the panel (story 6). */
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
          h('td', null, t.id ? h('button', { class: 'btn', data: { action: 'load' } }, 'Load values') : null),
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
 * Everything below a summary row: Input, Tokenizer, Session runs, (Generation, story 6)
 * and Result. Re-rendered wholesale when the call updates while expanded.
 */
export function renderRowDetails(call: CallView, ctx: RenderContext): HTMLElement {
  void ctx; // Unused until story 6's Generation section and value loading need the bus.
  const sections: Child[] = [renderInput(call)];
  if (call.tokenize.length > 0) sections.push(section('Tokenizer', call.tokenize.map(renderTokenize)));
  if (call.runs.length > 0) sections.push(section('Session runs', call.runs.map(renderRun)));
  // Story 6: when `call.steps.length > 0`, push `renderGeneration(call)` here (per-step token
  // id/text and a top-k table whose `.bar` widths are set via CSSOM `style.width`).
  sections.push(renderResult(call));
  return h('div', { class: 'details', data: { details: call.id } }, sections);
}
