/**
 * Pure summaries for the panel's simple view: no DOM, no bus. `summarizeResult` reads a
 * pipeline result (as it arrives on the `result` event, tensors already replaced by
 * `$tensor` markers) into one of four readable shapes, and `describeRuns` folds a call's
 * session runs into the one-line `Model` description. Both are exercised by
 * `test/panel-summary.test.ts` against the fixture shapes.
 */
import type { TensorSummary } from '../events';
import { fmtDims, fmtMs } from './dom';
import type { CallView } from './model';

export type ViewMode = 'simple' | 'detail';

export interface LabelScore {
  label: string;
  score: number;
}

export type ResultSummary =
  /** `generated_text` / `text` results and plain strings; several texts are joined with newlines. */
  | { kind: 'text'; text: string }
  /** `{ label, score }` rows in the order the pipeline returned them (a batched result is flattened). */
  | { kind: 'labels'; rows: LabelScore[] }
  /** One `$tensor` marker (possibly the only content of a wrapper object): `embedding · float32 · 384 values`. */
  | { kind: 'tensor'; tensor: TensorSummary; text: string }
  /** Anything else: the JSON `<pre>` as in the detail view. */
  | { kind: 'json'; json: unknown };

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

function isTensorSummary(x: unknown): x is TensorSummary {
  return isRecord(x) && typeof x.id === 'string' && Array.isArray(x.dims) && typeof x.dtype === 'string';
}

/** The marker of `{ $tensor }`; also of a wrapper whose single entry (array or object) resolves to one. */
function tensorOf(value: unknown, depth = 0): TensorSummary | null {
  if (depth > 4 || typeof value !== 'object' || value === null) return null;
  if (isRecord(value) && isTensorSummary(value.$tensor)) return value.$tensor;
  const entries = Array.isArray(value) ? value : Object.values(value);
  return entries.length === 1 ? tensorOf(entries[0], depth + 1) : null;
}

/** `embedding · float32 · 384 values`; leading 1s are dropped from the dims, which are shown only beyond one axis. */
export function describeTensor(t: TensorSummary): string {
  const count = t.dims.reduce((a, b) => a * b, 1);
  let first = 0;
  while (first < t.dims.length - 1 && t.dims[first] === 1) first++;
  const dims = t.dims.slice(first);
  const shape = dims.length > 1 ? ` ${fmtDims(dims)}` : '';
  return `embedding · ${t.dtype}${shape} · ${count} value${count === 1 ? '' : 's'}`;
}

const textOf = (v: unknown): string | null => {
  if (typeof v === 'string') return v;
  if (!isRecord(v)) return null;
  if (typeof v.generated_text === 'string') return v.generated_text;
  if (typeof v.text === 'string') return v.text;
  return null;
};

const isLabelScore = (v: unknown): v is LabelScore => isRecord(v) && typeof v.label === 'string' && typeof v.score === 'number';

/** Flattens one level of batching; `null` unless every leaf is a `{ label, score }`. */
function labelsOf(value: unknown): LabelScore[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const rows: LabelScore[] = [];
  for (const item of value) {
    if (isLabelScore(item)) rows.push(item);
    else if (Array.isArray(item) && item.length > 0 && item.every(isLabelScore)) rows.push(...item);
    else return null;
  }
  return rows;
}

/**
 * `[{ generated_text }]`, `{ generated_text }`, `{ text }` (ASR) and strings → `text`; an
 * array of `{ label, score }` (or one level of arrays of them) → `labels`; a `$tensor` marker,
 * or a wrapper whose only content is one → `tensor`; everything else → `json`.
 */
export function summarizeResult(result: unknown): ResultSummary {
  const direct = textOf(result);
  if (direct !== null) return { kind: 'text', text: direct };
  if (Array.isArray(result) && result.length > 0) {
    const texts = result.map((item) => (Array.isArray(item) && item.length > 0 ? item.map(textOf) : [textOf(item)])).flat();
    if (texts.every((t): t is string => t !== null)) return { kind: 'text', text: texts.join('\n') };
  }
  const rows = labelsOf(result);
  if (rows) return { kind: 'labels', rows };
  const tensor = tensorOf(result);
  if (tensor) return { kind: 'tensor', tensor, text: describeTensor(tensor) };
  return { kind: 'json', json: result };
}

/**
 * The simple view's `Model` line: `1 model run · 34 ms`, or for a generation call (steps
 * seen) with several runs `3 model runs · 1 prefill + 2 decode · 870 ms`. The time is the sum
 * over finished runs (`…` while none has finished); failed runs are counted at the end.
 */
export function describeRuns(call: CallView): string {
  const runs = call.runs;
  if (runs.length === 0) return 'no model runs';
  const parts = [`${runs.length} model run${runs.length === 1 ? '' : 's'}`];
  if (call.steps.length > 0 && runs.length > 1) parts.push(`1 prefill + ${runs.length - 1} decode`);
  const finished = runs.filter((r) => typeof r.ms === 'number' && Number.isFinite(r.ms));
  parts.push(finished.length ? fmtMs(finished.reduce((a, r) => a + (r.ms ?? 0), 0)) : fmtMs(null));
  const errors = runs.filter((r) => r.error !== null).length;
  if (errors > 0) parts.push(`${errors} error${errors === 1 ? '' : 's'}`);
  return parts.join(' · ');
}
