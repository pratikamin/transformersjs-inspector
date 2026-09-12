import { describe, expect, test } from 'vitest';
import type { InspectorEvent, TensorSummary } from '../src/events';
import { createState, reduce } from '../src/panel/model';
import type { CallView, PanelState } from '../src/panel/model';
import { describeRuns, describeTensor, summarizeResult } from '../src/panel/summary';
import { fixtureEvents, fixtureMediaEvents } from './fakes';

/** Folds the events into a state and returns its calls by id. */
function callsOf(events: InspectorEvent[]): Map<string, CallView> {
  const state: PanelState = createState();
  for (const ev of events) reduce(state, ev);
  return state.byId;
}

type ResultEvent = Extract<InspectorEvent, { type: 'result' }>;
const resultOf = (events: InspectorEvent[], callId: string): unknown => events.find((e): e is ResultEvent => e.type === 'result' && e.callId === callId)?.result;

const tensor = (dims: number[], dtype = 'float32', name = 'last_hidden_state'): TensorSummary => ({
  id: 't1',
  name,
  dtype,
  dims,
  location: 'cpu',
  size: dims.reduce((a, b) => a * b, 1),
  bytes: 0,
  head: null,
});

describe('summarizeResult', () => {
  test('c1: the { $tensor } embedding result becomes one line with the squeezed dims and the value count', () => {
    const s = summarizeResult(resultOf(fixtureEvents(), 'c1'));
    expect(s.kind).toBe('tensor');
    if (s.kind !== 'tensor') return;
    expect(s.tensor.id).toBe('t4');
    expect(s.text).toBe('embedding · float32 [7, 384] · 2688 values');
  });

  test('a pooled [1, 384] tensor reads `embedding · float32 · 384 values`; leading 1s are dropped, a scalar keeps one value', () => {
    expect(describeTensor(tensor([1, 384]))).toBe('embedding · float32 · 384 values');
    expect(describeTensor(tensor([384]))).toBe('embedding · float32 · 384 values');
    expect(describeTensor(tensor([1, 1, 5, 6], 'float16'))).toBe('embedding · float16 [5, 6] · 30 values');
    expect(describeTensor(tensor([1, 1]))).toBe('embedding · float32 · 1 value');
    expect(describeTensor(tensor([]))).toBe('embedding · float32 · 1 value');
  });

  test('a wrapper whose only content is a marker is a tensor; two markers or a marker beside data is json', () => {
    const marker = { $tensor: tensor([1, 384]) };
    expect(summarizeResult({ embedding: marker }).kind).toBe('tensor');
    expect(summarizeResult([marker]).kind).toBe('tensor');
    expect(summarizeResult({ a: marker, b: marker }).kind).toBe('json');
    expect(summarizeResult({ a: marker, n: 1 }).kind).toBe('json');
    expect(summarizeResult({ $tensor: { not: 'a summary' } }).kind).toBe('json');
  });

  test('c2: [{ generated_text }] is the text; so are { generated_text }, { text } (ASR) and a plain string', () => {
    expect(summarizeResult(resultOf(fixtureEvents(), 'c2'))).toEqual({ kind: 'text', text: 'hi the quick brown' });
    expect(summarizeResult({ generated_text: 'a' })).toEqual({ kind: 'text', text: 'a' });
    expect(summarizeResult(resultOf(fixtureMediaEvents(), 'c3'))).toEqual({ kind: 'text', text: 'hello' });
    expect(summarizeResult('plain')).toEqual({ kind: 'text', text: 'plain' });
    expect(summarizeResult('')).toEqual({ kind: 'text', text: '' });
  });

  test('batched texts join with newlines; a batch with a non-text entry is json', () => {
    expect(summarizeResult([{ generated_text: 'a' }, { generated_text: 'b' }])).toEqual({ kind: 'text', text: 'a\nb' });
    expect(summarizeResult([[{ generated_text: 'a' }], [{ generated_text: 'b' }, { generated_text: 'c' }]])).toEqual({ kind: 'text', text: 'a\nb\nc' });
    expect(summarizeResult(['x', 'y'])).toEqual({ kind: 'text', text: 'x\ny' });
    expect(summarizeResult([{ generated_text: 'a' }, { other: 1 }]).kind).toBe('json');
  });

  test('c4: label/score rows in the given order; a batched classification is flattened; a mixed list is json', () => {
    const s = summarizeResult(resultOf(fixtureMediaEvents(), 'c4'));
    expect(s).toEqual({
      kind: 'labels',
      rows: [
        { label: 'tabby', score: 0.61 },
        { label: 'tiger cat', score: 0.2 },
        { label: 'Egyptian cat', score: 0.07 },
      ],
    });
    const batched = summarizeResult([
      [{ label: 'POSITIVE', score: 0.99 }],
      [{ label: 'NEGATIVE', score: 0.6 }, { label: 'POSITIVE', score: 0.4 }],
    ]);
    expect(batched.kind).toBe('labels');
    if (batched.kind === 'labels') expect(batched.rows.map((r) => r.label)).toEqual(['POSITIVE', 'NEGATIVE', 'POSITIVE']);
    expect(summarizeResult([{ label: 'a', score: 1 }, { label: 'b' }]).kind).toBe('json');
    expect(summarizeResult([]).kind).toBe('json');
  });

  test('anything else is json, carrying the value untouched', () => {
    for (const v of [null, undefined, 42, true, { scores: [1, 2] }, [1, 2], { text: 1 }]) {
      expect(summarizeResult(v)).toEqual({ kind: 'json', json: v });
    }
  });
});

describe('describeRuns', () => {
  test('c1: one run; c2: three runs with the first called prefill and the times summed', () => {
    const calls = callsOf(fixtureEvents());
    expect(describeRuns(calls.get('c1') as CallView)).toBe('1 model run · 12.7 ms');
    expect(describeRuns(calls.get('c2') as CallView)).toBe('3 model runs · 1 prefill + 2 decode · 18.3 ms');
  });

  test('media fixtures: one run each, no prefill wording without steps', () => {
    const calls = callsOf(fixtureMediaEvents());
    expect(describeRuns(calls.get('c3') as CallView)).toBe('1 model run · 41.2 ms');
    expect(describeRuns(calls.get('c4') as CallView)).toBe('1 model run · 9.8 ms');
  });

  test('several runs without steps are not called prefill; a single generation run is not either', () => {
    const events = fixtureEvents().filter((e) => e.type !== 'logits' && e.type !== 'token');
    expect(describeRuns(callsOf(events).get('c2') as CallView)).toBe('3 model runs · 18.3 ms');
    const one = fixtureEvents().filter((e) => !('runId' in e) || e.runId === 'r2');
    expect(describeRuns(callsOf(one).get('c2') as CallView)).toBe('1 model run · 6.1 ms');
  });

  test('a still-running call shows … for the time; errors are counted; no runs is stated', () => {
    const events = fixtureEvents().filter((e) => e.type !== 'run:end' && e.type !== 'result');
    const c2 = callsOf(events).get('c2') as CallView;
    expect(describeRuns(c2)).toBe('3 model runs · 1 prefill + 2 decode · …');
    const failed = fixtureEvents().map((e) => (e.type === 'run:end' && e.runId === 'r3' ? { ...e, error: 'boom' } : e));
    expect(describeRuns(callsOf(failed).get('c2') as CallView)).toBe('3 model runs · 1 prefill + 2 decode · 18.3 ms · 1 error');
    expect(describeRuns({ ...(callsOf(fixtureEvents()).get('c1') as CallView), runs: [] })).toBe('no model runs');
  });

  test('a second of runs reads in seconds', () => {
    const c1 = callsOf(fixtureEvents()).get('c1') as CallView;
    expect(describeRuns({ ...c1, runs: [{ ...c1.runs[0], ms: 1250 }] })).toBe('1 model run · 1.25 s');
  });
});
