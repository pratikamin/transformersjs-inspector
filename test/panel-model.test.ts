import { describe, expect, test } from 'vitest';
import type { InspectorEvent, TensorSummary } from '../src/events';
import type { Change, PanelState } from '../src/panel/model';
import { createState, DEFAULT_MAX_CALLS, reduce } from '../src/panel/model';
import { fixtureEvents } from './fakes';

const reduceAll = (state: PanelState, events: InspectorEvent[]): Change[] => events.map((ev) => reduce(state, ev).change);

const sum = (id: string, name: string): TensorSummary => ({ id, name, dtype: 'float32', dims: [1, 4], location: 'cpu', size: 4, bytes: 16, head: [0, 1, 2, 3] });

const start = (callId: string, t = 0): InspectorEvent => ({ type: 'call:start', callId, label: `call ${callId}`, task: 'feature-extraction', input: { kind: 'text', text: 'hi' }, t });
const runStart = (callId: string | null, runId: string, t = 0, session = 'model'): InspectorEvent => ({ type: 'run:start', callId, runId, session, inputs: [sum(`${runId}-in`, 'input_ids')], t });
const runEnd = (callId: string | null, runId: string, t = 0, error: string | null = null, session = 'model'): InspectorEvent => ({
  type: 'run:end',
  callId,
  runId,
  session,
  outputs: [sum(`${runId}-out`, 'logits')],
  ms: 5,
  error,
  t,
});
const logits = (callId: string | null, step: number, t = 0): InspectorEvent => ({ type: 'logits', callId, step, vocab: 10, topK: [{ id: 1, token: 'a', logit: 1, prob: 0.9 }], tensorId: null, t });
const token = (callId: string | null, step: number, t = 0): InspectorEvent => ({ type: 'token', callId, step, ids: [1], text: 'a', t });
const result = (callId: string, t = 0, error: string | null = null): InspectorEvent => ({ type: 'result', callId, result: error ? undefined : 'ok', ms: 9, error, t });
const tokenize = (callId: string | null, t = 0): InspectorEvent => ({ type: 'tokenize', callId, text: 'hi', ids: [[101, 7632, 102]], tokens: [['[CLS]', 'hi', '[SEP]']], ms: 0.1, t });

describe('createState', () => {
  test('defaults to 200 calls and empty indexes', () => {
    const state = createState();
    expect(DEFAULT_MAX_CALLS).toBe(200);
    expect(state.maxCalls).toBe(200);
    expect(state.calls).toEqual([]);
    expect(state.byId.size).toBe(0);
    expect(state.byRun.size).toBe(0);
    expect(state.total).toBe(0);
  });

  test('maxCalls is at least 1', () => {
    expect(createState({ maxCalls: 0 }).maxCalls).toBe(1);
    expect(createState({ maxCalls: 7 }).maxCalls).toBe(7);
  });
});

describe('reduce(fixtureEvents())', () => {
  test('yields 2 calls with tokenize, runs, steps and result on the right call', () => {
    const state = createState();
    const events = fixtureEvents();
    const changes = reduceAll(state, events);

    expect(state.calls.map((c) => c.id)).toEqual(['c1', 'c2']);
    expect(state.total).toBe(2);
    expect(state.calls.map((c) => c.n)).toEqual([1, 2]);
    expect([...state.byId.keys()]).toEqual(['c1', 'c2']);
    expect(changes.filter((c) => c === 'new')).toHaveLength(2);
    expect(changes[0]).toBe('new');
    expect(changes[5]).toBe('new');
    expect(changes.every((c, i) => (i === 0 || i === 5 ? c === 'new' : c === 'updated'))).toBe(true);

    const [c1, c2] = state.calls;
    expect(c1).toMatchObject({ label: 'feature-extraction', task: 'feature-extraction', input: { kind: 'text', text: 'the quick brown fox.' }, startedAt: 1000, done: true, synthetic: false, ms: 14.2, error: null });
    expect(c1.tokenize).toHaveLength(1);
    expect(c1.tokenize[0].tokens[0]).toEqual(['[CLS]', 'the', 'quick', 'brown', 'fox', '.', '[SEP]']);
    expect(c1.runs).toHaveLength(1);
    expect(c1.runs[0]).toMatchObject({ runId: 'r1', session: 'model', ms: 12.7, error: null, done: true });
    expect(c1.runs[0].inputs.map((s) => s.name)).toEqual(['input_ids', 'attention_mask', 'token_type_ids']);
    expect(c1.runs[0].outputs[0]).toMatchObject({ name: 'last_hidden_state', dims: [1, 7, 384] });
    expect(c1.steps).toEqual([]);
    expect(c1.result).toMatchObject({ $tensor: { id: 't4', dims: [1, 7, 384] } });

    expect(c2).toMatchObject({ label: 'text-generation', task: 'text-generation', done: true, synthetic: false, ms: 25.3, error: null });
    expect(c2.tokenize).toHaveLength(1);
    expect(c2.runs.map((r) => r.runId)).toEqual(['r2', 'r3', 'r4']);
    expect(c2.runs.every((r) => r.done && r.ms === 6.1)).toBe(true);
    expect(c2.runs[0].inputs.map((s) => s.name)).toEqual(['input_ids', 'attention_mask', 'position_ids', 'past_key_values.0.key', 'past_key_values.0.value']);
    expect(c2.runs[2].outputs[0]).toMatchObject({ name: 'logits', dims: [1, 1, 128256] });
    expect(c2.steps.map((s) => s.step)).toEqual([0, 1, 2]);
    for (const s of c2.steps) {
      expect(s.topK).toHaveLength(5);
      expect(s.tensorId).toMatch(/^t\d+$/);
      expect(s.ids).toHaveLength(1);
      expect(typeof s.text).toBe('string');
    }
    expect(c2.steps[1].ids).toEqual([7742]);
    expect(c2.steps[1].text).toBe('quick');
    expect(c2.result).toEqual([{ generated_text: 'hi the quick brown' }]);

    expect([...state.byRun.entries()]).toEqual([
      ['r1', 'c1'],
      ['r2', 'c2'],
      ['r3', 'c2'],
      ['r4', 'c2'],
    ]);
  });

  test('every reduce returns the call the event landed on', () => {
    const state = createState();
    for (const ev of fixtureEvents()) {
      const { call } = reduce(state, ev);
      expect(call.id).toBe(ev.callId);
    }
  });

  test('does not mutate the events it consumes', () => {
    const state = createState();
    const events = fixtureEvents();
    const before = JSON.stringify(events);
    reduceAll(state, events);
    expect(JSON.stringify(events)).toBe(before);
  });
});

describe('orphan run:start', () => {
  test('opens a synthetic call labelled direct · <session>, closed by its run:end', () => {
    const state = createState();
    const a = reduce(state, runStart(null, 'r1', 10, 'decoder_model_merged'));
    expect(a.change).toBe('new');
    expect(a.call).toMatchObject({ label: 'direct · decoder_model_merged', task: null, input: null, synthetic: true, done: false, startedAt: 10, n: 1 });
    expect(a.call.runs).toHaveLength(1);
    expect(a.call.runs[0]).toMatchObject({ runId: 'r1', session: 'decoder_model_merged', done: false, ms: null });
    expect(state.byRun.get('r1')).toBe(a.call.id);
    expect(state.byId.get(a.call.id)).toBe(a.call);

    const b = reduce(state, runEnd(null, 'r1', 18, null, 'decoder_model_merged'));
    expect(b.change).toBe('updated');
    expect(b.call).toBe(a.call);
    expect(b.call.runs[0]).toMatchObject({ done: true, ms: 5, error: null });
    expect(b.call.runs[0].outputs[0].name).toBe('logits');
    expect(b.call).toMatchObject({ done: true, ms: 8, error: null });
    expect(state.calls).toHaveLength(1);
  });

  test('each orphan run:start is its own row, and a failed run marks the call errored', () => {
    const state = createState();
    reduce(state, runStart(null, 'r1', 0));
    reduce(state, runEnd(null, 'r1', 1));
    reduce(state, runStart(null, 'r2', 2));
    reduce(state, runEnd(null, 'r2', 3, 'boom'));
    expect(state.calls.map((c) => c.label)).toEqual(['direct · model', 'direct · model']);
    expect(state.calls.map((c) => c.id)).toEqual(['s1', 's2']);
    expect(state.calls[0]).toMatchObject({ done: true, error: null });
    expect(state.calls[1]).toMatchObject({ done: true, error: 'boom' });
    expect(state.calls[1].runs[0].error).toBe('boom');
    expect(state.byRun.get('r2')).toBe('s2');
  });

  test('orphan run:start while a real call is open still opens a synthetic call', () => {
    const state = createState();
    reduce(state, start('c1'));
    const { call, change } = reduce(state, runStart(null, 'r1'));
    expect(change).toBe('new');
    expect(call.synthetic).toBe(true);
    expect(state.byId.get('c1')?.runs).toHaveLength(0);
  });
});

describe('run:end matching', () => {
  test('matches its run via byRun even when its callId is null', () => {
    const state = createState();
    reduce(state, start('c1'));
    reduce(state, runStart('c1', 'r1'));
    const { call, change } = reduce(state, runEnd(null, 'r1'));
    expect(change).toBe('updated');
    expect(call.id).toBe('c1');
    expect(call.runs[0].done).toBe(true);
    expect(call.done).toBe(false); // a real call waits for result
    expect(state.calls).toHaveLength(1);
  });

  test('run:end for an unknown run records a run with no inputs', () => {
    const state = createState();
    reduce(state, start('c1'));
    const { call } = reduce(state, runEnd('c1', 'r9'));
    expect(call.id).toBe('c1');
    expect(call.runs[0]).toMatchObject({ runId: 'r9', inputs: [], done: true, ms: 5 });
    expect(state.byRun.get('r9')).toBe('c1');
  });
});

describe('orphan logits/token/tokenize', () => {
  test('attach to the newest open call', () => {
    const state = createState();
    reduce(state, start('c1', 0));
    reduce(state, start('c2', 1));
    expect(reduce(state, logits(null, 0, 2)).call.id).toBe('c2');
    expect(reduce(state, token(null, 0, 3)).call.id).toBe('c2');
    expect(reduce(state, tokenize(null, 4)).call.id).toBe('c2');
    const c2 = state.byId.get('c2')!;
    expect(c2.steps).toHaveLength(1);
    expect(c2.steps[0]).toMatchObject({ step: 0, ids: [1], text: 'a', tensorId: null });
    expect(c2.steps[0].topK).toHaveLength(1);
    expect(c2.tokenize).toHaveLength(1);
    expect(state.byId.get('c1')!.steps).toEqual([]);
    expect(state.byId.get('c1')!.tokenize).toEqual([]);

    // Once c2 is done, c1 is the newest open call.
    reduce(state, result('c2', 5));
    expect(reduce(state, token(null, 1, 6)).call.id).toBe('c1');
  });

  test('open a synthetic call when nothing is open', () => {
    const state = createState();
    reduce(state, start('c1', 0));
    reduce(state, result('c1', 1));
    const a = reduce(state, logits(null, 0, 2));
    expect(a.change).toBe('new');
    expect(a.call).toMatchObject({ label: 'direct · generate', synthetic: true, done: false });
    const b = reduce(state, token(null, 0, 3));
    expect(b.change).toBe('updated');
    expect(b.call).toBe(a.call);
    expect(a.call.steps).toHaveLength(1);
    expect(a.call.steps[0]).toMatchObject({ step: 0, ids: [1], text: 'a' });
    expect(a.call.steps[0].topK).toHaveLength(1);
  });

  test('steps are keyed by step number regardless of arrival order', () => {
    const state = createState();
    reduce(state, start('c1'));
    reduce(state, token('c1', 1));
    reduce(state, logits('c1', 0));
    reduce(state, token('c1', 0));
    reduce(state, logits('c1', 1));
    const steps = state.byId.get('c1')!.steps;
    expect(steps.map((s) => s.step)).toEqual([0, 1]);
    expect(steps.every((s) => s.topK && s.ids)).toBe(true);
  });
});

describe('result', () => {
  test('marks the call done with result, error and ms', () => {
    const state = createState();
    reduce(state, start('c1'));
    const ok = reduce(state, result('c1', 3));
    expect(ok.change).toBe('updated');
    expect(ok.call).toMatchObject({ done: true, result: 'ok', error: null, ms: 9 });

    reduce(state, start('c2'));
    const bad = reduce(state, result('c2', 4, 'failed'));
    expect(bad.call).toMatchObject({ done: true, result: undefined, error: 'failed', ms: 9 });
  });

  test('an unknown callId opens a synthetic call of that id (capped history)', () => {
    const state = createState();
    const { call, change } = reduce(state, result('c7', 1));
    expect(change).toBe('new');
    expect(call).toMatchObject({ id: 'c7', synthetic: true, done: true, result: 'ok', label: 'direct' });
    expect(state.byId.get('c7')).toBe(call);
  });

  test('a late call:start upgrades a synthetic placeholder in place', () => {
    const state = createState();
    reduce(state, runStart('c1', 'r1', 0));
    const { call, change } = reduce(state, start('c1', 0));
    expect(change).toBe('updated');
    expect(call).toMatchObject({ id: 'c1', label: 'call c1', synthetic: false });
    expect(call.runs).toHaveLength(1);
    expect(state.calls).toHaveLength(1);
  });
});

describe('maxCalls cap', () => {
  test('drops the oldest call and cleans byId and byRun', () => {
    const state = createState({ maxCalls: 3 });
    for (let i = 1; i <= 5; i++) {
      reduce(state, start(`c${i}`, i));
      reduce(state, runStart(`c${i}`, `r${i}`, i));
      reduce(state, runEnd(`c${i}`, `r${i}`, i));
      reduce(state, result(`c${i}`, i));
    }
    expect(state.calls.map((c) => c.id)).toEqual(['c3', 'c4', 'c5']);
    expect(state.calls.map((c) => c.n)).toEqual([3, 4, 5]);
    expect(state.total).toBe(5);
    expect([...state.byId.keys()]).toEqual(['c3', 'c4', 'c5']);
    expect([...state.byRun.keys()]).toEqual(['r3', 'r4', 'r5']);
  });

  test('applies to synthetic calls too', () => {
    const state = createState({ maxCalls: 2 });
    for (let i = 1; i <= 4; i++) {
      reduce(state, runStart(null, `r${i}`, i));
      reduce(state, runEnd(null, `r${i}`, i));
    }
    expect(state.calls).toHaveLength(2);
    expect(state.calls.map((c) => c.n)).toEqual([3, 4]);
    expect([...state.byRun.keys()]).toEqual(['r3', 'r4']);
    expect(state.byRun.size).toBe(2);
  });

  test('events for a dropped call reopen it as synthetic rather than throwing', () => {
    const state = createState({ maxCalls: 1 });
    reduce(state, start('c1'));
    reduce(state, start('c2'));
    expect(state.byId.has('c1')).toBe(false);
    const { call, change } = reduce(state, result('c1'));
    expect(change).toBe('new');
    expect(call).toMatchObject({ id: 'c1', synthetic: true, done: true });
    expect(state.calls.map((c) => c.id)).toEqual(['c1']);
  });
});
