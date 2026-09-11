/**
 * Panel state reducer. Pure data, no DOM: the panel (and its node tests) feed
 * `InspectorEvent`s through `reduce` and render the resulting `CallView`s.
 *
 * `reduce` mutates `state` in place and reports which call the event landed on and
 * whether that call is `'new'` (append a row) or `'updated'` (patch its row).
 *
 * Attachment rules:
 * - `call:start` opens a call. `result` closes it (`done`, `result`/`error`/`ms`).
 * - `run:start` with `callId: null` opens a synthetic call labelled `direct · <session>`
 *   (the preload path, which never sees a pipeline call). `run:end` finds its run via
 *   `byRun` regardless of its own `callId`; a synthetic call is closed by the `run:end`
 *   of the run that opened it.
 * - `tokenize`, `logits` and `token` with `callId: null` attach to the newest open call,
 *   opening a synthetic one when none is open.
 * - An event naming a `callId`/`runId` the state has never seen (history was capped)
 *   gets a synthetic call/run of that id so later events still land together.
 * - At most `maxCalls` calls are kept; the oldest is dropped and its `byId`/`byRun`
 *   entries removed.
 */
import type { InputPreview, InspectorEvent, TensorSummary, TopKEntry } from '../events';

export const DEFAULT_MAX_CALLS = 200;

export type TokenizeEvent = Extract<InspectorEvent, { type: 'tokenize' }>;

export interface RunView {
  runId: string;
  session: string;
  inputs: TensorSummary[];
  outputs: TensorSummary[];
  ms: number | null;
  error: string | null;
  done: boolean;
}

export interface StepView {
  step: number;
  topK?: TopKEntry[];
  tensorId?: string | null;
  ids?: number[];
  text?: string | null;
}

export interface CallView {
  id: string;
  /** 1-based sequence number across the whole session (survives the `maxCalls` cap). */
  n: number;
  label: string;
  task: string | null;
  /** `null` for synthetic calls, which have no pipeline input. */
  input: InputPreview | null;
  tokenize: TokenizeEvent[];
  runs: RunView[];
  steps: StepView[];
  result: unknown;
  error: string | null;
  ms: number | null;
  startedAt: number;
  done: boolean;
  /** Opened by the reducer for events that named no call (or an unknown one). */
  synthetic: boolean;
}

export interface PanelState {
  /** Oldest first. */
  calls: CallView[];
  byId: Map<string, CallView>;
  /** runId -> callId */
  byRun: Map<string, string>;
  /** Calls ever opened, including ones dropped by the cap. */
  total: number;
  readonly maxCalls: number;
}

export type Change = 'new' | 'updated';

export function createState(opts: { maxCalls?: number } = {}): PanelState {
  return {
    calls: [],
    byId: new Map(),
    byRun: new Map(),
    total: 0,
    maxCalls: Math.max(1, Math.floor(opts.maxCalls ?? DEFAULT_MAX_CALLS)),
  };
}

type CallInit = { id?: string; label: string; task: string | null; input: InputPreview | null; t: number; synthetic: boolean };

function openCall(state: PanelState, init: CallInit): CallView {
  const n = ++state.total;
  // Synthetic ids use a prefix no wrapper emits (`c`/`r`/`t` are taken), so they never collide.
  const id = init.id ?? `s${n}`;
  const call: CallView = {
    id,
    n,
    label: init.label,
    task: init.task,
    input: init.input,
    tokenize: [],
    runs: [],
    steps: [],
    result: undefined,
    error: null,
    ms: null,
    startedAt: init.t,
    done: false,
    synthetic: init.synthetic,
  };
  state.calls.push(call);
  state.byId.set(id, call);
  while (state.calls.length > state.maxCalls) {
    const dropped = state.calls.shift();
    if (!dropped) break;
    state.byId.delete(dropped.id);
    for (const run of dropped.runs) state.byRun.delete(run.runId);
  }
  return call;
}

function newestOpen(state: PanelState): CallView | undefined {
  for (let i = state.calls.length - 1; i >= 0; i--) {
    if (!state.calls[i].done) return state.calls[i];
  }
  return undefined;
}

/**
 * The call an event with `callId` belongs to. A named but unknown id gets a synthetic
 * call of that id; `null` goes to the newest open call, or a fresh synthetic one.
 */
function resolveCall(state: PanelState, callId: string | null, label: string, t: number): { call: CallView; created: boolean } {
  if (callId !== null) {
    const known = state.byId.get(callId);
    if (known) return { call: known, created: false };
    return { call: openCall(state, { id: callId, label, task: null, input: null, t, synthetic: true }), created: true };
  }
  const open = newestOpen(state);
  if (open) return { call: open, created: false };
  return { call: openCall(state, { label, task: null, input: null, t, synthetic: true }), created: true };
}

function stepOf(call: CallView, step: number): StepView {
  let view = call.steps.find((s) => s.step === step);
  if (!view) {
    view = { step };
    call.steps.push(view);
    call.steps.sort((a, b) => a.step - b.step);
  }
  return view;
}

const changeOf = (created: boolean): Change => (created ? 'new' : 'updated');

export function reduce(state: PanelState, ev: InspectorEvent): { call: CallView; change: Change } {
  switch (ev.type) {
    case 'call:start': {
      const existing = state.byId.get(ev.callId);
      if (existing) {
        // A synthetic placeholder (or a replayed start): adopt the real metadata.
        existing.label = ev.label;
        existing.task = ev.task;
        existing.input = ev.input;
        existing.startedAt = ev.t;
        existing.synthetic = false;
        return { call: existing, change: 'updated' };
      }
      const call = openCall(state, { id: ev.callId, label: ev.label, task: ev.task, input: ev.input, t: ev.t, synthetic: false });
      return { call, change: 'new' };
    }

    case 'tokenize': {
      const { call, created } = resolveCall(state, ev.callId, 'direct · tokenizer', ev.t);
      call.tokenize.push(ev);
      return { call, change: changeOf(created) };
    }

    case 'run:start': {
      const label = `direct · ${ev.session}`;
      const { call, created } =
        ev.callId === null
          ? { call: openCall(state, { label, task: null, input: null, t: ev.t, synthetic: true }), created: true }
          : resolveCall(state, ev.callId, label, ev.t);
      const run = call.runs.find((r) => r.runId === ev.runId);
      if (run) {
        run.session = ev.session;
        run.inputs = ev.inputs;
      } else {
        call.runs.push({ runId: ev.runId, session: ev.session, inputs: ev.inputs, outputs: [], ms: null, error: null, done: false });
      }
      state.byRun.set(ev.runId, call.id);
      return { call, change: changeOf(created) };
    }

    case 'run:end': {
      const ownerId = state.byRun.get(ev.runId) ?? ev.callId;
      const { call, created } = resolveCall(state, ownerId, `direct · ${ev.session}`, ev.t);
      let run = call.runs.find((r) => r.runId === ev.runId);
      if (!run) {
        run = { runId: ev.runId, session: ev.session, inputs: [], outputs: [], ms: null, error: null, done: false };
        call.runs.push(run);
        state.byRun.set(ev.runId, call.id);
      }
      run.outputs = ev.outputs;
      run.ms = ev.ms;
      run.error = ev.error;
      run.done = true;
      if (call.synthetic) {
        // No `result` will ever close a synthetic call; the run that opened it does.
        call.done = true;
        call.ms = Math.max(0, ev.t - call.startedAt);
        if (ev.error !== null) call.error = ev.error;
      }
      return { call, change: changeOf(created) };
    }

    case 'logits': {
      const { call, created } = resolveCall(state, ev.callId, 'direct · generate', ev.t);
      const step = stepOf(call, ev.step);
      step.topK = ev.topK;
      step.tensorId = ev.tensorId;
      return { call, change: changeOf(created) };
    }

    case 'token': {
      const { call, created } = resolveCall(state, ev.callId, 'direct · generate', ev.t);
      const step = stepOf(call, ev.step);
      step.ids = ev.ids;
      step.text = ev.text;
      return { call, change: changeOf(created) };
    }

    case 'result': {
      const { call, created } = resolveCall(state, ev.callId, 'direct', ev.t);
      call.result = ev.result;
      call.error = ev.error;
      call.ms = ev.ms;
      call.done = true;
      return { call, change: changeOf(created) };
    }
  }
}
