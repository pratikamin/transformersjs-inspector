/**
 * Offline fakes shaped like the real Transformers.js 4.2 / onnxruntime-web objects,
 * with the tensor shapes recorded in docs/01-research.md:
 *   encoder: input_ids/attention_mask/token_type_ids int64 [1,7] -> last_hidden_state float32 [1,7,384]
 *   decoder: input_ids [1,1], attention_mask [1,N], position_ids [1,1], past_key_values.0.{key,value}
 *            float32 [1,2,N,16] (gpu-buffer) -> logits float32 [1,1,128256], present.0.{key,value} [1,2,N+1,16]
 */
import type { InspectorEvent, TensorSummary, TopKEntry } from '../src/events';
import type { PipelineLike, SessionLike, StreamerLike, TensorLike, TokenizerLike } from '../src/types';

// ---- tensors ---------------------------------------------------------------

export interface FakeTensor extends TensorLike {
  location: string;
  size: number;
  /** How many times the `data` getter was read (successfully or not). */
  dataReads: number;
  disposed: boolean;
  getData(): Promise<ArrayLike<unknown>>;
  dispose(): void;
}

export const CPU_LOCATIONS: readonly string[] = ['cpu', 'cpu-pinned'];

function defaultData(type: string, size: number): ArrayLike<unknown> {
  switch (type) {
    case 'float32':
      return Float32Array.from({ length: size }, (_, i) => ((i * 7919) % 1000) / 500 - 1);
    case 'float64':
      return Float64Array.from({ length: size }, (_, i) => ((i * 7919) % 1000) / 500 - 1);
    case 'float16':
      return Uint16Array.from({ length: size }, (_, i) => 0x3c00 + (i % 64));
    case 'int64':
      return BigInt64Array.from({ length: size }, (_, i) => BigInt(i));
    case 'uint64':
      return BigUint64Array.from({ length: size }, (_, i) => BigInt(i));
    case 'int32':
      return Int32Array.from({ length: size }, (_, i) => i);
    case 'uint32':
      return Uint32Array.from({ length: size }, (_, i) => i);
    case 'int16':
      return Int16Array.from({ length: size }, (_, i) => i);
    case 'uint16':
      return Uint16Array.from({ length: size }, (_, i) => i);
    case 'int8':
      return Int8Array.from({ length: size }, (_, i) => i % 128);
    case 'uint8':
    case 'bool':
      return Uint8Array.from({ length: size }, (_, i) => i % 2);
    case 'string':
      return Array.from({ length: size }, (_, i) => `s${i}`);
    default:
      throw new Error(`fakeTensor: no default data for dtype '${type}'`);
  }
}

/** Accepts plain numbers for int64 tensors and converts them to bigint like ORT would. */
function coerce(type: string, data: ArrayLike<unknown>): ArrayLike<unknown> {
  if ((type === 'int64' || type === 'uint64') && !(data instanceof BigInt64Array) && !(data instanceof BigUint64Array)) {
    const C = type === 'int64' ? BigInt64Array : BigUint64Array;
    return C.from(Array.from(data as ArrayLike<number | bigint>, (v) => BigInt(v)));
  }
  return data;
}

export function fakeTensor(opts: { type: string; dims: readonly number[]; data?: ArrayLike<unknown>; location?: string }): FakeTensor {
  const dims = [...opts.dims];
  const size = dims.reduce((a, b) => a * b, 1);
  const location = opts.location ?? 'cpu';
  const backing = opts.data === undefined ? defaultData(opts.type, size) : coerce(opts.type, opts.data);
  if (backing.length !== size) throw new Error(`fakeTensor: data length ${backing.length} != dims product ${size}`);
  const t: FakeTensor = {
    type: opts.type,
    dims,
    location,
    size,
    dataReads: 0,
    disposed: false,
    get data(): ArrayLike<unknown> {
      t.dataReads++;
      if (t.disposed) throw new Error('The tensor has been disposed.');
      if (!CPU_LOCATIONS.includes(location)) {
        throw new Error(`The data is not on CPU. Use \`getData()\` to download GPU data to CPU, or use \`texture\` or \`gpuBuffer\` property to access the GPU data directly. (location: ${location})`);
      }
      return backing;
    },
    async getData() {
      if (t.disposed) throw new Error('The tensor has been disposed.');
      return backing;
    },
    dispose() {
      t.disposed = true;
    },
  };
  return t;
}

// ---- sessions --------------------------------------------------------------

export interface FakeSession extends SessionLike {
  name: string;
  /** Every feeds object passed to `run`, in order. */
  runs: Record<string, TensorLike>[];
}

export function fakeSession(opts: {
  name: string;
  inputNames: readonly string[];
  outputNames: readonly string[];
  produce: (feeds: Record<string, TensorLike>) => Record<string, TensorLike> | Promise<Record<string, TensorLike>>;
}): FakeSession {
  const session: FakeSession = {
    name: opts.name,
    inputNames: [...opts.inputNames],
    outputNames: [...opts.outputNames],
    runs: [],
    async run(feeds) {
      for (const n of session.inputNames) {
        if (!(n in feeds)) throw new Error(`input '${n}' is missing in 'feeds'.`);
      }
      session.runs.push(feeds);
      return opts.produce(feeds);
    },
  };
  return session;
}

/** Encoder session with the all-MiniLM-L6-v2 shapes: [1,n] int64 feeds -> last_hidden_state [1,n,384]. */
export function fakeEncoderSession(): FakeSession {
  return fakeSession({
    name: 'model',
    inputNames: ['input_ids', 'attention_mask', 'token_type_ids'],
    outputNames: ['last_hidden_state'],
    produce: (feeds) => {
      const [b, n] = feeds.input_ids.dims;
      return { last_hidden_state: fakeTensor({ type: 'float32', dims: [b, n, 384] }) };
    },
  });
}

// ---- tokenizer -------------------------------------------------------------

const SPECIAL = { '[PAD]': 0, '[UNK]': 100, '[CLS]': 101, '[SEP]': 102 } as const;
const SEED_VOCAB: Record<string, number> = { the: 1996, quick: 7742, brown: 5927, fox: 2673, '.': 1012, hi: 7632, hello: 7592, world: 2088 };

export interface FakeTokenizer extends TokenizerLike {
  _tokenizer: { id_to_token(id: number): string | undefined; token_to_id(token: string): number | undefined };
  decode(ids: number[], opts?: unknown): string;
  encode(text: string): number[];
  calls: unknown[];
}

/** WordPiece-flavoured BERT-style tokenizer: [CLS] words… [SEP]; unknown words get fresh ids. */
export function fakeTokenizer(): FakeTokenizer {
  const vocab = new Map<string, number>(Object.entries({ ...SPECIAL, ...SEED_VOCAB }));
  const inverse = new Map<number, string>([...vocab].map(([k, v]) => [v, k]));
  let nextId = 30000;
  const idOf = (word: string): number => {
    let id = vocab.get(word);
    if (id === undefined) {
      id = nextId++;
      vocab.set(word, id);
      inverse.set(id, word);
    }
    return id;
  };
  const encode = (text: string): number[] => {
    const words = text.toLowerCase().match(/[a-z0-9]+|[^\sa-z0-9]/g) ?? [];
    return [SPECIAL['[CLS]'], ...words.map(idOf), SPECIAL['[SEP]']];
  };
  const tok: FakeTokenizer = {
    calls: [],
    _tokenizer: {
      id_to_token: (id) => inverse.get(id),
      token_to_id: (token) => vocab.get(token),
    },
    encode,
    decode(ids) {
      return ids
        .map((id) => inverse.get(id) ?? '[UNK]')
        .filter((t) => !(t in SPECIAL))
        .join(' ');
    },
    _call(text, opts) {
      tok.calls.push({ text, opts });
      const texts = Array.isArray(text) ? (text as string[]) : [String(text)];
      const rows = texts.map(encode);
      const n = Math.max(...rows.map((r) => r.length));
      const ids: number[] = [];
      const mask: number[] = [];
      for (const r of rows) {
        ids.push(...r, ...new Array<number>(n - r.length).fill(SPECIAL['[PAD]']));
        mask.push(...r.map(() => 1), ...new Array<number>(n - r.length).fill(0));
      }
      const dims = [rows.length, n];
      return {
        input_ids: fakeTensor({ type: 'int64', dims, data: ids }),
        attention_mask: fakeTensor({ type: 'int64', dims, data: mask }),
        token_type_ids: fakeTensor({ type: 'int64', dims, data: ids.map(() => 0) }),
      };
    },
  };
  return tok;
}

// ---- generative model ------------------------------------------------------

export const FAKE_VOCAB_SIZE = 128256;
/** Ids the fake decoder "samples", in order, cycling; all are in the fake tokenizer's seed vocab. */
export const FAKE_PICKS: readonly number[] = [1996, 7742, 5927, 2673, 1012];

export interface FakeGenerativeModel {
  sessions: { model: FakeSession };
  config: Record<string, unknown>;
  generate(opts: Record<string, unknown>): Promise<FakeTensor>;
  /** Number of `generate` calls so far. */
  generations: number;
}

type ProcessorCallable = ((input_ids: unknown, logits: TensorLike) => TensorLike) | { _call(input_ids: unknown, logits: TensorLike): TensorLike };

function toProcessorList(x: unknown): ProcessorCallable[] {
  if (!x) return [];
  if (Array.isArray(x)) return x as ProcessorCallable[];
  const inner = (x as { processors?: unknown }).processors;
  if (Array.isArray(inner)) return inner as ProcessorCallable[];
  if (typeof x === 'function' || typeof (x as { _call?: unknown })._call === 'function') return [x as ProcessorCallable];
  return [];
}

function callProcessor(p: ProcessorCallable, ids: unknown, logits: TensorLike): TensorLike {
  return typeof p === 'function' ? p(ids, logits) : p._call(ids, logits);
}

function argmax(data: ArrayLike<unknown>): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = Number(data[i]);
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}

/** Decoder-only model with the tiny-random-Llama shapes; each step is one `sessions.model.run`. */
export function fakeGenerativeModel(opts: { vocab?: number } = {}): FakeGenerativeModel {
  const V = opts.vocab ?? FAKE_VOCAB_SIZE;
  let currentStep = 0; // set by generate() before each run so the session's argmax is FAKE_PICKS[step]
  const session = fakeSession({
    name: 'model',
    inputNames: ['input_ids', 'attention_mask', 'position_ids', 'past_key_values.0.key', 'past_key_values.0.value'],
    outputNames: ['logits', 'present.0.key', 'present.0.value'],
    produce: (feeds) => {
      const total = feeds.attention_mask.dims[1]; // sequence length including this step's tokens
      const pick = FAKE_PICKS[currentStep % FAKE_PICKS.length];
      const logits = new Float32Array(V);
      for (let i = 0; i < V; i++) logits[i] = -((i * 13) % 11) - 1;
      logits[pick] = 8;
      return {
        logits: fakeTensor({ type: 'float32', dims: [1, 1, V], data: logits }),
        'present.0.key': fakeTensor({ type: 'float32', dims: [1, 2, total, 16], location: 'gpu-buffer' }),
        'present.0.value': fakeTensor({ type: 'float32', dims: [1, 2, total, 16], location: 'gpu-buffer' }),
      };
    },
  });
  const model: FakeGenerativeModel = {
    sessions: { model: session },
    config: { model_type: 'llama', vocab_size: V },
    generations: 0,
    async generate(o) {
      model.generations++;
      const inputIds = o.input_ids as TensorLike;
      const prompt = Array.from(inputIds.data as ArrayLike<number | bigint>, Number);
      const maxNew = typeof o.max_new_tokens === 'number' ? o.max_new_tokens : 3;
      const processors = toProcessorList(o.logits_processor);
      const streamer = o.streamer as StreamerLike | undefined;
      const all: bigint[][] = [prompt.map(BigInt)];
      streamer?.put(all);
      for (let step = 0; step < maxNew; step++) {
        const seq = all[0];
        const past = step === 0 ? 0 : seq.length - 1;
        const cur = step === 0 ? seq : seq.slice(-1);
        const feeds: Record<string, TensorLike> = {
          input_ids: fakeTensor({ type: 'int64', dims: [1, cur.length], data: cur }),
          attention_mask: fakeTensor({ type: 'int64', dims: [1, seq.length], data: seq.map(() => 1) }),
          position_ids: fakeTensor({ type: 'int64', dims: [1, cur.length], data: cur.map((_, i) => past + i) }),
          'past_key_values.0.key': fakeTensor({ type: 'float32', dims: [1, 2, past, 16], location: 'gpu-buffer' }),
          'past_key_values.0.value': fakeTensor({ type: 'float32', dims: [1, 2, past, 16], location: 'gpu-buffer' }),
        };
        currentStep = step;
        const out = await model.sessions.model.run(feeds);
        // outputs.logits.slice(null, -1, null): last position, [1, V]
        let logits: TensorLike = fakeTensor({ type: 'float32', dims: [1, V], data: out.logits.data });
        for (const p of processors) logits = callProcessor(p, all, logits);
        const next = argmax(logits.data);
        seq.push(BigInt(next));
        streamer?.put([[BigInt(next)]]);
      }
      streamer?.end();
      return fakeTensor({ type: 'int64', dims: [1, all[0].length], data: all[0] });
    },
  };
  return model;
}

// ---- pipeline --------------------------------------------------------------

export type FakePipelineTask = 'feature-extraction' | 'text-generation';

export interface FakePipeline extends PipelineLike {
  (...args: unknown[]): Promise<unknown>;
  task: FakePipelineTask;
  tokenizer: FakeTokenizer;
  model: { sessions: Record<string, FakeSession>; config: Record<string, unknown>; generate?: FakeGenerativeModel['generate'] };
  _call(...args: unknown[]): Promise<unknown>;
}

/**
 * Callable-shaped pipeline: `pipe(text)` forwards to `pipe._call(text)`, which runs
 * tokenizer -> session(s) -> result, like the real `Pipeline` subclasses.
 */
export function fakePipeline(opts: { task?: FakePipelineTask } = {}): FakePipeline {
  const task = opts.task ?? 'feature-extraction';
  const tokenizer = fakeTokenizer();
  const pipe = ((...args: unknown[]) => pipe._call(...args)) as FakePipeline;
  pipe.task = task;
  pipe.tokenizer = tokenizer;

  if (task === 'text-generation') {
    const gen = fakeGenerativeModel();
    pipe.model = { sessions: gen.sessions, config: gen.config, generate: (o) => gen.generate(o) };
    pipe._call = async (text: unknown, callOpts?: unknown) => {
      const enc = pipe.tokenizer._call(text);
      // through `pipe.model.generate`, as the real `TextGenerationPipeline._call` does (`this.model.generate`)
      const out = await pipe.model.generate!({ ...enc, ...((callOpts as Record<string, unknown>) ?? {}) });
      const ids = Array.from(out.data as ArrayLike<bigint>, Number);
      return [{ generated_text: pipe.tokenizer.decode(ids) }];
    };
  } else {
    const session = fakeEncoderSession();
    pipe.model = { sessions: { model: session }, config: { model_type: 'bert' } };
    pipe._call = async (text: unknown) => {
      const enc = pipe.tokenizer._call(text);
      const feeds: Record<string, TensorLike> = {};
      for (const n of session.inputNames) feeds[n] = enc[n] as TensorLike;
      const out = await session.run(feeds);
      return out.last_hidden_state;
    };
  }
  return pipe;
}

// ---- fixture events --------------------------------------------------------

function summary(id: string, name: string, dtype: string, dims: number[], location = 'cpu', head: (number | string)[] | null = []): TensorSummary {
  const size = dims.reduce((a, b) => a * b, 1);
  const bpe = dtype === 'int64' ? 8 : dtype === 'float32' ? 4 : 0;
  return { id, name, dtype, dims, location, size, bytes: size * bpe, head };
}

const F32_HEAD = [-0.0347, 0.0521, -0.0128, 0.0873, 0.0044, -0.0619, 0.0211, 0.0985];
const kvHead = (): null => null;

function topK(step: number): TopKEntry[] {
  const picks = [1996, 7742, 5927];
  const tokens = ['the', 'quick', 'brown'];
  const chosen = step % 3;
  const entries: TopKEntry[] = [{ id: picks[chosen], token: tokens[chosen], logit: 8, prob: 0.9971 }];
  const others = [
    [0, '[PAD]'],
    [102, '[SEP]'],
    [7632, 'hi'],
    [2088, 'world'],
  ] as const;
  for (let i = 0; i < 4; i++) entries.push({ id: others[i][0], token: others[i][1], logit: -1 - i, prob: 0.0012 / (i + 1) });
  return entries;
}

/**
 * One embedding call (c1: one encoder run) and one 3-step generation call
 * (c2: three decoder runs with logits + token events), with the researched shapes.
 * Every entry is plain JSON: numbers, strings, null, arrays, objects.
 */
export function fixtureEvents(): InspectorEvent[] {
  const embedText = 'the quick brown fox.';
  const embedIds = [101, 1996, 7742, 5927, 2673, 1012, 102];
  const embedTokens = ['[CLS]', 'the', 'quick', 'brown', 'fox', '.', '[SEP]'];
  const events: InspectorEvent[] = [
    { type: 'call:start', callId: 'c1', label: 'feature-extraction', task: 'feature-extraction', input: { kind: 'text', text: embedText }, t: 1000 },
    { type: 'tokenize', callId: 'c1', text: embedText, ids: [embedIds], tokens: [embedTokens], ms: 0.4, t: 1000.5 },
    {
      type: 'run:start',
      callId: 'c1',
      runId: 'r1',
      session: 'model',
      inputs: [
        summary('t1', 'input_ids', 'int64', [1, 7], 'cpu', embedIds),
        summary('t2', 'attention_mask', 'int64', [1, 7], 'cpu', [1, 1, 1, 1, 1, 1, 1]),
        summary('t3', 'token_type_ids', 'int64', [1, 7], 'cpu', [0, 0, 0, 0, 0, 0, 0]),
      ],
      t: 1001,
    },
    {
      type: 'run:end',
      callId: 'c1',
      runId: 'r1',
      session: 'model',
      outputs: [summary('t4', 'last_hidden_state', 'float32', [1, 7, 384], 'cpu', F32_HEAD)],
      ms: 12.7,
      error: null,
      t: 1013.7,
    },
    { type: 'result', callId: 'c1', result: { $tensor: summary('t4', 'last_hidden_state', 'float32', [1, 7, 384], 'cpu', F32_HEAD) }, ms: 14.2, error: null, t: 1014.2 },
  ];

  const genText = 'hi';
  const promptIds = [101, 7632, 102];
  const promptTokens = ['[CLS]', 'hi', '[SEP]'];
  const picks = [1996, 7742, 5927];
  const pickTokens = ['the', 'quick', 'brown'];
  let tid = 5;
  let t = 2000;
  events.push(
    { type: 'call:start', callId: 'c2', label: 'text-generation', task: 'text-generation', input: { kind: 'text', text: genText }, t },
    { type: 'tokenize', callId: 'c2', text: genText, ids: [promptIds], tokens: [promptTokens], ms: 0.3, t: (t += 0.3) },
  );
  for (let step = 0; step < 3; step++) {
    const N = promptIds.length + step; // sequence length before this step
    const runId = `r${2 + step}`;
    const cur = step === 0 ? promptIds : [picks[step - 1]];
    const past = step === 0 ? 0 : N - 1;
    events.push({
      type: 'run:start',
      callId: 'c2',
      runId,
      session: 'model',
      inputs: [
        summary(`t${tid++}`, 'input_ids', 'int64', [1, cur.length], 'cpu', cur),
        summary(`t${tid++}`, 'attention_mask', 'int64', [1, N], 'cpu', new Array<number>(N).fill(1)),
        summary(`t${tid++}`, 'position_ids', 'int64', [1, cur.length], 'cpu', cur.map((_, i) => past + i)),
        summary(`t${tid++}`, 'past_key_values.0.key', 'float32', [1, 2, past, 16], 'gpu-buffer', kvHead()),
        summary(`t${tid++}`, 'past_key_values.0.value', 'float32', [1, 2, past, 16], 'gpu-buffer', kvHead()),
      ],
      t: (t += 1),
    });
    const logitsId = `t${tid++}`;
    events.push({
      type: 'run:end',
      callId: 'c2',
      runId,
      session: 'model',
      outputs: [
        summary(logitsId, 'logits', 'float32', [1, 1, 128256], 'cpu', F32_HEAD),
        summary(`t${tid++}`, 'present.0.key', 'float32', [1, 2, N, 16], 'gpu-buffer', kvHead()),
        summary(`t${tid++}`, 'present.0.value', 'float32', [1, 2, N, 16], 'gpu-buffer', kvHead()),
      ],
      ms: 6.1,
      error: null,
      t: (t += 6.1),
    });
    events.push(
      { type: 'logits', callId: 'c2', step, vocab: 128256, topK: topK(step), tensorId: logitsId, t: (t += 0.2) },
      { type: 'token', callId: 'c2', step, ids: [picks[step]], text: pickTokens[step], t: (t += 0.1) },
    );
  }
  events.push({ type: 'result', callId: 'c2', result: [{ generated_text: 'hi the quick brown' }], ms: 25.3, error: null, t: t + 1 });
  return events;
}
