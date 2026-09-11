# Plan: transformersjs-inspector

> **The agent writes this from `00-brief.md` + `01-research.md`; you review it hard.**
>
> This is the highest-leverage document in the process. One bad line here becomes
> hundreds of bad lines of code. Reviewing this properly is strictly cheaper than
> reviewing the diff it produces.
>
> Rule of thumb: if you cannot predict what the diff will look like from reading
> this plan, the plan is not specific enough. Send it back.

## Approach

Ship a zero-runtime-dependency TypeScript library whose headline API is `attach(pipe, opts?)`:
it wraps, on the *instances* the host already holds, `pipe.model.sessions[name].run`
(named input/output tensors), `pipe.tokenizer._call` (text → ids → token strings),
`pipe.model.generate` (injects a duck-typed logits processor and streamer for per-step top-k
and token ids), and `pipe._call` (one row per pipeline call: raw input, decoded result,
wall time). Every wrapper emits structured-clone-safe JSON events onto one
transport-agnostic `InspectorBus`; tensors are summarised eagerly as
`{name, dtype, dims, location, size, bytes, head[8]}` and the full data is fetched lazily by
id through a request/response on the same bus, so GPU-resident tensors are never read back
unless a user clicks. A vanilla-DOM panel in a shadow root (constructed stylesheet, no
inline `style=`) subscribes to the bus and renders one expandable row per call. A secondary
`preload` entry point implements the zero-touch path from the research (`globalThis[Symbol.for('onnxruntime')]`)
and is documented as requiring `device: 'auto'`; a `worker` entry point relays the same bus
over `postMessage`.

Rejected alternatives, and why each loses:

- **Preload-only zero-touch instead of `attach()`.** The research proved the ORT injection
  hook works but breaks default device selection (`Unsupported device: "wasm"`) unless the
  host passes `device:'auto'`, needs a second ORT download version-locked to the one
  Transformers.js pins, must run as a separate earlier `<script>`, and cannot see the
  tokenizer, the pipeline result, or per-step logits (no pipeline reference). `attach()` sees
  all of it, has no ORT coupling, and always works. Preload ships as the secondary entry.
- **A UI framework (Preact/Lit) for the panel.** Forbidden by the brief, and it would be the
  library's only runtime dependency, add a second framework runtime to the host page, and
  make CSS isolation harder. Vanilla DOM in a shadow root with a `h()` helper is ~600 lines
  and collides with nothing.
- **Eager tensor readback at capture time.** Copying every tensor on the hot path means a
  512 KB copy per generation step for logits, a GPU→CPU sync per KV-cache tensor on WebGPU,
  and unbounded memory. That alone would blow the brief's ~5 % timing budget. Lazy readback
  reads 8 values eagerly and the rest only when the user expands a tensor, from a
  byte-budgeted store.
- **Importing `LogitsProcessor`/`TextStreamer` classes from `@huggingface/transformers` to
  subclass them.** That would make the peer a runtime dependency and, for hosts that load
  Transformers.js from a CDN (the verified spike path), pull a second copy of the library.
  `LogitsProcessorList._call` just invokes each processor as `processor(input_ids, logits)`
  and `generate` only calls `streamer.put()`/`streamer.end()`, so plain callables satisfy both
  contracts. The e2e generation test is the proof; if it fails, the fallback is the
  `opts.transformers` module-namespace escape hatch described in Phase 4.
- **Reading logits at the session boundary instead of via a logits processor.** The session
  output is the raw model output; the processor sees what the sampler sees (after repetition
  penalty etc.). The panel shows the latter because it answers "why this token". Raw logits
  are still visible on the session run row.
- **Bundling Transformers.js into the demo via npm/Vite instead of the CDN import.** The
  spikes verified the CDN path; npm-through-Vite has known ORT/WASM pre-bundling friction and
  is not what most host pages do. The npm package is a dev dependency for *types* and for the
  ORT version pin only.
- **`WeakRef` tensor store instead of a strong-ref LRU.** WeakRefs would make "expand this
  row later" fail nondeterministically. A strong-ref store with a 64 MiB byte budget (LRU,
  evicted tensors report `evicted`) is deterministic and bounded.

Conventions that hold for the whole plan:

- `src/` never imports `@huggingface/transformers` or `onnxruntime-web` at runtime except
  `src/preload.ts` (which imports `onnxruntime-web/webgpu`, rewritten to the CDN URL at build).
  All library shapes are structural types in `src/types.ts`.
- Every emitted event must survive `structuredClone` and `JSON.stringify` (no bigint, no
  typed arrays, no DOM nodes). Request *responses* may carry typed arrays.
- `npm test` (Vitest) never touches the network. Browser behaviour is `npm run e2e`
  (Playwright, Chromium) against the Vite demo server using
  `Xenova/all-MiniLM-L6-v2` (encoder) and `onnx-community/tiny-random-LlamaForCausalLM-ONNX`
  (decoder, 41 MB); a persistent Chromium profile under `.cache/pw-profile` keeps the HF Hub
  and CDN downloads cached between runs.
- Node ≥ 20. Dev dependencies (10): `typescript`, `@types/node`, `vite`, `vitest`,
  `happy-dom`, `eslint`, `@eslint/js`, `typescript-eslint`, `@playwright/test`,
  `@huggingface/transformers@4.2.0` (types + ORT pin; transitively provides `onnxruntime-web`).
  Optional peer: `@huggingface/transformers >=4 <5`. Runtime dependencies: none.

### Event schema (`src/events.ts`)

```ts
export type TensorSummary = {
  id: string;            // handle for lazy readback via bus.request('tensor', { id })
  name: string;          // feed / output name, e.g. 'input_ids', 'logits', 'present.0.key'
  dtype: string;         // ORT type string: 'float32' | 'int64' | 'bool' | ...
  dims: number[];
  location: string;      // 'cpu' | 'cpu-pinned' | 'gpu-buffer' | 'ml-tensor' | ...
  size: number;          // element count
  bytes: number;         // size * bytesPerElement(dtype); 0 for 'string'
  head: (number | string)[] | null;   // first `head` values; null when not CPU-resident
};

export type InputPreview =
  | { kind: 'text'; text: string }                       // truncated to 2000 chars
  | { kind: 'texts'; texts: string[] }
  | { kind: 'image'; width?: number; height?: number; channels?: number; src?: string }
  | { kind: 'audio'; samples: number; sampleRate?: number }
  | { kind: 'other'; json: unknown };                    // toCloneSafe() of the value

export type TopKEntry = { id: number; token: string | null; logit: number; prob: number };

export type InspectorEvent =
  | { type: 'call:start'; callId: string; label: string; task: string | null; input: InputPreview; t: number }
  | { type: 'tokenize';   callId: string | null; text: string | string[]; ids: number[][]; tokens: (string | null)[][]; ms: number; t: number }
  | { type: 'run:start';  callId: string | null; runId: string; session: string; inputs: TensorSummary[]; t: number }
  | { type: 'run:end';    callId: string | null; runId: string; session: string; outputs: TensorSummary[]; ms: number; error: string | null; t: number }
  | { type: 'logits';     callId: string | null; step: number; vocab: number; topK: TopKEntry[]; tensorId: string | null; t: number }
  | { type: 'token';      callId: string | null; step: number; ids: number[]; text: string | null; t: number }
  | { type: 'result';     callId: string; result: unknown; ms: number; error: string | null; t: number };

// Request/response (not events); responses may carry typed arrays.
export interface RequestMap {
  tensor: { req: { id: string }; res: TensorData };
}
export type TensorData =
  | { id: string; dtype: string; dims: number[]; data: ArrayBufferView | string[] }
  | { id: string; error: 'unknown' | 'evicted' | 'disposed' | string };
```

`call:start` is the one addition to the six event names in the guidance: `result` closes a
row, so something has to open it. Ids are string counters (`c1`, `r7`, `t42`); `t` is
`performance.now()`.

## Phases

> Each phase must be independently verifiable and independently committable.
> A phase that cannot be verified is not a phase, it is a hope.

### Phase 1: Toolchain

Scripts, configs and a smoke test so that `npm run typecheck`, `npm test`, `npm run lint`
and `npm run build` all pass on an almost-empty `src/`.

**Changes**

| File | Change |
|---|---|
| `package.json` | `name: transformersjs-inspector`, `version: 0.1.0`, `type: module`, `engines.node: >=20`, `files: ["dist"]`, `sideEffects: ["./dist/preload.js"]`. Scripts: `dev` = `vite --config vite.demo.config.ts`, `typecheck` = `tsc -p tsconfig.json --noEmit`, `test` = `vitest run`, `lint` = `eslint .`, `build` = `vite build && tsc -p tsconfig.build.json`, `build:demo` = `vite build --config vite.demo.config.ts`, `e2e` = `playwright test`, `screenshots` = `playwright test e2e/screenshots.spec.ts`. The 10 dev deps above; `peerDependencies` + `peerDependenciesMeta.optional` for `@huggingface/transformers`. |
| `tsconfig.json` | `strict`, `target: ES2022`, `module: ESNext`, `moduleResolution: Bundler`, `lib: [ES2022, DOM, DOM.Iterable]`, `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`, `isolatedModules`, `verbatimModuleSyntax`, `skipLibCheck`, `noEmit`; `include: ["src", "test", "demo", "e2e", "*.config.ts"]`. |
| `tsconfig.build.json` | extends base; `include: ["src"]`, `declaration`, `emitDeclarationOnly`, `outDir: dist`, `noEmit: false`. |
| `vite.config.ts` | `defineConfig` from `vitest/config`. `build.lib = { entry: { index: 'src/index.ts' }, formats: ['es'] }`, `build.target: 'es2022'`, `rollupOptions.external: ['@huggingface/transformers']`, `build.sourcemap: true`. `test = { include: ['test/**/*.test.ts'], environment: 'node' }`. |
| `vite.demo.config.ts` | `root: 'demo'`, `base: './'`, `build.outDir: '../dist-demo'`, `emptyOutDir: true`, `server: { port: 5173, strictPort: true }`. |
| `eslint.config.js` | Flat config: `ignores: ['dist/**', 'dist-demo/**', 'docs/**', '.cache/**', 'node_modules/**']`, `@eslint/js` recommended, `typescript-eslint` `recommended` (not type-checked), rules: `@typescript-eslint/consistent-type-imports: error`, `no-console: ['warn', { allow: ['warn', 'error'] }]`. |
| `src/index.ts` | `export const VERSION = '0.1.0';` |
| `test/smoke.test.ts` | asserts `VERSION` matches `package.json`. |
| `demo/index.html`, `demo/main.ts` | placeholder page that imports `../src/index.ts` and prints `VERSION`; replaced in Phase 5. |
| `.gitignore` | add `dist-demo/`, `.cache/`, `test-results/`, `playwright-report/`. |
| `AGENTS.md` | Fill the template (< 60 lines): what this is, stack, the six commands, layout (`src/`, `src/wrap/`, `src/panel/`, `test/`, `demo/`, `e2e/`, `docs/`, `spike/`), conventions (no runtime import of transformers in `src/`, events must pass `structuredClone`, no inline `style=`, `npm test` is offline, `spike/` is read-only reference), do-not (edit `node_modules`, edit `docs/00-*`/`01-*`/`02-*`). |

**Verify**

```bash
npm install && npm run typecheck && npm test && npm run lint && npm run build
```

**Done when:** all four commands exit 0, `dist/index.js` and `dist/index.d.ts` exist, and
`npm run dev` serves the placeholder demo on :5173.

### Phase 2: Event bus, tensor summaries, tensor store

The transport-agnostic core. No DOM, no Transformers.js. Everything here is unit-tested
against the fake tensor shapes recorded in the research (`input_ids [1,7] int64`,
`last_hidden_state [1,7,384] float32`, `logits [1,1,128256] float32`,
`present.0.key [1,2,N,16] float32` with `location:'gpu-buffer'`).

**Changes**

| File | Change |
|---|---|
| `src/events.ts` | The schema above, plus `export function isInspectorEvent(x: unknown): x is InspectorEvent`. |
| `src/types.ts` | Structural types: `TensorLike { type: string; dims: readonly number[]; location?: string; readonly data: ArrayLike<unknown>; getData?(): Promise<ArrayLike<unknown>> }`, `SessionLike { inputNames: readonly string[]; outputNames: readonly string[]; run(feeds: Record<string, TensorLike>, ...rest: unknown[]): Promise<Record<string, TensorLike>> }`, `TokenizerLike { _call(text: unknown, opts?: unknown): { input_ids: TensorLike; [k: string]: unknown }; _tokenizer?: { id_to_token?(id: number): string | undefined }; decode?(ids: number[]): string }`, `StreamerLike { put(value: unknown): void; end(): void }`, `LogitsProcessorLike = (input_ids: unknown, logits: TensorLike) => TensorLike`, `GenerateLike { generate(opts: Record<string, unknown>): Promise<unknown> }`, `PipelineLike { task?: string; model: { sessions: Record<string, SessionLike>; config?: Record<string, unknown>; generate?: GenerateLike['generate'] }; tokenizer?: TokenizerLike; _call(...args: unknown[]): Promise<unknown> }`. |
| `src/bus.ts` | `export type WireMessage = { __tjsi: 1 } & ({ kind: 'event'; event: InspectorEvent } \| { kind: 'request'; reqId: string; name: string; payload: unknown } \| { kind: 'response'; reqId: string; ok: boolean; payload?: unknown; error?: string })`. `export interface Transport { post(msg: WireMessage): void; subscribe(cb: (msg: WireMessage) => void): () => void; close?(): void }`. `export class InspectorBus { constructor(opts?: { maxHistory?: number /* 500 */ }); readonly history: readonly InspectorEvent[]; emit(ev): void; on(listener): () => void; handle<K extends keyof RequestMap>(name: K, h: (req: RequestMap[K]['req']) => RequestMap[K]['res'] \| Promise<...>): () => void; request<K>(name: K, req, opts?: { timeoutMs?: number /* 10000 */ }): Promise<RequestMap[K]['res']>; connect(t: Transport): () => void; clear(): void }`. `connect` relays local emits to the transport, delivers transport events locally without re-posting to the transport they came from, forwards `request` to connected transports when no local handler exists, and answers incoming requests from local handlers. `export function loopbackPair(): [Transport, Transport]` (in-memory, async delivery via `queueMicrotask`). `export class InspectorError extends Error`. |
| `src/summarize.ts` | `export const DEFAULT_HEAD = 8`; `bytesPerElement(dtype): number` (float32 4, float16 2, float64 8, int64/uint64 8, int32/uint32 4, int16/uint16 2, int8/uint8/bool 1, string 0); `elementCount(dims)`; `isCpuResident(t)` (`location` undefined, `'cpu'` or `'cpu-pinned'`); `headOf(t, n): (number\|string)[] \| null` (bigint → `Number`, never touches `.data` when not CPU-resident); `summarizeTensor(t, name, id, head = DEFAULT_HEAD): TensorSummary`; `readTensor(t): Promise<ArrayBufferView \| string[]>` (`t.data` when CPU-resident, otherwise `await t.getData()`; rejects `InspectorError('disposed')` when either throws). |
| `src/store.ts` | `export class TensorStore { constructor(opts?: { maxBytes?: number /* 64 MiB */; head?: number }); put(t: TensorLike, name: string): TensorSummary; read(id: string): Promise<TensorData>; has(id): boolean; readonly bytes: number; readonly count: number; clear(): void; attachTo(bus: InspectorBus): () => void /* registers the 'tensor' handler */ }`. Same tensor object → same id (`WeakMap<object, string>`). Insertion-order LRU by `summary.bytes`; evicted ids answer `{ error: 'evicted' }`, unknown ids `{ error: 'unknown' }`. |
| `src/preview.ts` | `previewInput(x: unknown): InputPreview` (string, string[], objects with `width/height/channels` → image, `Float32Array`/`Float64Array` → audio, `{ audio, sampling_rate }`, `Blob`/`URL`/http string → image `src`, else `other`); `toCloneSafe(v: unknown, store?: TensorStore, opts?: { maxString?: 2000; maxArray?: 64; maxDepth?: 6 }): unknown` (tensor-like → `{ $tensor: TensorSummary }` via `store.put` when a store is given, else summary without id; typed arrays → truncated number arrays + `$truncated`; bigint → number; functions/symbols/undefined dropped; cycles → `'[cycle]'`; `Error` → `{ message }`). |
| `test/fakes.ts` | `fakeTensor({ type, dims, data?, location? })` (gpu variants: `data` getter throws, `getData()` resolves), `fakeSession({ name, inputNames, outputNames, produce(feeds) })`, `fakeTokenizer()` (`_call` returns `{ input_ids: int64 [1,n], attention_mask }`, `_tokenizer.id_to_token`), `fakeGenerativeModel()` (a `generate` that calls each `logits_processor` entry once per step with `[1,V]` logits, calls `streamer.put([[prompt ids]])` first then `put([[id]])` per step, then `end()`), `fakePipeline()` (Callable-shaped: a function whose `_call` runs tokenizer → session → returns a result), plus `fixtureEvents(): InspectorEvent[]` for one embedding call and one 3-step generation call. |
| `test/bus.test.ts` | emit/on/unsubscribe/history cap; `handle`/`request` round trip and timeout; `loopbackPair` relays events once (no echo) and forwards requests to the side with a handler; every fixture event passes `structuredClone` and `JSON.parse(JSON.stringify())` unchanged. |
| `test/summarize.test.ts` | dtype table; head of int64 is plain numbers; gpu tensor summary has `head: null` and its `data` getter was never called; `readTensor` on gpu uses `getData`. |
| `test/store.test.ts` | put/read round trip; same object same id; eviction by byte budget; `attachTo(bus)` answers `bus.request('tensor')`. |
| `test/preview.test.ts` | each `InputPreview` branch; `toCloneSafe` handles cycles, bigint, tensor-like, truncation. |

**Verify**

```bash
npm test -- test/bus.test.ts test/summarize.test.ts test/store.test.ts test/preview.test.ts && npm run typecheck && npm run lint
```

**Done when:** the four test files pass offline, and no code under `src/` references
`document`, `window`, or `@huggingface/transformers`.

### Phase 3: Panel

A shadow-DOM widget driven purely by the bus. Built before the wrappers so it can be
developed and tested against `fixtureEvents()` with no model. Three levels of laziness:
collapsed panel only updates a counter; an open panel renders one summary line per call;
expanding a row renders its sections; "Load values" is the only thing that triggers
`bus.request('tensor')`.

**Changes**

| File | Change |
|---|---|
| `src/panel/model.ts` | Pure reducer, node-testable. `CallView { id; label; task; input; tokenize: TokenizeEvent[]; runs: RunView[]; steps: StepView[]; result: unknown; error: string \| null; ms: number \| null; startedAt: number; done: boolean; synthetic: boolean }`, `RunView { runId; session; inputs; outputs; ms; error; done }`, `StepView { step; topK?; tensorId?; ids?; text? }`, `PanelState { calls: CallView[]; byId: Map; byRun: Map<runId, callId>; total: number }`. `createState()`, `reduce(state, ev): { call: CallView; change: 'new' \| 'updated' }`. Orphan `run:start` (`callId: null`) opens a synthetic call labelled `direct · <session>`; orphan `logits`/`token` go to the newest open call. Rows capped at `maxCalls` (default 200, oldest dropped). |
| `src/panel/dom.ts` | `h(tag, attrs?, ...children)` creating elements with `textContent`/`dataset`/`className` only (never `innerHTML`); `fmtMs`, `fmtBytes`, `fmtNum` (4 significant digits), `fmtDims`. |
| `src/panel/styles.ts` | `export const PANEL_CSS: string` (host docked bottom-right, `:host { all: initial; position: fixed; ... z-index: 2147483647 }`, header, badge, row, sections, tensor table, top-k bars using CSSOM `style.width` on a `.bar` element — CSSOM property writes are not blocked by CSP), `export function adoptStyles(shadow: ShadowRoot): void` (constructed `CSSStyleSheet` + `replaceSync` when available, fallback to a `<style>` element for environments without it). |
| `src/panel/render.ts` | `renderRowSummary(call): HTMLElement` (`data-call`, `#n`, label, input excerpt, ms, status dot), `renderRowDetails(call, ctx): HTMLElement` with sections `Input`, `Tokenizer` (one chip per token: `id` + string, per batch row), `Session runs` (per run: inputs table then outputs table, one `<tr data-tensor=id>` per tensor: name, dtype, dims, location, bytes, head, and a `Load values` button when `id` present), `Generation` (per step: token id/text and a top-k table with `prob` bars, only when `steps.length > 0`), `Result` (`<pre>` of `JSON.stringify(result, null, 2)`, with `$tensor` entries rendered as tensor rows). `renderTensorValues(el, data: TensorData)` renders up to 4096 values (bigint arrays via `Number`) plus a `… N more` note, or the error string. |
| `src/panel/panel.ts` | `export interface PanelOptions { container?: Element; open?: boolean /* false */; title?: string; maxCalls?: number }`, `export interface InspectorPanel { host: HTMLElement; shadow: ShadowRoot; open(); close(); isOpen(): boolean; clear(); destroy() }`, `export function mountPanel(bus: InspectorBus, opts?): InspectorPanel`. Host is `<div data-tjsi-panel>` appended to `opts.container ?? document.body`, `attachShadow({ mode: 'open' })`. Replays `bus.history` then subscribes. While closed: reducer + badge `textContent` only. When opened: renders summary rows for calls not yet rendered, then incrementally on each event (`change === 'new'` → append; `'updated'` → patch that row's summary and, if expanded, re-render its details). Click handling by delegation on `data-action`: `toggle` (header), `expand` (row), `load` (tensor → `bus.request('tensor', { id })` → `renderTensorValues`), `clear`. |
| `src/index.ts` | add `export { mountPanel } from './panel/panel'`, `export { InspectorBus, loopbackPair } from './bus'`, `export { TensorStore } from './store'`, `export type * from './events'`. |
| `test/panel-model.test.ts` | reduce `fixtureEvents()` → 2 calls, run/step/tokenize attached to the right call, synthetic call for orphan runs, cap. |
| `test/panel.test.ts` | `// @vitest-environment happy-dom`. Mount on a bus, emit fixtures while closed → `shadow.querySelectorAll('[data-call]').length === 0` and badge reads `2`; `open()` → 2 rows; click `expand` → tokenizer chips, `[data-tensor]` rows with dims text `[1, 7, 384]`, generation section with 3 steps and top-k rows; click `load` with a `bus.handle('tensor')` fake → values rendered; error response renders the error; `destroy()` removes the host. |

**Verify**

```bash
npm test -- test/panel-model.test.ts test/panel.test.ts && npm run typecheck && npm run lint
```

**Done when:** both test files pass offline and `grep -rn "innerHTML\|style=" src/panel` returns nothing.

### Phase 4: Wrappers and `attach()`

The instance wrapping verified in the feasibility prototype, made restorable and idempotent, and
correlated into calls.

**Changes**

| File | Change |
|---|---|
| `src/context.ts` | `export interface InspectorOptions { head: number /* 8 */; topK: number /* 10 */; retainBytes: number /* 64 MiB */; retainLogits: boolean /* false */; label: string }`, `export class WrapContext { constructor(bus, store, opts); currentCallId: string \| null; tokenizer: TokenizerLike \| null; nextCallId(); nextRunId(); tokenToString(id: number): string \| null /* _tokenizer.id_to_token, else decode([id]), else null */; now() }`. |
| `src/wrap/session.ts` | `export const WRAPPED = Symbol.for('transformersjs-inspector.wrapped')`; `wrapSession(session: SessionLike, name: string, ctx): () => void` — no-op restore if `session[WRAPPED]`; replaces `session.run` with an async wrapper that (1) summarises every feed via `ctx.store.put` **before** calling through (proxy mode), (2) emits `run:start`, (3) awaits the original, (4) summarises outputs and emits `run:end` with `ms`; on throw emits `run:end` with `error: String(e)` and rethrows. `wrapSessions(sessions: Record<string, SessionLike>, ctx): () => void`. |
| `src/wrap/tokenizer.ts` | `wrapTokenizer(tok: TokenizerLike, ctx): () => void` — wraps the instance's `_call` (own property shadowing the prototype, as in the spike); `idsFrom(input_ids: TensorLike): number[][]` from `dims [b, s]` and `data` (BigInt64Array or Int32Array); tokens via `ctx.tokenToString`; emits `tokenize`. If `_call` is not a function, `console.warn` once and return a no-op (risk 1 in the research). |
| `src/wrap/generation.ts` | `topKFromLogits(data: ArrayLike<number>, k): { entries: { id; logit; prob }[]; vocab }` — one pass for max, one for the softmax denominator, partial selection for k. `wrapGenerate(model: GenerateLike, ctx): () => void` — replaces `model.generate` with `(opts) => real({ ...opts, logits_processor: mergeProcessors(opts.logits_processor, inspectorProcessor), streamer: mergeStreamer(opts.streamer, inspectorStreamer) })`. `mergeProcessors` spreads the caller's list if iterable (`LogitsProcessorList` has `[Symbol.iterator]`), else its `.processors` array, else `[]`, and appends ours as a plain function `(input_ids, logits) => { emit 'logits' from batch row 0; return logits }` (stores the full logits in the store only when `retainLogits`). `inspectorStreamer` is `{ put(value) , end() }`: first `put` after `end`/construction is the prompt (skipped, as `TextStreamer` does with `skip_prompt`), each later `put` emits `token` with `ids = value[0].map(Number)` and `text = tokens joined via ctx.tokenToString`; `mergeStreamer` calls ours then the host's. Escape hatch: `opts.transformers?: { LogitsProcessorList }` — when provided, build a real `LogitsProcessorList` and `.push` instead of an array (only used if the e2e test in Phase 5 shows arrays are rejected). |
| `src/wrap/pipeline.ts` | `wrapPipelineCall(pipe: PipelineLike, ctx): () => void` — wraps the instance's `_call` (same Callable mechanism as the tokenizer): allocates `callId`, sets `ctx.currentCallId`, emits `call:start` with `previewInput(args[0])`, awaits the original, emits `result` with `toCloneSafe(result, ctx.store)` and `ms`; on throw emits `result` with `error` and rethrows; always restores `ctx.currentCallId`. |
| `src/attach.ts` | `export interface AttachOptions extends Partial<InspectorOptions> { bus?: InspectorBus; store?: TensorStore; panel?: boolean \| PanelOptions /* default: typeof document !== 'undefined' */; transformers?: { LogitsProcessorList: new () => { push(p: unknown): void } } }`, `export interface AttachHandle { bus; store; detach(): void }`, `export function attach(pipe: unknown, opts?): AttachHandle`. `assertPipeline` throws `InspectorError('attach(): expected a Transformers.js pipeline with model.sessions')`. Default `label` = `${pipe.task ?? 'pipeline'} · ${pipe.model.config?.model_type ?? '?'}`. Wraps sessions, tokenizer (if present), `model.generate` (if a function), then `_call`; `detach()` restores in reverse. Uses `getDefaultBus()`/`getDefaultStore()` and `ensurePanel()` unless overridden. |
| `src/default.ts` | `getDefaultBus()`, `getDefaultStore()` (kept on `globalThis[Symbol.for('transformersjs-inspector')]` so the preload bundle and an `attach()` from a different bundle share one bus), `ensurePanel(bus, opts?)` (mounts once per bus; no-op without `document`). |
| `src/index.ts` | export `attach`, `AttachOptions`, `AttachHandle`, `getDefaultBus`, `WRAPPED`. |
| `test/wrap-session.test.ts` | inputs summarised before `run` resolves (spy order), `run:start`/`run:end` pair with matching `runId`, gpu outputs have `head: null`, error path, idempotent double wrap, `detach` restores the original function identity. |
| `test/wrap-tokenizer.test.ts` | `tokenize` event ids/tokens for `[1,n]` int64; batch `[2,n]`; missing `_call` degrades with one warning. |
| `test/generation.test.ts` | `topKFromLogits` numerics (probabilities sum ≤ 1, sorted desc, correct ids); with `fakeGenerativeModel`: one `logits` and one `token` event per step, prompt `put` skipped, host's own processor list and streamer still called, `retainLogits` stores a tensor id. |
| `test/attach.test.ts` | `attach(fakePipeline(), { panel: false, bus })` then calling the pipeline emits, in order, `call:start`, `tokenize`, `run:start`, `run:end`, `result` with one `callId`; every event passes `structuredClone`; `detach()` restores all four originals; attaching a non-pipeline throws `InspectorError`. |

**Verify**

```bash
npm test && npm run typecheck && npm run lint
```

**Done when:** the whole unit suite passes offline and `attach()` is exported from `src/index.ts`.

### Phase 5: Demo page and Playwright e2e

The real-model proof. Demo exercises feature extraction, text generation and text
classification; e2e covers the first two with the small fixture models, plus the timing
budget.

**Changes**

| File | Change |
|---|---|
| `demo/tf.ts` | `export const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'`; `export const tf = (await import(/* @vite-ignore */ TRANSFORMERS_URL)) as typeof import('@huggingface/transformers')` (types from the dev dep, runtime from the CDN, exactly the spike path). |
| `demo/index.html` | Three sections with `data-task="feature-extraction" \| "text-generation" \| "text-classification"`, each a `<textarea>`, a `Run` button, a `<pre data-output>`, and a `data-status` attribute cycling `idle → loading → running → done \| error`; a `Benchmark` button with `<output data-bench>`; a progress line. Script tag `./main.ts`. |
| `demo/main.ts` | Lazily builds each pipeline on first Run: `feature-extraction` → `Xenova/all-MiniLM-L6-v2` with `{ pooling: 'mean', normalize: true }`; `text-generation` → `onnx-community/tiny-random-LlamaForCausalLM-ONNX` (`{ device: 'auto' }` as in the spike) with `{ max_new_tokens: 3 }`; `text-classification` → `Xenova/distilbert-base-uncased-finetuned-sst-2-english`. Calls `attach(pipe, { label })` right after `pipeline()` resolves (one shared default bus and panel). `bench()`: warm-up 5 runs, then 30 embeddings with the handle detached, re-attach with the panel closed, 30 more; writes both medians and the ratio to `[data-bench]` and `window.__bench`. |
| `playwright.config.ts` | `testDir: 'e2e'`, `projects: [chromium]`, `webServer: { command: 'npm run dev', url: 'http://localhost:5173', reuseExistingServer: true, timeout: 60_000 }`, `timeout: 300_000` (first model download), `retries: 0`, `use.baseURL`. |
| `e2e/fixtures.ts` | `export const test = base.extend({ context: async ({}, use) => { const ctx = await chromium.launchPersistentContext('.cache/pw-profile', { headless: true }); await use(ctx); await ctx.close(); } })` so the browser Cache API keeps model and CDN files across runs. Helper `runTask(page, task, text)` clicks Run and waits for `[data-task=...][data-status="done"]`. |
| `e2e/demo.spec.ts` | feature extraction: after one run the panel badge reads `1`; open panel → one `[data-call]` row whose label contains `feature-extraction`; expand → tokenizer chips include `[CLS]`, session run rows list `input_ids`, `attention_mask`, `token_type_ids` with dims `[1, N]` and an output `last_hidden_state` with dims `[1, N, 384]`; click `Load values` on `last_hidden_state` → at least 384 numbers rendered; the `Result` section shows a `$tensor` with dims `[1, 384]`. |
| `e2e/generation.spec.ts` | text generation with `max_new_tokens: 3`: exactly 3 `run` rows in the call (one prefill + two decode, or 3 — assert `>= 3`), a Generation section with 3 steps, each with a top-k table of 10 rows whose probabilities are within `(0, 1]` and sorted descending, and 3 `token` ids; decoded result in the `Result` section equals the page's `[data-output]`. |
| `e2e/overhead.spec.ts` | clicks `Benchmark`, reads `window.__bench`, prints `attached/detached` medians to the test output, asserts `ratio < 1.25`. The strict ~5 % figure from the brief is read from the same output and recorded by hand in `docs/progress.md`; a 5 % assertion on ~10 ms runs is noise-bound and would flake. |
| `package.json` | no script changes (`e2e` exists); README gets a line about `npx playwright install chromium`. |

**Verify**

```bash
npx playwright install chromium && npm run e2e -- e2e/demo.spec.ts e2e/generation.spec.ts e2e/overhead.spec.ts && npm test && npm run typecheck && npm run lint
```

**Done when:** the three specs pass with network access (models cached after the first run),
and the overhead ratio printed by `overhead.spec.ts` is recorded in the README.

### Phase 6: Zero-touch preload

The research's secondary entry point, shipped with its caveat.

**Changes**

| File | Change |
|---|---|
| `src/preload-core.ts` | `export interface OrtLike { InferenceSession: { create(...args: unknown[]): Promise<SessionLike> }; env: unknown; Tensor: unknown }`; `export const ORT_SYMBOL = Symbol.for('onnxruntime')`; `export function installPreload(ort: OrtLike, ctx: WrapContext): OrtLike` — builds the shim from the spike (`{ ...ort, InferenceSession: Object.assign(Object.create(Real), Real, { create }) }`) whose `create` awaits the real one and `wrapSession(session, nameFor(args, session), ctx)`; `nameFor` uses the last path segment of a string arg without `.onnx`, else `session#<n> → <outputNames[0]>`; sets `globalThis[ORT_SYMBOL]` and returns the shim. `export function isInstalled(): boolean`. |
| `src/preload.ts` | Side-effect entry: `import * as ort from 'onnxruntime-web/webgpu'` (static, so the symbol is set synchronously before any later `<script type="module">` evaluates); `installPreload(ort, new WrapContext(getDefaultBus(), getDefaultStore(), defaults))`; `if (typeof document !== 'undefined') ensurePanel(bus)`; `export { bus, store }`. |
| `vite.shared.ts` | `export const ORT_VERSION = createRequire(import.meta.url)('@huggingface/transformers/package.json').dependencies['onnxruntime-web']`; `export const ORT_CDN_URL = \`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.bundle.min.mjs\``. The pin is derived from the installed Transformers.js, never hand-copied. |
| `vite.config.ts` | add entry `preload: 'src/preload.ts'`; `external` gains `'onnxruntime-web/webgpu'`; `rollupOptions.output.paths = { 'onnxruntime-web/webgpu': ORT_CDN_URL }` so `dist/preload.js` statically imports the CDN build (the verified spike URL) instead of bundling 2 MB of ORT. |
| `vite.demo.config.ts` | `resolve.alias = { 'onnxruntime-web/webgpu': ORT_CDN_URL }` so the dev server and `build:demo` load the same CDN file (Vite leaves `https://` specifiers to the browser). |
| `demo/preload.html`, `demo/preload-entry.ts`, `demo/preload-host.ts` | Page with two module scripts in order: `preload-entry.ts` (`import '../src/preload.ts'`) then `preload-host.ts` (imports `tf` from `./tf.ts`, `pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { device: 'auto' })`, runs one sentence, sets `data-status="done"`; **no** `attach()` call). A visible note: "zero-touch requires `device: 'auto'` — see README". |
| `test/preload.test.ts` | `installPreload(fakeOrt, ctx)`: `globalThis[Symbol.for('onnxruntime')]` is set, `create` returns a session whose `run` emits `run:start`/`run:end` on `ctx.bus`, `env` and `Tensor` are the original objects (same identity), double-install is a no-op. `test/ort-pin.test.ts`: `ORT_VERSION` from `vite.shared.ts` is a non-empty string equal to the dependency in `node_modules/@huggingface/transformers/package.json` (offline, filesystem only). |
| `e2e/preload.spec.ts` | opens `/preload.html`; waits for `done`; the panel shows exactly 1 row labelled `direct · session#1 → last_hidden_state` with a `last_hidden_state [1, N, 384]` output; `page.evaluate(() => 'onnxruntime' in Symbol.keyFor ...)` — assert `globalThis[Symbol.for('onnxruntime')]` is defined. |
| `README.md` | (section only; full README in Phase 8) `## Zero-touch preload` — the script-tag snippet, the `device:'auto'` requirement with the one-line reason from the research, the ORT version coupling, and the upstream fix to file. |

**Verify**

```bash
npm test -- test/preload.test.ts test/ort-pin.test.ts && npm run build && grep -c "cdn.jsdelivr.net/npm/onnxruntime-web@" dist/preload.js && npm run e2e -- e2e/preload.spec.ts && npm run typecheck && npm run lint
```

**Done when:** `dist/preload.js` contains the CDN ORT import and no bundled ORT (file < 100 KB),
and `preload.spec.ts` passes with a host page that never calls `attach()`.

### Phase 7: Web Worker bridge

The bus was designed for this in Phase 2; this phase only adds a `postMessage` transport and
proves it end to end. Severable: nothing in "Done looks like" depends on it.

**Changes**

| File | Change |
|---|---|
| `src/worker.ts` | `export interface PortLike { postMessage(msg: unknown): void; addEventListener(type: 'message', cb: (e: { data: unknown }) => void): void; removeEventListener(type: 'message', cb: ...): void; start?(): void }` (satisfied by `Worker`, `MessagePort`, and a worker's `self`); `export function messagePortTransport(port: PortLike): Transport` (filters on `data.__tjsi === 1`, calls `port.start?.()`); `export function connectWorker(worker: PortLike, bus = new InspectorBus()): InspectorBus` (page side); `export function exposeToPage(bus: InspectorBus, port: PortLike = self as unknown as PortLike): () => void` (worker side). |
| `src/index.ts` | export `messagePortTransport`, `connectWorker`, `exposeToPage`. |
| `vite.config.ts` | add entry `worker: 'src/worker.ts'` (also reachable from `index`; the separate entry keeps a worker bundle free of panel code). |
| `demo/worker.html`, `demo/worker-main.ts`, `demo/inference.worker.ts` | Page: `const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' })`, `const bus = connectWorker(worker)`, `mountPanel(bus, { open: true })`, a Run button posting `{ text }`. Worker: imports `tf` from `./tf.ts`, builds the MiniLM pipeline, `const bus = new InspectorBus(); exposeToPage(bus); attach(pipe, { bus, panel: false })`, runs on `{ text }` messages and posts `{ done: true }` (untagged host messages are ignored by the transport and vice versa). |
| `test/worker.test.ts` | Node's global `MessageChannel`: bus A on `port1`, bus B on `port2` via `messagePortTransport`; an event emitted on B arrives on A once; `A.request('tensor', { id })` is answered by B's `TensorStore` handler and the response `data` is a `Float32Array`; ports closed in `afterEach` so Vitest exits. |
| `e2e/worker.spec.ts` | opens `/worker.html`, clicks Run, waits for `done`; the page panel shows 1 row with `last_hidden_state`; `Load values` renders ≥ 384 numbers fetched from the worker. |

**Verify**

```bash
npm test -- test/worker.test.ts && npm run e2e -- e2e/worker.spec.ts && npm run typecheck && npm run lint
```

**Done when:** the unit test proves relay + request round trip over a real `MessageChannel`,
and the e2e test proves it across a real module worker with the panel unchanged.

### Phase 8: Packaging and README

**Changes**

| File | Change |
|---|---|
| `package.json` | `exports: { '.': { types: './dist/index.d.ts', import: './dist/index.js' }, './preload': { types: './dist/preload.d.ts', import: './dist/preload.js' }, './worker': { types: './dist/worker.d.ts', import: './dist/worker.js' } }`, `main`/`types` mirrors, `keywords`, `repository`, `license: MIT`. |
| `README.md` | Rewrite for the project: what it is (the "Network tab for local model calls" line from the brief), 30-second usage with `attach()`, the panel screenshot, per-step logits screenshot, zero-touch preload section (from Phase 6) with the `device:'auto'` caveat, worker section, options table (`AttachOptions`), the event schema (link to `src/events.ts`), the "what you cannot see" list from the brief's fence (no per-layer activations), development commands, and the two fixture models the e2e uses. |
| `e2e/screenshots.spec.ts` | Runs feature extraction and generation on the demo, opens and expands rows, writes `docs/img/panel-embedding.png` and `docs/img/panel-generation.png` via `page.screenshot({ clip })`. |
| `docs/img/` | the two PNGs (committed). |
| `LICENSE` | MIT. |

**Verify**

```bash
npm run build && node --input-type=module -e "import('./dist/index.js').then(m => { if (typeof m.attach !== 'function') process.exit(1); console.log(Object.keys(m)) })" && npm pack --dry-run && npm run screenshots && npm run build:demo && npm test && npm run typecheck && npm run lint
```

**Done when:** `npm pack --dry-run` lists only `dist/**`, `README.md`, `LICENSE`, `package.json`;
`dist/` has `index.js`, `preload.js`, `worker.js` and matching `.d.ts`; `dist-demo/` builds;
`docs/img/` contains the two screenshots referenced by the README.

## Out of scope

From the brief's fence, restated so the loop does not drift:

- Per-layer internal activations, or any rewriting of ONNX graphs. Only declared graph
  inputs/outputs are shown.
- TensorFlow.js, WebLLM, MediaPipe, raw onnxruntime-web sites. Transformers.js 4.x only.
- A browser extension or DevTools panel. The event stream is designed for one; not built here.
- Persisting, exporting, or uploading captured data. No `localStorage`, no downloads, no fetch.
- Editing or replaying inputs. The panel is read-only.
- Patching `@huggingface/transformers` or `onnxruntime-web` on disk, or vendoring either.
  Every hook is a runtime wrap of an instance property (`sessions[name].run`,
  `tokenizer._call`, `model.generate`, `pipe._call`) or the documented
  `globalThis[Symbol.for('onnxruntime')]` injection point.

Tempting things discovered during research, also out:

- Filing the upstream fix for the empty `supportedDevices` list in the custom-runtime branch.
  Worth doing, by a human, outside the loop. The plan documents the `device:'auto'` caveat
  instead.
- Wrapping `pipe.processor` (image/audio feature extractors). Inputs of those kinds get a
  metadata-only `InputPreview`; their preprocessed tensors are visible at the session boundary.
- Attributing session runs when the host issues concurrent pipeline calls on the same
  pipeline. v1 uses a single `currentCallId`; interleaved calls may attribute runs to the
  wrong row. Documented limitation.
- Pretty-printing of `present.*`/`past_key_values.*` as a growing cache view; they are listed
  as ordinary tensors.
- Encoder-decoder (Whisper, T5) demo sections. Covered by construction at the session
  boundary (`encoder_model` + `decoder_model_merged`), not demonstrated in v1.
- CI workflow, npm publish, GitHub Pages deploy of `dist-demo/`. Manual for v1.

## Rollback

Nothing here has a migration, a data store, or an external side effect until publish, so
`git revert` of the offending story's commit is the full rollback while unpublished. After
an npm publish: publish a patch that reverts, or `npm deprecate` the bad version; host pages
opt out by removing the `attach()` call or the preload `<script>` tag, which restores every
wrapped instance to library behaviour because nothing on disk was changed. `detach()` on the
`AttachHandle` restores originals at runtime for hosts that want to keep the dependency.

---

## Reviewer sign-off

- [ ] I can predict the shape of the resulting diff from this plan
- [ ] Every phase has a real verification command, and I have run each one at least once by hand
- [ ] Phase ordering has no hidden dependency (each phase leaves the tree green)
- [ ] The approach section's rejected alternative is genuinely worse
- [ ] Nothing in "must not change" from the brief gets touched

**Reviewed by:** Pratik Amin **Date:** 2026-09-10

> Unchecked box = do not start the loop. Fix the plan first. The loop will
> faithfully and tirelessly execute a bad plan.
