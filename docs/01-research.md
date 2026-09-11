# Research: transformersjs-inspector

> Greenfield technology scan. Written 2026-09-10
> against `@huggingface/transformers@4.2.0` (npm tarball, source read directly) with two
> throwaway browser prototypes in `spike/`. Reviewed before planning started.

## Verdict

The brief's premise holds: every model call in Transformers.js funnels through one
function, `sessionRun`, which calls `session.run` on an ONNX Runtime session that the model
instance exposes publicly, so a library can observe the exact named input and output
tensors of any pipeline without touching library files. Both prototypes ran end to end in
the browser and captured inputs, outputs, tokenizer ids, per-step logits and streamed
tokens. The one premise that is *not* clean is zero-touch: the documented ORT injection
hook works, but it breaks default device selection, so it only works when the host page
passes `device: 'auto'`. Explicit `attach(pipeline)` is the reliable v1 entry point.

## Nothing like this exists

Checked 2026-09-10. Closest neighbours, and why each is not it:

| Thing | What it is | Why it is not this |
|---|---|---|
| tfjs-vis | side-panel "visor" for TensorFlow.js model behaviour | TensorFlow.js only; good UI reference |
| Xenova Tokenizer Playground | text → ids demo | standalone app, one stage only |
| poloclub Transformer Explainer, Bycroft LLM Visualization | in-browser model internals | run their own fixed GPT-2 / minGPT, not the host page's model |
| WebGPU Inspector, Spector.gpu | capture GPU buffers | anonymous buffers, not named tensors; WebGPU only |
| HF Python `model_debugging_utils` | dump per-module tensors to JSON | Python |

## Relevant files (in `@huggingface/transformers@4.2.0`, `src/`)

| File | Why it matters |
|---|---|
| `models/session.js:192` `sessionRun(session, inputs)` | the single choke point; validates inputs against `session.inputNames`, calls `runInferenceSession`, wraps outputs in the library's `Tensor` |
| `models/session.js:246` `validateInputs` | drops extra inputs, clones tensors when `wasm.proxy` is on |
| `models/modeling_utils.js:218` | `this.sessions = sessions` — public dict of ORT `InferenceSession`s keyed `model`, `encoder_model`, `decoder_model_merged`, … |
| `backends/onnx.js:103-107` | `globalThis[Symbol.for('onnxruntime')]` — if set, used instead of bundled ORT |
| `backends/onnx.js:98-150` | `supportedDevices` is only populated in the node and web branches, **not** in the custom-runtime branch |
| `backends/onnx.js:161` `deviceToExecutionProviders` | throws for any named device when `supportedDevices` is empty; `'auto'` returns the (empty) list instead |
| `models/session.js:80` | `session_options.executionProviders ??= executionProviders` — empty list is passed through and ORT web picks its own default |
| `tokenization_utils.js:357` `PreTrainedTokenizer._call` | text → `{input_ids, attention_mask, …}`; instance method, wrappable |
| `tokenization_utils.js:249` | `this._tokenizer` is a `@huggingface/tokenizers` `Tokenizer` with `id_to_token`, `token_to_id`, `encode` |
| `generation/logits_process.js` `LogitsProcessor`, `LogitsProcessorList` | public; `_call(input_ids, logits)` runs every decode step with the last-position logits. `LogitsProcessorList()` takes **no** constructor args — use `.push()` |
| `generation/streamers.js:40` `TextStreamer` | `token_callback_function(bigint[])` per step |
| `models/modeling_utils.js:914-991` | where `generate` builds the processor list and applies it |
| `pipelines/text-generation.js:175` | pipeline spreads caller options into `model.generate`, so `logits_processor` and `streamer` pass straight through |
| `utils/tensor.js:25` | library `Tensor`: `dims`, `type`, `data` (getter → `ort_tensor.data`), `location`, `tolist()` |
| `dist/transformers.web.js:7655` | the symbol hook survives bundling |

## How it works today (control flow for one pipeline call)

```
pipe(text)                                   pipelines/<task>.js _call
  └─ this.tokenizer(text)                    → { input_ids, attention_mask, … }  (library Tensors)
  └─ this.model(model_inputs)                modeling_utils forward
       └─ sessionRun(this.sessions.model, pick(inputs, session.inputNames))
            └─ session.run(ortFeed)          ← onnxruntime-web InferenceSession
            └─ replaceTensors(output)        → { logits | last_hidden_state | present.* … }
  └─ postprocess (softmax / pooling / decode)
```

For generation, `model.generate` loops: `forward` → `sessionRun` once per token, then
`prepared_logits_processor(all_input_ids, logits)` → sampler → `streamer.put(tokens)`.

## Where the change lands (the seams)

1. **Session boundary** — replace `session.run` on each `pipe.model.sessions[name]` instance
   with a wrapper. Verified: gives every named input and output tensor, dtype, dims,
   location, and wall time. Proved in `spike/b.html` for encoder (MiniLM) and decoder-only
   (tiny Llama: `input_ids`, `attention_mask`, `position_ids`, `past_key_values.*` in;
   `logits [1,1,128256]`, `present.*` out).
2. **Tokenizer boundary** — wrap `pipe.tokenizer._call`; pair the raw text with ids and
   `_tokenizer.id_to_token(id)` strings. Verified.
3. **Generation** — push a `LogitsProcessor` subclass into the caller's list (or supply one)
   for per-step logits; wrap or supply a `TextStreamer` for token ids. `LogitsProcessor`
   verified live once constructed correctly; `TextStreamer.token_callback_function` verified.
4. **Zero-touch preload (optional)** — set `globalThis[Symbol.for('onnxruntime')]` to a
   copy of onnxruntime-web whose `InferenceSession.create` returns run-wrapped sessions.
   Verified in `spike/index.html` **but** only with `device: 'auto'` (see constraints).

The library therefore has two entry points: `attach(pipe, opts)` (always works) and a
preload script (works for hosts that already pass `device:'auto'`, or after an upstream
fix). Both feed the same event stream into one panel.

## Constraints discovered

- **Custom-runtime branch has an empty device list.** With the symbol set, the default
  device `'wasm'` throws `Unsupported device: "wasm". Should be one of: .` before any
  session is created. `'auto'` slips through because it returns `supportedDevices` as-is
  and ORT web then chooses its own default (wasm ran fine). A three-line upstream PR
  (populate `supportedDevices` from the injected module's `env`, or treat an injected
  web runtime like the web branch) would make zero-touch unconditional. Worth filing.
- **The preload must run before Transformers.js evaluates.** ORT is chosen at module
  evaluation, so with a static `import` in the host bundle the preload has to be a
  separate earlier `<script>`; it cannot be a normal import inside the same module graph.
- **ORT version coupling.** The preload loads its own copy of onnxruntime-web (second
  download, ~2 MB JS + WASM). It must match the version Transformers.js pins
  (`1.26.0-dev.20260416-b7804b056c` for 4.2.0) or session options may not line up.
  `attach()` has no such coupling because it never touches ORT.
- **Tokenizer internals are underscore-private.** `_call` and `_tokenizer` are stable in
  practice but not documented API. Pin to 4.x and test against each minor.
- **Proxy mode moves tensors.** When `env.backends.onnx.wasm.proxy` is true, inputs are
  transferred to a worker and become unreadable after `run`; summarise inputs *before*
  calling through. The wrapper already does this.
- **GPU-resident tensors.** On WebGPU, `location` can be `gpu-buffer` (KV cache with
  `cache_sessions`). Reading them needs `await tensor.getData()` and a GPU→CPU copy. Panel
  must never do this eagerly.
- **Logits are big.** `[1,1,128256]` float32 per step ≈ 512 KB. Keep top-k on the hot path;
  store the full tensor only when the user asks.
- **Web Worker hosts.** Many real pages run the pipeline in a worker. `attach()` then has to
  live in the worker and post summaries to the page; the panel needs a `postMessage`
  transport. Not prototyped; design the event schema to be structured-clone safe from day one.

## Risks

1. **Underscore-private tokenizer API changes in a 4.x minor** → tokenizer rows go blank.
   Mitigation: feature-detect, degrade to ids only, CI against latest 4.x.
2. **Wrapping `session.run` changes timing** through summarisation work on the hot path.
   Mitigation: only read `dims`/`type`/`location` and a fixed 8-value head eagerly; measure
   in the demo; the brief's ~5 % budget is the acceptance test.
3. **Zero-touch remains conditional** until upstream accepts a fix. Mitigation: ship
   `attach()` as the headline API; document the preload as "works with `device:'auto'`".
4. **Shadow-DOM panel on pages with strict CSP** (no inline styles) → panel unstyled.
   Mitigation: adopt a constructed stylesheet, no inline `style=`.

## Answers to the brief's open questions

1. **Zero-touch?** Partially. The documented injection point works and was proved live, but
   only when the host passes `device:'auto'`. `attach(pipe)` is unconditional. Ship both;
   file the upstream fix.
2. **What is visible at `session.run`?** Every declared graph input and output, by name,
   with dtype, dims, location and data: encoder (`last_hidden_state`), decoder-only
   (`logits`, `present.*` KV cache growing `[1,2,N,16]` per step), and by construction
   encoder-decoder (`encoder_model` + `decoder_model_merged` sessions). Cost: negligible for
   CPU tensors when only a head is read (10–40 ms model runs showed no measurable overhead
   in the spike); a full copy for GPU tensors.
3. **Text→ids and per-step logits without private APIs?** ids: yes via the tokenizer's
   public call (the *wrapping* uses `_call`, but `pipe.tokenizer(text)` is public and the
   library can also call it itself). Logits: yes, `LogitsProcessor` is public and receives
   `(input_ids, logits)` every step. Token stream: yes, `TextStreamer`.
4. **Workers and WebGPU?** Not prototyped. Design: transport-agnostic event bus, summaries
   are plain JSON, full tensors fetched lazily via a request/response over the same bus.
   GPU tensors are summarised as `location:'gpu-buffer'` with dims only until requested.
5. **Dependencies and version pin?** Runtime deps: none. Peer: `@huggingface/transformers
   >=4 <5` (only for types and the preload's version constant). Dev: TypeScript, Vite,
   Vitest, a browser test runner for the demo (Playwright). No UI framework.

## What was actually run

- `spike/index.html` — preload + unmodified `pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2')`.
  Default device failed as described; retry with `device:'auto'` captured 1 run
  (`input_ids/attention_mask/token_type_ids [1,7]` → `last_hidden_state [1,7,384]`).
- `spike/b.html` — `attach(pipe)` on the same model, then
  `pipeline('text-generation','onnx-community/tiny-random-LlamaForCausalLM-ONNX')`,
  3 tokens: 3 runs captured with logits and KV tensors, 3 streamed tokens, tokenizer
  text→ids→tokens. A `LogitsProcessor` pushed into a `LogitsProcessorList` fired once per
  step with `logits.dims [1,128256]` (verified from the console; the page version had the
  empty-constructor bug noted above).
- Serve with `.claude/launch.json` → `spike` (python http.server on :5178). Model files
  come from the HF Hub on first load and are cached by the browser Cache API after that.

---

## Reviewer sign-off

- [x] The verdict matches my mental model of the system
- [x] The file list is complete — nothing important is missing
- [x] The risks are real risks, not filler
- [x] No question was answered by guessing

**Reviewed by:** Pratik Amin **Date:** 2026-09-10
