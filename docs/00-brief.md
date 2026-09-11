# Brief: transformersjs-inspector

> Drafted by the agent from the 2026-09-10 research conversation; reviewed and accepted
> by Pratik the same day. If "Done looks like" or "Explicitly not doing" turns out wrong,
> fix it here, not downstream.

## Problem

When a web page runs a machine-learning model locally, everything between the user's
text or image and the model's answer is invisible. Developers debugging a page, and
people trying to understand what a page actually sends to a model, have no way to see
the exact inputs, their shapes, or the raw outputs without writing ad-hoc logging.
There is no equivalent of the browser's Network tab for local model calls.

## Done looks like

- [ ] A page that already uses Transformers.js can add one script tag (or one `attach(pipeline)`
      call) and a panel appears showing every model call as a row: the raw input
      (text / image / audio), the tokenizer output (ids and token strings), each named
      input tensor (name, dtype, shape, preview of values), each named output tensor, and the
      pipeline's decoded result.
- [ ] For text generation, the panel shows per-step logits for the generated token
      (top-k tokens with probabilities) and the streamed token ids.
- [ ] The panel does not change the page's inference timing by more than ~5% when
      collapsed, and a tensor's full data is only read when the user expands that row.
- [ ] A demo page in the repo exercises text classification, feature extraction, and
      text generation and is what the README screenshots come from.

## Explicitly not doing

- Per-layer internal activations. ONNX Runtime only returns declared graph outputs; seeing
  inside the model means rewriting the ONNX graph. That is a v2 with its own brief.
- Support for TensorFlow.js, WebLLM, MediaPipe, or raw onnxruntime-web sites. Transformers.js
  only, version 4.x only.
- A browser extension. The deliverable is a library the page opts into. A DevTools panel
  built on the same event stream is a later project.
- Persisting or uploading captured data anywhere. Everything stays in the tab.
- Editing or replaying inputs. Read-only for v1.

## Constraints

| | |
|---|---|
| Language / runtime | TypeScript, browser ESM. Zero runtime deps besides an optional peer on `@huggingface/transformers`. |
| Framework / key deps | No UI framework; the panel is vanilla DOM in a shadow root so it cannot collide with host CSS. Vite for the demo, Vitest for tests. |
| Deploy target | npm package + a static demo page (GitHub Pages or the pratikamin.com site). |
| Data store | none |
| Must not change | Never patch `@huggingface/transformers` files on disk. All hooks are runtime wrapping of instances the library already exposes (`pipe.model.sessions`, `pipe.tokenizer`, `LogitsProcessor`, `TextStreamer`) or the documented `globalThis[Symbol.for('onnxruntime')]` injection point. |
| Hard deadline | none |

## Verification commands

```bash
# typecheck
npm run typecheck

# test
npm test

# lint
npm run lint

# build
npm run build
```

## Open questions (answered in 01-research.md)

1. Can the library observe an unmodified page with no code changes ("zero-touch"), or does the host have to call `attach()`?
2. What exactly is visible at the `session.run` boundary for encoder, encoder-decoder, and decoder-only models, and what does it cost to read it?
3. How do we get text→ids and per-step logits without private APIs?
4. What happens when the host runs the pipeline in a Web Worker, or on WebGPU where tensors may live on the GPU?
5. What is the smallest dependency set, and which Transformers.js version do we pin to?
