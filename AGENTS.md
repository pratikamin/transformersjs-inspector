# transformersjs-inspector

## What this is

A zero-runtime-dependency TypeScript library: a "Network tab" for local Transformers.js (4.x)
model calls. `attach(pipe)` wraps the instances the host already holds (`model.sessions[name].run`,
`tokenizer._call`, `model.generate`, `pipe._call`), emits structured-clone-safe events onto an
`InspectorBus`, and renders them in a shadow-DOM panel. Secondary entries: `preload` (zero-touch
via `globalThis[Symbol.for('onnxruntime')]`, needs `device: 'auto'`) and `worker` (`postMessage`).

## Stack

TypeScript (strict, ESM, `moduleResolution: Bundler`), Vite lib build, Vitest (node + happy-dom),
ESLint flat config + typescript-eslint, Playwright (Chromium) e2e against the Vite demo. Node >= 20
(nvm: `~/.nvm/versions/node/v22.14.0/bin`). `@huggingface/transformers@4.2.0` is a dev dep for
types and the ORT version pin only.

## Commands

```bash
npm install
npm run dev          # Vite demo on :5173 (vite.demo.config.ts)
npm run typecheck    # tsc -p tsconfig.json --noEmit
npm test             # vitest run, offline, test/**/*.test.ts only
npm run lint         # eslint .
npm run build        # vite build (dist/index.js) + tsc -p tsconfig.build.json (dist/*.d.ts)
npm run e2e          # playwright test (needs `npx playwright install chromium` once)
```

## Layout

```
src/         library: index.ts, events.ts, bus.ts, store.ts, attach.ts, preload.ts, worker.ts
src/wrap/    instance wrappers: session, tokenizer, generation, pipeline
src/panel/   vanilla-DOM panel: model (reducer), dom (h()), styles, render, panel
test/        Vitest unit tests + fakes.ts; never touches the network
demo/        Vite demo pages; Transformers.js loaded from the CDN (demo/tf.ts), not npm
e2e/         Playwright specs; persistent profile in .cache/pw-profile caches models
docs/        00-brief, 01-research, 02-plan (read-only), progress.md (append-only)
spike/       verified wrapping experiments; read-only reference
```

## Conventions

- `src/` never imports `@huggingface/transformers` or `onnxruntime-web` at runtime
  (only `src/preload.ts` imports `onnxruntime-web/webgpu`, rewritten to the CDN URL at build).
  Library-facing shapes are structural types in `src/types.ts`.
- Every emitted event must survive `structuredClone` and `JSON.stringify`: no bigint, typed
  arrays, or DOM nodes. Only request *responses* may carry typed arrays.
- Panel: shadow root + constructed stylesheet; no `innerHTML`, no inline `style=` attributes.
- `npm test` is offline. Anything needing a browser or a model is `npm run e2e`.
- `import type` is enforced (`consistent-type-imports`); `verbatimModuleSyntax` is on.
- `spike/` is read-only reference for the verified wrapping code.

## Do not

- Edit `node_modules`, `dist/`, or `dist-demo/` (generated).
- Edit `docs/00-brief.md`, `docs/01-research.md`, `docs/02-plan.md`, or anything in `spike/`.
- Add runtime dependencies or a UI framework; dev deps stay at the 10 listed in the plan.

<!-- loop-learned conventions appended below -->
- `test/fakes.ts` is the one home for offline stand-ins (tensors, sessions, tokenizer, generative model, pipeline, fixture events); extend it rather than hand-rolling fakes inside a test file.
- Never read `tensor.data` outside `src/summarize.ts`; it throws on GPU/WebNN tensors. Use `isTensorLike`, `summarizeTensor`, `headOf` and `readTensor` from there, and `TensorStore.put` when the value must be readable later.
- Transformers.js 4.x pipelines, models and tokenizers all `extend Callable`: `typeof` is `'function'`, never `'object'`. Duck-type instances with `isInstanceLike` (function or object) in `src/attach.ts`, not `isRecord`.
- e2e: `npm run e2e -- e2e/<spec>.spec.ts` (starts or reuses the Vite server on :5173). Specs import `test`/`expect`/`runTask` from `e2e/fixtures.ts`; the persistent profile `.cache/pw-profile` holds the model and CDN caches (`rm -rf` it for a cold run). Panel selectors go straight through the open shadow root of `[data-tjsi-panel]`.
- Call and run ids (`c<n>`, `r<n>`) are counted per *bus* (counters on the bus object under `Symbol.for('transformersjs-inspector.ids')`, see `src/context.ts`), never per `WrapContext`: every `attach()` makes a new context and the panel reducer folds a repeated `call:start` id into the existing row.
- `package.json` `sideEffects` must list every module imported for its side effects alone (`./src/preload.ts`, `./demo/preload-entry.ts`, `./dist/preload.js`): Vite applies the project's own field to source files in build mode and tree-shakes unlisted ones out of `build:demo`. Never let a shared chunk carry a top-level `await import(Transformers)` that must run after the preload; the bundler hoists shared-chunk imports above the entry body.
