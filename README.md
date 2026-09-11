# transformersjs-inspector

A "Network tab" for local Transformers.js (4.x) model calls: `attach(pipe)` wraps the pipeline
the page already holds and renders every call (tokenizer ids, per-session input/output
tensors, per-step logits, the decoded result) in a shadow-DOM panel. No runtime dependencies.

Full usage docs land with the packaging phase; until then `docs/02-plan.md` is the reference.

## Zero-touch preload

`attach(pipe)` is the headline API and always works. The secondary entry, `preload`, needs no
change to the host's pipeline code at all: it installs a shim of onnxruntime-web on
`globalThis[Symbol.for('onnxruntime')]`, the hook Transformers.js 4.x checks when it
evaluates, whose `InferenceSession.create` wraps every session's `run`. Load it as a module
script **before** the script that loads Transformers.js:

```html
<script type="module" src="https://cdn.jsdelivr.net/npm/transformersjs-inspector/dist/preload.js"></script>
<script type="module">
  import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
  const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { device: 'auto' });
  await pipe('The inspector sees everything.');
  // The panel (bottom right) shows one row per session run, labelled `direct · session#1 → last_hidden_state`.
</script>
```

Module scripts execute in document order, so the shim is in place before Transformers.js
picks its runtime; the bundled preload entry (`demo/preload.html`) behaves the same way. See
`e2e/preload.spec.ts` for the page that proves it.

Three caveats, all from `docs/01-research.md`:

- **`device: 'auto'` is required.** With the symbol set, Transformers.js takes its
  "custom runtime" branch, whose supported-device list is empty, so the default `'wasm'`
  throws `Unsupported device: "wasm". Should be one of: .` before any session is created;
  `'auto'` returns the (empty) list as-is and ORT then picks its own default. The upstream
  fix is a few lines in `src/backends/onnx.js` — populate `supportedDevices` from the injected
  module's `env`, or treat an injected web runtime like the web branch — and would make the
  preload unconditional; it should be filed against transformers.js.
- **ORT version coupling.** The preload does not bundle onnxruntime-web; `dist/preload.js`
  statically imports `https://cdn.jsdelivr.net/npm/onnxruntime-web@<version>/dist/ort.webgpu.bundle.min.mjs`,
  where `<version>` is derived at build time from the `onnxruntime-web` dependency of the
  installed `@huggingface/transformers` (`1.26.0-dev.20260416-b7804b056c` for 4.2.0). Use the
  preload only with that Transformers.js minor; a mismatch means session options may not line
  up. It is a second ~2 MB download (JS + WASM) that `attach()` never needs.
- **Only the session boundary is visible.** The preload has no pipeline reference, so rows
  carry no input text, tokenizer ids, per-step logits or decoded result; those need `attach()`.
  Both share one bus and one panel, so a page may use both.

## Development

```bash
npm install
npx playwright install chromium   # once; the e2e suite drives headless Chromium

npm run dev          # Vite demo on http://localhost:5173 (Transformers.js from the jsDelivr CDN)
npm run typecheck    # tsc --noEmit
npm test             # Vitest, offline (test/**/*.test.ts)
npm run lint         # ESLint
npm run build        # dist/index.js + dist/*.d.ts
npm run e2e          # Playwright against the demo; e.g. npm run e2e -- e2e/demo.spec.ts
```

`npm run e2e` starts the demo server itself (or reuses one on :5173). The Chromium profile
under `.cache/pw-profile` persists between runs, so the fixture models
(`Xenova/all-MiniLM-L6-v2`, `onnx-community/tiny-random-LlamaForCausalLM-ONNX`) and the CDN
module are downloaded once and served from the browser cache afterwards; delete that
directory to force a fresh download.
