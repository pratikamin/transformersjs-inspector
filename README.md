# transformersjs-inspector

A "Network tab" for local Transformers.js (4.x) model calls: `attach(pipe)` wraps the pipeline
the page already holds and renders every call (tokenizer ids, per-session input/output
tensors, per-step logits, the decoded result) in a shadow-DOM panel. No runtime dependencies.

Full usage docs land with the packaging phase; until then `docs/02-plan.md` is the reference.

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
