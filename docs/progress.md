# Progress log

> **Append-only. Written by the loop, not by you.**
>
> This is the loop's institutional memory: each iteration starts with a fresh
> context window and knows nothing except what is on disk. This file is how
> iteration 40 learns what iteration 3 discovered the hard way.
>
> Your only job here is to *delete wrong lessons*. A false lesson in this file
> gets re-read and re-applied on every subsequent iteration, so a single bad
> entry compounds. Skim it every ten iterations or so.
>
> Format, one entry per iteration:
>
> ```
> ## <iso date> — story <id>
> - Learned: <convention, gotcha, or dead end — something the next iteration needs>
> - Watch out: <trap that cost time>
> ```

<!-- entries appended below this line -->

## 2026-09-10 — story 1
- Learned: `typescript` latest on npm is 7.0.2 (the native port) but `typescript-eslint@8.70.0` peers on `typescript >=4.8.4 <6.1.0`, so the pin is `typescript@6.0.3`. TS 6.0 also refuses to emit without an explicit `rootDir` (TS5011: "common source directory is ./src ... rootDir must be explicitly set"); `tsconfig.build.json` sets `rootDir: "src"` so declarations land in `dist/index.d.ts`, not `dist/src/index.d.ts`. Keep it when adding `preload.ts`/`worker.ts` entries.
- Watch out: `vitest@5.0.0` needs Node `^22.12.0 || ^24 || >=26` (v22.14.0 via nvm at `~/.nvm/versions/node/v22.14.0/bin` is fine; a Node 20 shell will fail install). Full pins: `@eslint/js@10.0.1`, `eslint@10.10.0`, `typescript-eslint@8.70.0`, `vite@8.3.0`, `vitest@5.0.0`, `happy-dom@20.14.3`, `@playwright/test@1.63.0`, `@types/node@22.20.2`, `@huggingface/transformers@4.2.0` (its `onnxruntime-web` dep is `1.26.0-dev.20260416-b7804b056c` — that is the CDN pin Phase 6 will derive).

## 2026-09-10 — story 2
- Learned: over a `loopbackPair` a local `emit` lands in that bus's history synchronously while the relayed copy arrives a microtask later, so two buses that both emit end up with the same events in *different* orders. Tests must compare histories as sets (sort by `t`), never as ordered arrays, and must `await` a macrotask (`setTimeout 0`) before asserting on the peer. The real `generate` loop (`modeling_utils.js:943-1031`) passes `bigint[][]` ids to logits processors and `[bigint][]` to `streamer.put`; `test/fakes.ts` reproduces that exactly, so wrappers in story 4 must `Number()` them before emitting.
- Watch out: ESLint 10's `js.configs.recommended` enables `no-useless-assignment`, which errors on the `t: (t += 1)` idiom when it is the last use of `t` — write `t + 1` for the final one. TS 6 will not let you cast a discriminated union member directly (`res as { data: X }` is TS2352); narrow with `'data' in res` instead.

## 2026-09-11 — story 3
- Learned: `'data' in obj` is the only safe duck-type test for a tensor — any property *read* of `.data` on a `gpu-buffer`/`ml-tensor` tensor throws (and the fake counts it in `dataReads`), so `isTensorLike`, `summarizeTensor` and `toCloneSafe` must gate on `isCpuResident` before touching it. `TensorStore` ids are nominal-byte LRU: a tensor bigger than `maxBytes` is summarised, handed an id, and evicted in the same `put`, so `read()` answers `evicted` and the panel still gets a head. `Blob`/`URL` are Node 22 globals, so `previewInput` can reference them and the `document|window` grep stays clean.
- Watch out: `toCloneSafe` truncation returns a *different shape* (`{ $truncated, values }`) for arrays over 64 items, not an array with a flag — arrays with extra properties do not survive `JSON.stringify`. Cycle detection uses an ancestor stack, not a global seen-set, so shared acyclic references (the same tensor in `result` and in `outputs`) are not mislabelled `[cycle]`.
