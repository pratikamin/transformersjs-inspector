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
