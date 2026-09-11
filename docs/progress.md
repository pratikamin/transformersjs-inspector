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

## 2026-09-11 — story 4
- Learned: events naming an unknown `callId`/`runId` (history capped, or replay after `clear`) must open a synthetic call of *that id* rather than fall through to "newest open", otherwise later events for the same call scatter across rows; a late `call:start` upgrades the placeholder in place (`change: 'updated'`, not `'new'`). `CallView.n` is the stable 1-based sequence number for `#n`; array index is wrong once the cap drops rows.
- Watch out: synthetic ids use the `s` prefix; wrappers must keep `c`/`r`/`t` so they never collide. `run:end` resolves its call through `byRun` *before* trusting `ev.callId` — on the preload path `callId` is always null and "newest open" picks the wrong row when runs overlap.

## 2026-09-11 — story 7
- Learned: restoring a wrapped method must know whether the original was an *own* property. ORT's `InferenceSession.run` and the tokenizer `Callable._call` live on the prototype, so `target.run = original` after detach would leave a shadowing own property behind; `replaceMethod` in `src/wrap/session.ts` records `hasOwnProperty` at wrap time and `delete`s on restore. Reuse it for `generate` and `pipe._call` in stories 8-9. `WRAPPED`'s value is the installed wrapper, so restore can tell if someone else replaced the method since.
- Watch out: in zsh a bare `echo ===` fails ("not found", `=` triggers equals-expansion) and `echo "$?"` after `cmd | tail` reports tail's status; use `set -o pipefail` and quoted separators or the "all green" check lies.

## 2026-09-11 — story 5
- Learned: `fixtureEvents()` is 5 events for c1 (start, tokenize, run:start, run:end, result) and c2 begins at index 5; slicing at 6 silently pulls c2's `call:start` in. happy-dom 20 supports `CSSStyleSheet.replaceSync` and `adoptedStyleSheets`, so the constructed-sheet path is what the tests exercise; the `<style>` fallback is covered with a plain object lacking `adoptedStyleSheets`. `RenderContext` is `{ bus }` so story 6 can `request('tensor')` without changing render signatures; `data-call` lives on the `.summary` element, details carry `data-details=<id>`.
- Watch out: the `grep -rn "innerHTML\|style="` gate also matches *comments* — never write those tokens in doc comments under `src/panel`. Vitest 5 swallows `console.warn` inside happy-dom tests; throw an `Error` carrying the string to inspect DOM state.

## 2026-09-11 — story 8
- Learned: in 4.2.0 `LogitsProcessorList.extend(items)` spreads any iterable and calls each entry as a function, so a plain array of callables is accepted; the `transformers.LogitsProcessorList` escape hatch exists but is untested against the real library (story 11's e2e is the arbiter). Read logits through `cpuData()` in `src/summarize.ts`, never `tensor.data` directly. `fakeGenerativeModel` puts ~11.7k of 128,256 logits at −1, so the picked token's softmax share is ≈0.30 — assert exact `exp(x−max)/Σ` values, not "peak ≈ 1".
- Watch out: `${PIPESTATUS[0]}` is bash-only; in zsh it prints blank and reads as success. ESLint `no-unused-vars` (after-used) flags an `_`-prefixed *sole* parameter — drop it instead.

## 2026-09-11 — story 6
- Learned: `fixtureEvents()` carries 5 top-k entries per step (the wrapper default is 10); tests read the count from the `logits` event instead of hardcoding either. Re-rendering details on `updated` drops already-loaded tensor values (wholesale re-render, as story 5 built it); a second click reloads — documented, not fixed. `bus.dispatch` wraps handlers in a Promise, so a synchronously throwing handler rejects the request the same way a rejected promise does.
- Watch out: `panel.ts` declares a local `const empty`, so importing `empty` from `./dom` there shadows it and only fails at typecheck. The sandbox refuses inline python heredocs inside a worktree; write the script to the scratchpad and run `python3 <abs path>`.

## 2026-09-11 — story 9
- Learned: `attach()` only wraps what the pipeline *instance* reaches through `pipe.model.generate` / `pipe.tokenizer._call`; a fake (or host) that captured the function in a closure silently bypasses the wrapper — `test/fakes.ts` text-generation fake was fixed to go through `pipe.model.generate` like the real `TextGenerationPipeline._call`. `store.attachTo(bus)` is once per (bus, store) pair and never unregistered on detach because other attached pipelines share it. `ensurePanel` remounts if the cached host was destroyed (`host.isConnected`).
- Watch out: the word `document` in a doc comment trips the `src/` grep gate the same way `innerHTML` did for the panel — write "DOM" instead. `assertPipeline` must accept function-shaped pipelines (`Callable`), i.e. `typeof === 'function' || object`. In orchestration scripts, `grep | wc -l` under `set -o pipefail` fails the chain when grep finds nothing — use `grep -c` or `|| true`.

## 2026-09-11 — story 10
- Learned: in Transformers.js 4.2.0 `pipe`, `pipe.model` *and* `pipe.tokenizer` all `extend Callable`, so `typeof` is `'function'` for each; `assertPipeline`'s `isRecord(model)` threw on every real pipeline and `isRecord(tokenizer)` silently skipped the tokenizer wrapper (no chips). Both now use `isInstanceLike` (function or object) in `src/attach.ts`, with a fake-based regression test in `test/attach.test.ts`. `demo/main.ts` runs only after `demo/tf.ts` finishes its top-level `await import(CDN)`; `page.goto` resolves before that on a cold profile, so a click on a not-yet-wired Run button is lost *silently* and the test waits the full timeout with `data-status="idle"` — that, not slow downloads, was the "4-minute cold run". The Run buttons ship `disabled` in the HTML, `main.ts` enables them, and `runTask` waits `toBeEnabled` before clicking. After one run `.cache/pw-profile` is 75 MB (45 MB Cache API `transformers-cache` for the model, 27 MB HTTP cache for the CDN module and ORT wasm); measured cold 3.6 s vs warm 1.2 s for the spec on this network. `consistent-type-imports` forbids the plan's inline `typeof import('@huggingface/transformers')`; `import type * as Transformers` + `typeof Transformers` is the equivalent. Playwright pierces the panel's open shadow root with plain CSS; `section.section` filtered by an exact-`h3` `has:` locator selects the Tokenizer/Session runs/Result sections. `E2E_DEBUG=1 npm run e2e -- …` echoes the page console and every Hub/CDN response with timestamps.
- Watch out: Playwright wipes `test-results/` at the start of each run, so read a failure's `error-context.md` *before* rerunning. A `nohup npm run dev &` from the tool shell dies with the call; the tool's background mode keeps it alive, and `webServer.reuseExistingServer` then picks it up. `du` of the profile right after a failed run tells you whether anything was downloaded at all (2.6 MB = only the CDN module).
