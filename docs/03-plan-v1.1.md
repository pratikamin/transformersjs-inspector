# Plan: transformersjs-inspector v1.1 (0.2.0)

> Written 2026-09-11 against `main` at v0.1.1 (all suites green). Same rule as `02-plan.md`:
> if you cannot predict the diff from this document, send it back. `00-brief.md` and
> `02-plan.md` are read-only; where this plan lifts two items from their fence (export,
> replay) it says so and the README is updated to match. Everything else in the fence holds.

## Approach

v1.1 adds five things to the panel without changing what the library *is*: decoded token
text next to the raw vocab strings (A), a dark theme (B), a dock option and drag-to-resize
(C), media previews at the input boundary plus tensor-as-image previews (D), and export of the
captured events and replay of a captured call (E). The one design rule tying them together:
**everything new flows through the existing bus.** New information is optional fields on
existing events (all clone-safe, all backward compatible); anything that needs the capture
side (tensor bytes for an image, re-running a call) is a `bus.request` answered where
`attach()` ran, so every feature works unchanged across the worker bridge; and the panel
stays a pure consumer of events and responses. Nothing is persisted: size, theme and dock
live for the panel's lifetime; export is a user-initiated download.

Rejected alternatives, and why each loses:

- **SVG polyline vs canvas for the input waveform.** SVG: the preview is 200 min/max pairs
  already in the event, an inline `<svg>` scales with the row, needs no `getContext` (which
  happy-dom returns `null` for), and is testable by reading the `d` attribute. Canvas is kept
  for D2 only, where the pixels come from a lazily fetched typed array and can be 512 px wide.
- **Data URL vs Blob URL for the export download.** Blob URL: a data URL puts the whole JSON
  (up to 500 events, tens of MB with previews) into an attribute string; Blob + `URL.createObjectURL`
  + a temporary `<a download>` streams it and is what Playwright's `download` event intercepts.
  `navigator.clipboard.writeText` is the fallback where object URLs are unavailable.
- **Replay registry in the panel vs in `attach()`.** In `attach()`: the panel only ever sees
  clone-safe previews, never the raw `args` (a `Float32Array` of audio, a `RawImage`), and in
  the worker case the panel is in a different realm from the pipeline. The registry lives in the
  `WrapContext`'s realm, is bounded (200 entries, strong refs, dropped on `detach()`), and the
  panel only sends `bus.request('replay', { callId })`.
- **Async `previewInput` to allow `OffscreenCanvas.convertToBlob` thumbnails in workers.**
  Rejected: `call:start` is emitted synchronously at the top of `pipe._call`, and making it
  async would reorder events relative to `tokenize`/`run:start`. Thumbnails use a `<canvas>`
  element + `toDataURL` (sync) when `document` exists; in a worker the image preview stays
  metadata-only. Documented.
- **Including loaded tensor values in the export.** Deferred: `exportEvents(bus)` stays
  synchronous and JSON-only; tensor bytes would need an async fan-out of `request('tensor')`
  and a typed-array-to-array encoding that dwarfs the events. Listed under "still out".
- **A visible `␣` glyph for leading spaces in token strings.** Rejected because it changes
  copy/paste. Leading/trailing whitespace runs are wrapped in `<span class="ws">` (a tinted
  background under the real space character), so the text stays intact.

Conventions that hold for the whole plan (from `AGENTS.md`): no runtime dependencies; `src/`
never imports `@huggingface/transformers`; every event survives `structuredClone` and
`JSON.stringify`; no `innerHTML` and no inline style attributes in `src/panel` (the gate is
`grep -rn "innerHTML\|style=" src/panel` and it matches comments too, so never write that
literal even in prose; CSSOM property writes like `el.style.width = …` are fine); `tensor.data`
is read only in `src/summarize.ts` (and `idsFrom` in the tokenizer wrapper); `npm test` is
offline; fakes live in `test/fakes.ts`.

## Event schema changes (`src/events.ts`)

Every field below is optional or a new request name; a v0.1 consumer ignores them and a v0.1
producer still validates. `isInspectorEvent` checks each optional field only when present.

```ts
// A
export type TopKEntry = { id: number; token: string | null; logit: number; prob: number; raw?: string | null };
//   token = tokenizer-decoded text (was the vocab string); raw = the vocab string (id_to_token)
tokenize: { …; tokens: (string | null)[][]; raw?: (string | null)[][] }   // tokens decoded, raw = vocab strings
token:    { …; text: string | null; raw?: string | null }                  // text decoded, raw = vocab strings joined

// D1
| { kind: 'image'; width?; height?; channels?; src?; thumb?: string }     // thumb: data:image/jpeg… ≤ 32 768 chars
| { kind: 'audio'; samples; sampleRate?; duration?: number; peaks?: number[] }
//   duration in seconds (3 dp) when sampleRate is known; peaks = [min0, max0, min1, max1, …] for
//   WAVEFORM_BUCKETS = 200 buckets (≤ 400 numbers, 3 dp, in [-1, 1]) → ≤ ~3 KB of JSON

// E2
| { type: 'call:start'; …; replayOf?: string }                             // callId of the call this one re-ran

export interface RequestMap {
  tensor: { req: { id: string }; res: TensorData };
  replay: { req: { callId: string }; res: ReplayResult };                  // E2
}
export type ReplayResult = { ok: true; callId: string } | { ok: false; error: string };

// E1 (src/export.ts)
export interface InspectorExport { version: string; exportedAt: string; events: InspectorEvent[] }
```

Byte budget per event: a `call:start` with an image thumbnail is ≤ 32 KiB (typical JPEG at
96 px, quality 0.75: 3–6 KB); with an audio waveform ≤ 3 KB. Everything else is unchanged.

## Phases

### Phase A: Decoded token strings

`tokens`/`token`/`text` become the tokenizer-decoded text (`decode([id], { skip_special_tokens: false,
clean_up_tokenization_spaces: false })`); the vocab string moves to `raw` and to the `title`
attribute. Caveat, documented in the README: decoding one id at a time means a Metaspace
(`▁word`) or WordPiece (`##ing`) token loses its leading-space/continuation marker in `token`
(`word`, `ing`); `raw` keeps it.

| File | Change |
|---|---|
| `src/types.ts` | `TokenizerLike.decode?(ids: number[], opts?: { skip_special_tokens?: boolean; clean_up_tokenization_spaces?: boolean }): string`. |
| `src/context.ts` | `tokenStrings(id): { raw: string \| null; text: string \| null }`: `raw` = `_tokenizer.id_to_token(id)` (else `null`), `text` = `decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false })` (else `raw`). Never throws. Memoised in a `Map<number, …>` cleared when it reaches `TOKEN_CACHE_MAX = 4096` (generation calls it `topK` times per step). `tokenToString(id)` stays as `tokenStrings(id).text` for callers that only need text. |
| `src/wrap/tokenizer.ts` | `tokenize` gains `raw: ids.map(row => row.map(id => tokenStrings(id).raw))`; `tokens` uses `.text`. |
| `src/wrap/generation.ts` | `emitLogits`: each entry `{ …e, token: s.text, raw: s.raw }`. `emitToken`: `text` joins `.text`, `raw` joins `.raw` (both `null` when every part is `null`). |
| `src/events.ts` | Schema above; `isInspectorEvent` accepts `raw` on `tokenize` (rows of string/null), `token` (string/null) and top-k entries. |
| `src/panel/render.ts` | `tokenText(s: string \| null): Child[]` — `'∅'` for null; otherwise `/^(\s*)([\s\S]*?)(\s*)$/` and non-empty groups 1 and 3 become `span.ws`. Chips: `chip-str` = `tokenText(text)`, `title` = `` `id ${id} · raw ${raw ?? '∅'}` ``. Top-k `td.tok` = `tokenText(e.token)`, `title` = `e.raw ?? ''`. Step head keeps `"${step.text}"`. |
| `src/panel/styles.ts` | One rule inserted directly after `.chip-str`: `.ws { background: #ddf4ff; border-radius: 2px; }` (B recolours it with a token). |
| `test/fakes.ts` | `SEED_VOCAB` gains `'##ing': 2075` and `'Ġfilm': 2143` (decode-only entries). `fakeTokenizer().decode(ids, opts?)`: with `opts.skip_special_tokens === false` specials are kept; with `clean_up_tokenization_spaces === false` single ids decode like the real decoders (`##ing` → `ing`, `Ġfilm` → ` film`, `[CLS]` → `[CLS]`); default behaviour (skip specials, join with spaces) unchanged so `generated_text` fixtures hold. `fixtureEvents()`: `tokenize` gets `raw` (= tokens), every `TopKEntry` gets `raw`, every `token` gets `raw`. |
| `test/wrap-tokenizer.test.ts`, `test/generation.test.ts`, `test/attach.test.ts` | Assert `raw` alongside `tokens`/`token`/`text`; a tokenizer without `decode` yields `text === raw`; a `decode` that throws yields `raw` only; `##ing` → `{ raw: '##ing', text: 'ing' }`; `Ġfilm` → `{ raw: 'Ġfilm', text: ' film' }`. |
| `test/panel.test.ts` | Chip for `[CLS]` shows `[CLS]` with `title` `id 101 · raw [CLS]`; a chip for `' film'` renders one `span.ws` containing a single space and `textContent === ' film'`; top-k `td.tok` `title` equals the entry's `raw`. |
| `e2e/demo.spec.ts` | The `[CLS]` chip's `title` contains `raw [CLS]`. |
| `README.md` | Events table: `tokens` decoded, `raw` vocab strings; the single-token decode caveat. |

**Verify**

```bash
npm test -- test/wrap-tokenizer.test.ts test/generation.test.ts test/attach.test.ts test/panel.test.ts && npm run typecheck && npm run lint && npm run e2e -- e2e/demo.spec.ts e2e/generation.spec.ts
```

**Done when:** the fixture and real (`all-MiniLM`) chips show decoded text with the vocab
string on hover, every event still passes `structuredClone`, and `grep -rn "innerHTML\|style=" src/panel` is empty.

### Phase B: Dark theme

| File | Change |
|---|---|
| `src/panel/styles.ts` | Every colour becomes a custom property on `:host`: `--tjsi-fg #1f2328`, `--tjsi-muted #57606a`, `--tjsi-bg #ffffff`, `--tjsi-bg-alt #f6f8fa` (header, `th`, hover), `--tjsi-bg-detail #fbfcfd`, `--tjsi-border #d0d7de`, `--tjsi-border-soft #eaeef2`, `--tjsi-accent #0969da`, `--tjsi-on-accent #ffffff`, `--tjsi-ok #1a7f37`, `--tjsi-warn #bf8700`, `--tjsi-err #cf222e`, `--tjsi-picked #dafbe1`, `--tjsi-ws #ddf4ff`, `--tjsi-shadow rgba(31,35,40,.18)`. `const DARK_VARS = '--tjsi-fg: #e6edf3; --tjsi-muted: #8b949e; --tjsi-bg: #0d1117; --tjsi-bg-alt: #161b22; --tjsi-bg-detail: #10151c; --tjsi-border: #30363d; --tjsi-border-soft: #21262d; --tjsi-accent: #388bfd; --tjsi-on-accent: #ffffff; --tjsi-ok: #3fb950; --tjsi-warn: #d29922; --tjsi-err: #f85149; --tjsi-picked: #12361f; --tjsi-ws: #1f3a5a; --tjsi-shadow: rgba(0,0,0,.6); color-scheme: dark;'` interpolated twice, appended at the end of `PANEL_CSS`: `@media (prefers-color-scheme: dark) { :host(:not([data-theme="light"])) { ${DARK_VARS} } }` and `:host([data-theme="dark"]) { ${DARK_VARS} }`. `:host` keeps `color-scheme: light`. |
| `src/panel/panel.ts` | `PanelOptions.theme?: 'auto' \| 'light' \| 'dark'` (default `'auto'`); `host.dataset.theme = opts.theme ?? 'auto'`. |
| `demo/main.ts` | `panelOptionsFromQuery(): PanelOptions` reads `?theme=` (and, in C, `?dock=`); `attach(p, { label, panel: panelOptionsFromQuery() })`. |
| `e2e/screenshots.spec.ts` | Third test: `page.goto('/?theme=dark')`, run text generation, expand, assert `getComputedStyle(.panel).backgroundColor` is `rgb(13, 17, 23)`, `shoot(…, 'panel-dark.png')`. |
| `docs/img/panel-dark.png` | Committed. |
| `test/panel.test.ts` | `data-theme` is `auto` by default, `dark` with the option; `PANEL_CSS` contains `prefers-color-scheme: dark`, `:host([data-theme="dark"])`, `color-scheme: dark`, and no hex colour outside a `--tjsi-` declaration (regex over the sheet with the two blocks stripped). |
| `README.md` | `PanelOptions` table gains `theme`; a "Dark theme" line with the new image. |

**Verify**

```bash
npm test -- test/panel.test.ts && npm run typecheck && npm run lint && npm run screenshots
```

**Done when:** `docs/img/panel-dark.png` exists, the panel follows `prefers-color-scheme`
under `theme: 'auto'` and the explicit option wins over it in both directions.

### Phase C: Dock option and drag to resize

| File | Change |
|---|---|
| `src/panel/fit.ts` | `export type Dock = 'bottom-right' \| 'bottom-left' \| 'top-right' \| 'top-left'`; `DEFAULT_DOCK = 'bottom-right'`. `computeFit(anchorX, anchorY, vp, dock = DEFAULT_DOCK, margin = FIT_MARGIN)`: `fitAxis` gains a `side: 'near' \| 'far'` argument; `'far'` is today's logic, `'near'` mirrors it (anchor below `lo` is pushed to `lo`; `available = hi - anchor`; below `wanted` the anchor moves inward by the shortfall). x is `'far'` for `*-right`, y is `'far'` for `bottom-*`. `fitElement(el, vp, dock)` reads the anchor corner named by `dock` from the rect. Existing call sites and tests keep working (defaults). |
| `src/panel/resize.ts` | New. `MIN_WIDTH`/`MIN_HEIGHT` re-exported from fit; `installResize(root: HTMLElement, grip: HTMLElement, opts: { dock: () => Dock; onResize(): void }): () => void`. `pointerdown` on the grip: record `start = { x, y, w: root.offsetWidth, h: root.offsetHeight }`, `grip.setPointerCapture?.(e.pointerId)`; `pointermove` (while a drag is active): `dw = dock ends in 'right' ? start.x - e.clientX : e.clientX - start.x`, `dh = dock starts with 'bottom' ? start.y - e.clientY : e.clientY - start.y`; `root.style.width = clamp(start.w + dw, MIN_WIDTH, capOf(root.style.maxWidth)) + 'px'`, same for height; `opts.onResize()`; `pointerup`/`pointercancel` end the drag; `dblclick` on the grip clears `root.style.width`/`height` and calls `onResize()`. Returns a disposer removing every listener. |
| `src/panel/styles.ts` | `:host([data-dock="bottom-left"]) { right: auto; left: 16px; }`, `:host([data-dock="top-right"]) { bottom: auto; top: 16px; }`, `:host([data-dock="top-left"]) { right: auto; bottom: auto; left: 16px; top: 16px; }`. `.panel { position: relative; }`, `.body { flex: 1 1 auto; min-height: 0; }` (so an explicit height scrolls the body). `.grip { position: absolute; width: 16px; height: 16px; touch-action: none; cursor: nwse-resize; }` placed at the corner opposite the anchor: default `top: 0; left: 0`; `bottom-left` → `top: 0; right: 0; cursor: nesw-resize`; `top-right` → `bottom: 0; left: 0; cursor: nesw-resize`; `top-left` → `bottom: 0; right: 0`. `.panel.closed .grip { display: none; }`. |
| `src/panel/panel.ts` | `PanelOptions.dock?: Dock` (default `'bottom-right'`); `host.dataset.dock`; `const grip = h('div', { class: 'grip', data: { grip: '' }, title: 'drag to resize · double-click to reset' })` appended to `root`; `fit()` passes the dock; `installResize(root, grip, { dock: () => dock, onResize: fit })`, disposed in `destroy()`. |
| `demo/main.ts` | `panelOptionsFromQuery()` also reads `?dock=`. |
| `test/panel-fit.test.ts` | `computeFit` for each dock: `top-left` anchor at `(16, 16)` caps to `vp - 2·margin` with no shift; `top-left` anchor at `(-50, -50)` shifts by `+66, +66`; `bottom-left` near the right edge is capped to the space to its right; `top-right` below the bottom edge shrinks to the space below it. `fitElement` with `dock: 'top-left'` reads `rect.left/top`. |
| `test/panel-resize.test.ts` | New, `happy-dom`. Mount with `{ open: true }`; stub `root.getBoundingClientRect` and `offsetWidth/Height` (`Object.defineProperty`) to 560×400 at the bottom-right of a 1280×900 viewport; dispatch `new PointerEvent('pointerdown', { clientX: 700, clientY: 500, pointerId: 1, bubbles: true })` on `[data-grip]`, `pointermove` to `(600, 420)`, `pointerup` → `root.style.width === '660px'`, `height === '480px'`; a move to `(-1000, -1000)` clamps to `root.style.maxWidth`/`maxHeight`; a move to `(1500, 1500)` clamps to `MIN_WIDTH`×`MIN_HEIGHT`; `dblclick` resets both to `''`; with `dock: 'top-left'` the same drag *down-right* grows the panel; `destroy()` removes the listeners (a later `pointerdown` changes nothing). |
| `e2e/resize.spec.ts` | New. Run feature extraction, open the panel; `box0 = panel.boundingBox()`; `page.mouse` drag from the grip centre by `(-120, -80)`; `box1.width ≈ box0.width + 120` and `box1.height ≈ box0.height + 80` (±2), the box stays inside `1280×900`; `dblclick` the grip → width back to `560`. Second test on `/?dock=top-left`: host box `x ≈ 16`, `y ≈ 16`; drag by `(+100, +60)` grows it. |
| `README.md` | `PanelOptions` table gains `dock`; "Where the panel sits" describes the grip and reset, and that size is not persisted. |

**Verify**

```bash
npm test -- test/panel-fit.test.ts test/panel-resize.test.ts test/panel.test.ts && npm run typecheck && npm run lint && npm run e2e -- e2e/resize.spec.ts && grep -rn "innerHTML\|style=" src/panel; test $? -eq 1
```

**Done when:** all four docks keep the panel on screen under the existing fit tests, a real
drag in Chromium changes the size and double-click restores it.

### Phase D: Media previews

D1 enriches the `call:start` preview at capture time; D2 renders tensors as images on demand
in the panel. Demo/e2e use **image classification with `onnx-community/mobilenet_v2_1.0_224`**
(`model_quantized.onnx`, 3.7 MB, one session, ~10 ms per run on wasm; `Xenova/whisper-tiny.en`
is 10.1 + 30.7 MB quantized and needs encoder + autoregressive decoder runs and an audio
source). The image path exercises both D1 (`RawImage` thumbnail) and D2 (`pixel_values [1,3,224,224]`);
the audio path (waveform, `input_features [1,80,3000]` heatmap) is covered by unit tests with
fakes. The demo needs no binary asset: it draws a 224×224 gradient with a disc onto a `<canvas>`
and passes `tf.RawImage.fromCanvas(canvas)` (sync, 4 channels) to the pipeline.

| File | Change |
|---|---|
| `src/preview.ts` (D1) | `WAVEFORM_BUCKETS = 200`, `MAX_WAVEFORM_SAMPLES = 1 << 22` (beyond that, a stride skips samples so at most 4 M reads happen), `THUMB_MAX = 96`, `MAX_THUMB_CHARS = 32768`. `waveformOf(samples: ArrayLike<number>, buckets = WAVEFORM_BUCKETS): number[]` (one pass, interleaved min/max per bucket, `Math.round(v * 1000) / 1000`, clamped to [-1, 1]; fewer buckets than samples → one bucket per sample). `audioPreview(samples, sampleRate?)` adds `duration = round(samples / rate, 3)` and `peaks`. Audio inputs recognised: `Float32Array`/`Float64Array`; `{ audio, sampling_rate }` where `audio` is a typed array (RawAudio) — a chunked `Float32Array[]` gets `samples` = summed length and no `peaks`. `export interface CanvasLike { width: number; height: number; getContext(kind: '2d'): { createImageData(w: number, h: number): { data: Uint8ClampedArray }; putImageData(img: unknown, x: number, y: number): void } \| null; toDataURL(type?: string, quality?: number): string }`; `thumbnailOf(img: { width; height; channels?; data: ArrayLike<number> }, opts: { max?: number; createCanvas?: (w, h) => CanvasLike \| null } = {}): string \| undefined` — nearest-neighbour sample into RGBA (`channels` 1 → grey, 2 → grey+alpha, 3 → rgb, 4 → rgba; data length must equal `w·h·channels` else `undefined`), `createImageData` + `putImageData`, `toDataURL('image/jpeg', 0.75)`; `undefined` when no canvas factory (default factory: `document.createElement('canvas')` when `document` exists, else none), when `getContext` returns `null`, or when the result exceeds `MAX_THUMB_CHARS` or does not start with `data:image/`. `previewInput` is wrapped: each enrichment is its own `try/catch` (field omitted on failure); the whole function is `try/catch` → `{ kind: 'other', json: toCloneSafe(x) }` → `{ kind: 'other', json: '[unpreviewable]' }`. `Blob`/`URL`/http inputs keep `src` only (decoding is async). |
| `src/events.ts` (D1) | Schema above; validation of `peaks` (numbers), `duration` (number), `thumb` (string). |
| `src/panel/dom.ts` (D) | `svg(tag: 'svg' \| 'path' \| 'polyline' \| 'rect', attrs: Record<string, string \| number>, ...children): SVGElement` via `createElementNS` + `setAttribute` (values are computed numbers/strings, never host text). `Attrs.src?: string` applied only to `<img>` and only when it starts with `data:image/` (a remote `src` would make the panel fetch). `Attrs.width`/`height` numeric for `img`/`canvas`. |
| `src/panel/render.ts` (D1) | `renderInput` audio: meta `audio · ${samples} samples @ ${rate} Hz · ${duration} s`, then `renderWaveform(peaks)`: `svg('svg', { class: 'wave', viewBox: '0 0 200 40', preserveAspectRatio: 'none' }, svg('path', { class: 'wave-area', d }))` where `d` walks the maxes left→right and the mins right→left (`y = 20 - v * 19`), closed with `Z`. Image: `h('img', { class: 'thumb', src: input.thumb, title: … })` when `thumb` is present, then the existing meta line. `inputExcerpt` unchanged. |
| `src/panel/tensor-image.ts` (D2) | New, pure. `ImageShape = { kind: 'rgb'; layout: 'chw' \| 'hwc'; height; width } \| { kind: 'gray'; height; width }`. `imageShapeOf(dims, dtype): ImageShape \| null`: `null` for `float16`/`string`; `[1,3,H,W]`/`[3,H,W]` → rgb chw; `[1,H,W,3]` → rgb hwc; `[H,W]`, `[1,H,W]`, `[1,1,H,W]`, `[1,C,T]` (C rows, T cols) → gray; every rule needs `H, W ≥ 8` (`MIN_IMAGE_SIDE`). `MAX_IMAGE_SIDE = 512`. `rasterize(data: ArrayLike<number \| bigint>, shape, maxSide = MAX_IMAGE_SIDE): { width; height; rgba: Uint8ClampedArray; ranges: { min: number; max: number }[] }` — per-channel min/max in one pass, nearest-neighbour downsample of each axis to ≤ `maxSide`, `(v - min) / (max - min) * 255` (flat channel → 0). `describeMapping(shape, r): string` → e.g. `gray 80×3000 → 80×512 · min -1.234 · max 2.5` or `rgb chw 224×224 · R -2.118…2.64 · G … · B …`. |
| `src/panel/render.ts` (D2) | Tensor rows whose `imageShapeOf(t.dims, t.dtype)` is non-null get a second button `h('button', { class: 'btn', data: { action: 'preview', tensor: t.id } }, 'Preview')`. `renderTensorImage(cell, data: TensorData)`: error → same as `renderTensorValues`; else `rasterize`, `h('canvas', { class: 'tensor-image', width, height })`, `getContext('2d')` (`null` → a `.muted` note `canvas unavailable`), `createImageData` + `putImageData`, then `.meta` with `describeMapping`. Reuses `valuesCellFor(row)` (Preview and Load values share the cell; the last click wins). |
| `src/panel/panel.ts` (D2) | Click action `preview`: same flow as `loadValues` but ends in `renderTensorImage`. |
| `src/panel/styles.ts` (D) | `.wave { display: block; width: 100%; max-width: 400px; height: 40px; }`, `.wave-area { fill: var(--tjsi-accent); opacity: .7; }`, `img.thumb { display: block; max-width: 96px; max-height: 96px; border: 1px solid var(--tjsi-border); margin-bottom: 4px; }`, `canvas.tensor-image { display: block; max-width: 100%; image-rendering: pixelated; border: 1px solid var(--tjsi-border); }`. |
| `test/fakes.ts` (D) | `fakeCanvas(): CanvasLike & { puts: unknown[] }` returning `data:image/jpeg;base64,ZmFrZQ==`. `fakeRawImage(w, h, channels)` (gradient bytes). `fixtureMediaEvents(): InspectorEvent[]` — c3 `automatic-speech-recognition` with `{ kind: 'audio', samples: 48000, sampleRate: 16000, duration: 3, peaks }` (200 pairs), run with `input_features float32 [1, 80, 3000]` in and `last_hidden_state [1, 1500, 384]` out, result `{ text: 'hello' }`; c4 `image-classification` with `{ kind: 'image', width: 224, height: 224, channels: 4, thumb: 'data:image/jpeg;base64,…' }`, run with `pixel_values float32 [1, 3, 224, 224]` in and `logits [1, 1000]` out. Kept separate from `fixtureEvents()` so its two-call assertions stand. |
| `test/preview.test.ts` (D1) | `waveformOf`: a 1000-sample ramp gives 200 pairs, monotone, 3 dp; 8 samples give 8 pairs; a 5 M `Float32Array` completes with ≤ 400 numbers; RawAudio-shaped `{ audio, sampling_rate }` gives `duration`; chunked audio gives `samples` and no `peaks`. `thumbnailOf` with `fakeCanvas`: 1/2/3/4-channel inputs put a `96×48` image for a 200×100 source; a factory returning `null` → `undefined`; a `toDataURL` longer than `MAX_THUMB_CHARS` → `undefined`; a throwing `getContext` → `undefined`; `previewInput` on a getter that throws → `{ kind: 'other', … }`. Every preview round-trips `structuredClone`/JSON. |
| `test/tensor-image.test.ts` (D2) | Node env. Shape table (positive and negative cases incl. `[1,7,384]` → null, `[1,1,128256]` → null, `[1,2,5,16]` → null, `[1,80,3000]` → gray 80×3000, `[3,8,8]` → rgb chw). `rasterize` of a 2×2 chw tensor with known values → exact bytes and ranges; `[1,80,3000]` → `width === 512`, `height === 80`; flat channel → zeros. |
| `test/panel.test.ts` (D) | With `fixtureMediaEvents()`: c3 details render `svg.wave` with a `path[d]` starting with `M0,`; c4 renders `img.thumb[src^="data:image/jpeg"]`; `pixel_values` and `input_features` rows have `button[data-action="preview"]`, `last_hidden_state [1,7,384]` in `fixtureEvents()` does not; clicking Preview requests exactly that id and renders `canvas.tensor-image` (or the `canvas unavailable` note under happy-dom) plus a `.meta` containing `min`; an `{ error }` response renders the error. |
| `demo/index.html`, `demo/main.ts` (D3) | New `<section data-task="image-classification">` with `<canvas data-source width="224" height="224">`, Run, output (no textarea). `TASKS['image-classification']`: `load` → `tf.pipeline('image-classification', 'onnx-community/mobilenet_v2_1.0_224', { progress_callback })`; `run(pipe)` → `pipe(tf.RawImage.fromCanvas(canvas), { top_k: 3 })`; `format` → `label · score` lines. `wireSection` tolerates a missing textarea (`text = ''`). `drawSample(canvas)` paints the gradient + disc on load. |
| `e2e/fixtures.ts` (D3) | `runTask(page, task, text?)` skips the `fill` when the section has no textarea. |
| `e2e/media.spec.ts` (D3) | Run image classification; the row's Input section shows `img.thumb[src^="data:image/jpeg"]` and meta `image 224×224×4`; `pixel_values` row dims `[1, 3, 224, 224]` has a Preview button; click → `tr.values canvas.tensor-image` with `width === 224` and `.meta` containing `rgb chw`; Result `pre` contains `"label"`. |
| `README.md` | "What the panel shows": Input now includes a waveform and a thumbnail; the Preview button and its dims rules; the worker caveat (no thumbnails); limitations list updated ("Processor inputs are metadata only" → replaced); demo model list gains mobilenet. |

**Verify**

```bash
npm test -- test/preview.test.ts test/tensor-image.test.ts test/panel.test.ts && npm run typecheck && npm run lint && npm run e2e -- e2e/media.spec.ts && grep -rn "innerHTML\|style=" src/panel; test $? -eq 1
```

**Done when:** a real `RawImage` call shows a thumbnail and its `pixel_values` renders as an
image from bytes fetched through `bus.request('tensor')`; the log-mel heatmap path passes on the
`[1,80,3000]` fake; no preview code path can throw into `pipe._call`.

### Phase E: Export and replay

| File | Change |
|---|---|
| `src/export.ts` (E1) | New, no DOM. `InspectorExport` (above); `exportEvents(bus: InspectorBus): InspectorExport` → `{ version: VERSION, exportedAt: new Date().toISOString(), events: [...bus.history] }` (`VERSION` moves to `src/version.ts` to avoid an `index` cycle; `index.ts` re-exports it). `exportFileName(d = new Date())` → `transformersjs-inspector-YYYYMMDD-HHMMSS.json`. Note in the doc comment: the bus keeps `maxHistory` (500) events; raise it on the bus for longer exports. |
| `src/panel/download.ts` (E1) | New. `downloadJson(text: string, name: string): boolean` — `false` unless `URL.createObjectURL` is a function; else `Blob([text], { type: 'application/json' })`, `const a = h('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)`; `true`. `copyText(text): Promise<boolean>` — `navigator.clipboard?.writeText` guarded, `false` on rejection. |
| `src/panel/panel.ts` (E1) | Header gains `Export` (`data-action="export"`) and `Copy JSON` (`data-action="copy"`) buttons before `Clear`. `export`: `text = JSON.stringify(exportEvents(bus))`; `downloadJson(text, exportFileName())` else `await copyText(text)`; result written to `root.dataset.exportStatus` (`downloaded` \| `copied` \| `failed`). `copy`: `copyText` → `copied` \| `failed`. |
| `src/index.ts` (E1) | `export { exportEvents, exportFileName } from './export'`, `export type { InspectorExport } from './export'`, `VERSION` from `./version`. |
| `src/events.ts`, `src/bus.ts` (E2) | `RequestMap.replay`, `ReplayResult`; `call:start.replayOf?`. No bus logic change (a request with no local handler is already forwarded to transports). |
| `src/replay.ts` (E2) | New. `ReplayEntry = { pipe: PipelineLike; args: unknown[]; ctx: WrapContext; inFlight: boolean }`; `class ReplayRegistry { constructor(max = DEFAULT_MAX_CALLS /* 200, from panel/model */); record(callId, entry) /* insertion-order Map; oldest evicted past max */; get(callId); dropByPipe(pipe); get size; clear() }`. `registryFor(bus): ReplayRegistry` (`WeakMap<InspectorBus, ReplayRegistry>`, created on first use together with `bus.handle('replay', handler)`, mirroring `registerStore`). `handler({ callId })`: unknown → `{ ok: false, error: 'unknown call ' + callId }`; `inFlight` → `{ ok: false, error: 'replay of ' + callId + ' already in flight' }`; else `entry.inFlight = true; entry.ctx.pendingReplayOf = callId; p = Promise.resolve(entry.pipe._call(...entry.args))` (synchronous throw → `ok: false`), `newId = entry.ctx.lastCallId`, `p.then(noop, noop).finally(() => { entry.inFlight = false })`, return `{ ok: true, callId: newId }` **as soon as the new call has started** (a long generation must not hit the 10 s request timeout; its outcome arrives as ordinary events). |
| `src/context.ts` (E2) | `pendingReplayOf: string \| null = null` (consumed by the pipeline wrapper), `lastCallId: string \| null = null`, `replays: ReplayRegistry \| null = null`. |
| `src/wrap/pipeline.ts` (E2) | At the top of the wrapper: `const replayOf = ctx.pendingReplayOf; ctx.pendingReplayOf = null; ctx.lastCallId = callId; ctx.replays?.record(callId, { pipe, args, ctx, inFlight: false })`; `call:start` gains `...(replayOf ? { replayOf } : {})`. |
| `src/attach.ts` (E2) | `AttachOptions.replayHistory?: number` (default `DEFAULT_MAX_CALLS`; `0` disables recording, and the handler then answers `ok: false, error: 'replay disabled'` for ids it does not know). `ctx.replays = registryFor(bus)` (the registry's `max` is set on creation from the first attach's option). `detach()` also calls `ctx.replays?.dropByPipe(pipe)`. |
| `src/panel/model.ts` (E2) | `CallView.replayOf: string \| null`, set from `call:start` (also when adopting a synthetic placeholder). |
| `src/panel/render.ts` (E2) | Summary grid gains two cells before the dot: `span.replay-of` (`↻ ${replayOf}`, `data-replay-of`, empty when null) and, for non-synthetic calls, `button.btn.replay[data-action="replay"][data-call-id]` (`Replay`). |
| `src/panel/styles.ts` (E2) | `.summary { grid-template-columns: 3em minmax(8em, max-content) 1fr auto auto auto 1em; }`, `.replay-of { color: var(--tjsi-muted); white-space: nowrap; }`, `.btn.replay[data-replay-error] { color: var(--tjsi-err); }`. |
| `src/panel/panel.ts` (E2) | Action `replay`: the button is disabled until the request settles; `ok` → nothing more (the new row arrives as events); `ok: false` or rejection → `button.dataset.replayError = message; button.title = message`. The click does not expand the row (the button is the closest `[data-action]`). |
| `test/export.test.ts` (E1) | Node. Shape, `version === VERSION` and `package.json`, ISO `exportedAt`, `events` equal `bus.history` after `fixtureEvents()`, round-trips `JSON.parse(JSON.stringify())`, honours `maxHistory`; `exportFileName(new Date(Date.UTC(2026, 8, 11, 9, 5, 7)))` matches `/^transformersjs-inspector-\d{8}-\d{6}\.json$/`. |
| `test/panel.test.ts` (E1) | Stub `URL.createObjectURL`/`revokeObjectURL` and spy `HTMLAnchorElement.prototype.click`: Export clicks an anchor whose `download` matches the pattern and sets `data-export-status="downloaded"`; with `createObjectURL` removed and `navigator.clipboard.writeText` stubbed, Export copies the JSON (`JSON.parse` of the argument has `events.length === 12`) and sets `copied`; Copy JSON with no clipboard sets `failed`. |
| `test/replay.test.ts` (E2) | Node. `attach(fakePipeline(), { panel: false, bus })`, run once → `bus.request('replay', { callId: 'c1' })` resolves `{ ok: true, callId: 'c2' }` before the call completes; after settling, history has two `call:start`s, the second with `replayOf: 'c1'`, identical `tokenize.ids`, and every event clone-safe. A pipeline whose `_call` awaits a deferred: a second request while in flight → `{ ok: false, error: /in flight/ }`; after resolving, a third succeeds. Unknown id → `ok: false`. `detach()` → the id is unknown. `replayHistory: 1` evicts `c1` after `c2`. A replayed call is itself replayable (`replayOf` chains). Two pipelines on one bus share one handler and both replay. |
| `test/worker.test.ts` (E2) | Over a real `MessageChannel`: page bus `request('replay')` is answered by the worker-side handler. |
| `test/panel-model.test.ts`, `test/panel.test.ts` (E2) | Reducer copies `replayOf`; Replay button on `c1`/`c2`, none on a synthetic row; click → `bus.request` called with `('replay', { callId: 'c1' })` and the row is not expanded; an `{ ok: false }` response sets `data-replay-error`; a `call:start` with `replayOf: 'c1'` renders `[data-replay-of="c1"]`. |
| `e2e/export.spec.ts` (E1) | Run feature extraction; `Promise.all([page.waitForEvent('download'), panel.locator('[data-action="export"]').click()])`; `suggestedFilename()` matches the pattern; `JSON.parse(readFileSync(await download.path()))` has `version` equal to the demo's `[data-version]`, an ISO `exportedAt`, and `events.map(e => e.type)` equal to `['call:start','tokenize','run:start','run:end','result']`. |
| `e2e/replay.spec.ts` (E2) | Run feature extraction, open the panel, click Replay on the row → 2 rows, badge `2`, the second row has `data-replay-of` equal to the first's `data-call`; expand both → identical `.chip-id` lists. |
| `README.md` | Usage: Export and Replay paragraphs, `exportEvents(bus)` snippet; `AttachOptions` gains `replayHistory`; events table gains `replayOf`; a "Requests" table (`tensor`, `replay`); "What you cannot see": "Read-only" becomes "Replay re-runs a captured call with the same input; editing inputs is still out", and "Nothing persisted or uploaded" becomes "Nothing is stored or sent by the library; Export is a user-initiated download (or clipboard copy) of the event history". |

**Verify**

```bash
npm test && npm run typecheck && npm run lint && npm run e2e -- e2e/export.spec.ts e2e/replay.spec.ts e2e/worker.spec.ts && grep -rn "innerHTML\|style=" src/panel; test $? -eq 1
```

**Done when:** a Playwright-intercepted download parses to `{ version, exportedAt, events }`,
`exportEvents` is on the public API, and Replay produces a second, identically tokenised row
both in-page and through the worker bridge.

## Still out (the fence, restated for v1.1)

Per-layer activations; other runtimes; a browser extension; `localStorage` or any storage;
editing inputs before replay; tensor bytes in the export; thumbnails inside workers; image
previews for `Blob`/URL inputs (metadata + `src` only); wrapping `pipe.processor`.

## Stories

Each is one focused agent session. "Parallel group" lists stories that touch disjoint files
and may run concurrently; everything else runs after its dependencies. Every story ends with
`npm run typecheck && npm run lint && npm test` green and the `src/panel` grep gate empty.

1. **A · Decoded token strings.** Files: `src/types.ts`, `src/context.ts`, `src/wrap/tokenizer.ts`,
   `src/wrap/generation.ts`, `src/events.ts`, `src/panel/render.ts`, `src/panel/styles.ts`
   (one `.ws` rule after `.chip-str`), `test/fakes.ts`, `test/wrap-tokenizer.test.ts`,
   `test/generation.test.ts`, `test/attach.test.ts`, `test/panel.test.ts`, `e2e/demo.spec.ts`,
   `README.md`. Accept: Phase A verify passes; `raw` present on every tokenize/logits/token
   event in `test/attach.test.ts`. Deps: none. **Group 1** (with 2, 3).
2. **B · Dark theme.** Files: `src/panel/styles.ts` (tokens in place, dark blocks appended),
   `src/panel/panel.ts`, `demo/main.ts`, `e2e/screenshots.spec.ts`, `docs/img/panel-dark.png`,
   `test/panel.test.ts`, `README.md`. Accept: Phase B verify; `panel-dark.png` committed.
   Deps: none. **Group 1.** (Shares `styles.ts`, `panel.test.ts` and `README.md` with 1 in
   disjoint hunks; whichever lands second rebases.)
3. **C1 · Fit for any corner.** Files: `src/panel/fit.ts`, `test/panel-fit.test.ts`. Accept:
   the new `computeFit` dock cases pass and the existing cases are unchanged. Deps: none.
   **Group 1.**
4. **C2 · Dock option and resize grip.** Files: `src/panel/resize.ts` (new), `src/panel/panel.ts`,
   `src/panel/styles.ts`, `demo/main.ts`, `test/panel-resize.test.ts` (new), `test/panel.test.ts`,
   `e2e/resize.spec.ts` (new), `README.md`. Accept: Phase C verify. Deps: 2, 3. **Group 2**
   (with 5).
5. **D1 · Capture-side media previews.** Files: `src/preview.ts`, `src/events.ts`, `test/fakes.ts`
   (`fakeCanvas`, `fakeRawImage`, `fixtureMediaEvents`), `test/preview.test.ts`. Accept:
   `npm test -- test/preview.test.ts test/bus.test.ts` passes; `fixtureMediaEvents()` events all
   pass `isInspectorEvent`, `structuredClone` and JSON. Deps: 1 (shares `events.ts`,
   `fakes.ts`). **Group 2.**
6. **D2 · Panel rendering and tensor images.** Files: `src/panel/dom.ts`, `src/panel/render.ts`,
   `src/panel/tensor-image.ts` (new), `src/panel/panel.ts`, `src/panel/styles.ts`,
   `test/tensor-image.test.ts` (new), `test/panel.test.ts`, `README.md`. Accept: the
   `test/tensor-image.test.ts` and `test/panel.test.ts` cases of Phase D. Deps: 4, 5.
7. **D3 · Image-classification demo and e2e.** Files: `demo/index.html`, `demo/main.ts`,
   `e2e/fixtures.ts`, `e2e/media.spec.ts` (new), `README.md`. Accept:
   `npm run e2e -- e2e/media.spec.ts` passes on a cold `.cache/pw-profile` (mobilenet 3.7 MB
   download). Deps: 6. **Group 3** (with 8).
8. **E1 · Export.** Files: `src/version.ts` (new), `src/export.ts` (new), `src/panel/download.ts`
   (new), `src/panel/panel.ts`, `src/index.ts`, `test/export.test.ts` (new), `test/panel.test.ts`,
   `test/smoke.test.ts` (import path), `e2e/export.spec.ts` (new), `README.md`. Accept: Phase E1
   tests and `npm run e2e -- e2e/export.spec.ts`; `node -e "import('./dist/index.js').then(m => typeof m.exportEvents)"`
   prints `function` after `npm run build`. Deps: 6. **Group 3.**
9. **E2 · Replay.** Files: `src/replay.ts` (new), `src/context.ts`, `src/wrap/pipeline.ts`,
   `src/attach.ts`, `src/events.ts`, `src/panel/model.ts`, `src/panel/render.ts`,
   `src/panel/panel.ts`, `src/panel/styles.ts`, `test/replay.test.ts` (new), `test/worker.test.ts`,
   `test/panel-model.test.ts`, `test/panel.test.ts`, `e2e/replay.spec.ts` (new), `README.md`.
   Accept: Phase E verify in full. Deps: 7, 8.
10. **Release 0.2.0.** Files: `src/version.ts` (`VERSION = '0.2.0'`), `package.json`,
    `README.md` (status line, `@0.2.0` CDN URLs, fence section reviewed once more),
    `AGENTS.md` (layout line lists `03-plan-v1.1`), `docs/img/*.png` regenerated with
    `npm run screenshots`. Accept: `npm run build && npm pack --dry-run && npm test && npm run typecheck && npm run lint && npm run e2e`
    all green; `test/smoke.test.ts` passes against the bumped `package.json`. Deps: 9.
