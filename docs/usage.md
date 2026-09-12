# Usage guide

Start with the [README](../README.md) for installation and a basic pipeline example.

## Attachment and memory

`attach(pipe, options)` returns `{ bus, store, detach }`. It wraps the pipeline,
its tokenizer, its ONNX sessions, and generation when available. `detach()` restores
the original methods; calling it twice is safe.

| Option | Default | Purpose |
|---|---|---|
| `label` | `task · model_type` | Name shown on call rows. |
| `panel` | enabled | `false` disables the panel; an object configures it. |
| `bus` | shared default | Event bus for this attachment. |
| `store` | shared default | Explicit tensor store; overrides `retainBytes` and store preview settings. |
| `retainBytes` | shared 64 MiB budget | When supplied, creates a dedicated store. `0` retains no tensors. |
| `retainLogits` | `false` | Retain each generation step's full logits tensor. |
| `topK` | `10` | Alternatives captured per generation step. |
| `replayHistory` | `200` | Captured calls per bus; `0` disables replay recording. The first attachment sets this. |

Oldest tensor entries are evicted first. The budget measures retained tensor references,
not total process memory or replay arguments. To share one budget across pipelines:

```ts
import { attach, InspectorBus, TensorStore } from 'transformersjs-inspector';

const bus = new InspectorBus();
const store = new TensorStore({ maxBytes: 16 * 1024 * 1024, head: 4 });
attach(firstPipeline, { bus, store });
attach(secondPipeline, { bus, store });
```

See [AttachOptions](../src/attach.ts) and [InspectorOptions](../src/context.ts)
for the complete types, including the `LogitsProcessorList` compatibility option.

## Panel

Pass panel options through `attach(pipe, { panel: options })`, or mount one yourself:

```ts
import { mountPanel } from 'transformersjs-inspector';

const panel = mountPanel(bus, {
  container: document.body,
  open: true,
  view: 'detail',
  theme: 'auto',
  dock: 'bottom-right',
  maxCalls: 200,
});
```

The panel starts collapsed in Simple view by default. `theme` accepts `auto`, `light`,
or `dark`; `dock` accepts any of the four corners. Drag the corner grip to resize,
or double-click it to reset. `panel.setView('simple')` changes the view from code.

**Clear** removes panel rows, not bus history. `maxCalls` limits displayed rows;
`new InspectorBus({ maxHistory: 500 })` controls event history independently.

[PanelOptions](../src/panel/panel.ts) lists all options.

## Web Workers

Attach inside the worker, then connect its bus to a panel on the page.

Worker (`inference.worker.ts`):

```ts
import { pipeline } from '@huggingface/transformers';
import { attach, InspectorBus, exposeToPage } from 'transformersjs-inspector';

const bus = new InspectorBus();
exposeToPage(bus);
const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
attach(pipe, { bus, panel: false });
```

Page:

```ts
import { connectWorker, mountPanel } from 'transformersjs-inspector';

const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
  type: 'module',
});
const bus = connectWorker(worker);
mountPanel(bus, { open: true });
```

For several workers, pass the same bus to each `connectWorker(worker, bus)`.
You can attach page pipelines to that bus too. Call/run IDs are namespaced per bus;
tensor IDs are namespaced per store. Preserve the complete ID for tensor reads and
replays so the request reaches its owner. Don't construct IDs such as `c1` or `t1`.
Connected transports must form a tree; cycles aren't supported.

Inspector messages carry `__tjsi: 1`. Your application's worker message handlers
must ignore those messages. See [the worker demo](../demo/worker.html) for a full example.

## Events, values, export, and replay

```ts
import { exportEvents, serializeExport } from 'transformersjs-inspector';

const unsubscribe = bus.on((event) => console.log(event.type, event));
const capture = serializeExport(exportEvents(bus));
unsubscribe();
```

Events cover call starts, tokenization, session runs, logits, generated tokens,
and results. They are JSON-safe summaries; [events.ts](../src/events.ts) defines
all payloads. Decoded token text is shown in the panel; `raw` preserves vocabulary
markers such as `▁` and `##` and is shown on hover.

Read a tensor using its summary's ID:

```ts
const tensor = await bus.request('tensor', { id: tensorSummary.id });
```

The response contains `{ id, dtype, dims, data }` or `{ id, error }`, including
`evicted` for an entry removed from its store. This response can contain typed arrays.

Replay a captured call using an ID from its event:

```ts
const call = bus.history.find((event) => event.type === 'call:start');
if (call) {
  const result = await bus.request('replay', { callId: call.callId });
}
```

A successful replay response contains the new call's ID **when it starts**;
completion arrives as a `result` event. Calls still in flight cannot be replayed.
Arguments are held by reference until eviction or detachment, and callbacks run again.

The panel's **Export** button downloads the same event history. Shift-click copies it
to the clipboard. Exports contain summaries and previews, not full tensor values;
importing a capture into the panel isn't supported.

## Preload

The `transformersjs-inspector/preload` entry instruments ONNX session creation without
a pipeline reference. Load it before Transformers.js evaluates and use `device: 'auto'`.
See [the preload demo](../demo/preload.html) for the required loading order.

This entry is coupled to the ONNX Runtime version pinned by the Transformers.js version
used to build it. It imports that runtime from a CDN. Unlike `attach()`, it adds a runtime
download and shows session boundaries only: no pipeline text, tokenization, or decoded result.
With Transformers.js 4.2.0, the default `device: 'wasm'` does not work with this injected runtime.

## Media and other limits

Audio inputs show waveforms; image inputs show thumbnails when a page canvas is available.
Worker image previews contain metadata only. Tensor previews support common image layouts
and 2-D maps; `float16` and string tensors aren't previewed. Full values are read on demand.

Processors aren't wrapped, though their output tensors appear at session boundaries.
Overlapping calls on one pipeline can mix row attribution. Token-by-token decoding may
remove word-boundary markers; use the raw vocabulary strings when those matter.
