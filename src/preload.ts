/**
 * Zero-touch entry point. Load it as a `<script type="module">` *before* the script that
 * loads Transformers.js: the ORT import below is static, so by the time this module's body
 * runs the runtime is here and `globalThis[Symbol.for('onnxruntime')]` is set synchronously,
 * before any later module script evaluates. The host then only needs `device: 'auto'`
 * (see `src/preload-core.ts`). Shares the process-wide bus and store with `attach()`, so a
 * page that does both gets one panel.
 *
 * This is the only file under `src/` that imports `onnxruntime-web`; `vite.config.ts`
 * externalises the specifier and rewrites it to the jsDelivr URL of the exact version
 * Transformers.js pins, so `dist/preload.js` stays small and version-locked.
 */
import * as ort from 'onnxruntime-web/webgpu';
import { WrapContext } from './context';
import { ensurePanel, getDefaultBus, getDefaultStore } from './default';
import { installPreload } from './preload-core';

const bus = getDefaultBus();
const store = getDefaultStore();
store.attachTo(bus); // the panel's lazy tensor reads go through the bus
installPreload(ort, new WrapContext(bus, store, { label: 'preload' }));
ensurePanel(bus); // no-op where there is no DOM (workers)

export { bus, store };
