/**
 * Process-wide defaults. The default bus and store live on
 * `globalThis[Symbol.for('transformersjs-inspector')]` rather than in module scope so a
 * preload bundle and an `attach()` from a differently bundled copy of this library share
 * one bus (and therefore one panel). `ensurePanel` mounts at most one panel per bus, and this
 * is the only file outside `src/panel/` allowed to look at `document`: it also registers the
 * `<canvas>` factory that gives `call:start` image previews a thumbnail on the page side.
 */
import type { InspectorPanel, PanelOptions } from './panel/panel';
import { InspectorBus } from './bus';
import { mountPanel } from './panel/panel';
import { getCanvasFactory, setCanvasFactory } from './preview';
import { TensorStore } from './store';

export const GLOBAL_KEY: unique symbol = Symbol.for('transformersjs-inspector');

interface Globals {
  bus?: InspectorBus;
  store?: TensorStore;
  panels?: WeakMap<InspectorBus, InspectorPanel>;
}

function globals(): Globals {
  const g = globalThis as unknown as Record<typeof GLOBAL_KEY, Globals | undefined>;
  return (g[GLOBAL_KEY] ??= {});
}

/**
 * Registers `document.createElement('canvas')` as the process-wide thumbnail canvas factory
 * once, where a `document` exists; a factory set by the host beforehand is kept. Workers and
 * Node have no `document`, so image previews there stay metadata-only (`OffscreenCanvas`
 * cannot make a data URL synchronously).
 */
export function registerPageCanvasFactory(): void {
  if (typeof document === 'undefined' || getCanvasFactory() !== null) return;
  setCanvasFactory((w, h) => {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    return canvas;
  });
}

registerPageCanvasFactory();

export function getDefaultBus(): InspectorBus {
  const g = globals();
  return (g.bus ??= new InspectorBus());
}

export function getDefaultStore(): TensorStore {
  const g = globals();
  return (g.store ??= new TensorStore());
}

/**
 * Mounts the panel for `bus` once (a later call returns the same panel; a destroyed panel
 * is mounted again). Returns `null` where there is no `document` (workers, Node).
 */
export function ensurePanel(bus: InspectorBus, opts?: PanelOptions): InspectorPanel | null {
  if (typeof document === 'undefined') return null;
  const panels = (globals().panels ??= new WeakMap());
  const existing = panels.get(bus);
  if (existing && existing.host.isConnected) return existing;
  const panel = mountPanel(bus, opts);
  panels.set(bus, panel);
  return panel;
}
