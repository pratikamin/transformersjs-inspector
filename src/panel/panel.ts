/**
 * The inspector panel: a shadow-DOM widget driven purely by an `InspectorBus`.
 *
 * Three levels of laziness: while closed only the reducer runs and the badge count is
 * written; an open panel renders one summary row per call; expanding a row renders its
 * sections. Tensor values are fetched only when "Load values" or "Preview" is clicked: those
 * are the sole callers of `bus.request('tensor')`. Details are re-rendered wholesale on every
 * update of an expanded call, so values loaded into a still-running call disappear and need
 * another click.
 */
import type { InspectorBus } from '../bus';
import type { InspectorEvent, TensorData } from '../events';
import { exportEvents, exportFilename, serializeExport } from '../export';
import { h } from './dom';
import { copyText, downloadJson } from './download';
import type { CallView, PanelState } from './model';
import { createState, reduce } from './model';
import { renderRowDetails, renderRowSummary, renderTensorImage, renderTensorValues, valuesCellFor } from './render';
import { DEFAULT_DOCK, fitElement, watchViewport } from './fit';
import type { Dock } from './fit';
import type { RenderContext } from './render';
import { installResize } from './resize';
import { adoptStyles } from './styles';

export interface PanelOptions {
  /** Where the host `<div data-tjsi-panel>` is appended; `document.body` by default. */
  container?: Element;
  /** Start expanded; default `false` (collapsed badge only). */
  open?: boolean;
  title?: string;
  /** Rows kept; oldest dropped beyond this. Default 200. */
  maxCalls?: number;
  /**
   * `'auto'` (default) follows `prefers-color-scheme`; `'light'` and `'dark'` force a theme
   * regardless of it. Written to `data-theme` on the host element.
   */
  theme?: 'auto' | 'light' | 'dark';
  /**
   * Which viewport corner the panel is fixed to; `'bottom-right'` by default. The panel grows
   * away from that corner and the resize grip sits on the opposite one. Written to
   * `data-dock` on the host element.
   */
  dock?: Dock;
}

export interface InspectorPanel {
  host: HTMLElement;
  shadow: ShadowRoot;
  open(): void;
  close(): void;
  isOpen(): boolean;
  /** Forgets every call and empties the list; the bus history is left alone. */
  clear(): void;
  /** Unsubscribes from the bus and removes the host element. */
  destroy(): void;
}

const DEFAULT_TITLE = 'Transformers.js inspector';
/** How long the header's export status text ("exported" / "copied" / "failed") stays visible. */
export const STATUS_FLASH_MS = 2000;

export type ExportStatus = 'exported' | 'copied' | 'failed';

type RowRefs = { row: HTMLElement; summary: HTMLElement; details: HTMLElement | null };

export function mountPanel(bus: InspectorBus, opts: PanelOptions = {}): InspectorPanel {
  const host = h('div', { data: { tjsiPanel: '' } });
  host.dataset.theme = opts.theme ?? 'auto';
  const dock: Dock = opts.dock ?? DEFAULT_DOCK;
  host.dataset.dock = dock;
  const shadow = host.attachShadow({ mode: 'open' });
  adoptStyles(shadow);

  const badge = h('span', { class: 'badge', data: { badge: '' } }, '0');
  const chevron = h('span', { class: 'chev' }, '▸');
  const rows = h('ol', { class: 'rows', data: { rows: '' } });
  const empty = h('div', { class: 'empty' }, 'No calls yet');
  // Drag handle on the corner opposite the anchor (placed by the stylesheet per `data-dock`).
  const grip = h('div', { class: 'grip', data: { grip: '' }, title: 'drag to resize · double-click to reset' });
  // Short-lived status text next to the header buttons (see `flash`).
  const status = h('span', { class: 'status', data: { exportStatus: '' } });
  const root = h(
    'div',
    { class: 'panel closed', data: { panel: '' } },
    h(
      'header',
      { class: 'header', data: { action: 'toggle' } },
      chevron,
      h('span', { class: 'title' }, opts.title ?? DEFAULT_TITLE),
      badge,
      h('span', { class: 'spacer' }),
      status,
      h('button', { class: 'btn', data: { action: 'export' }, title: 'download the event history as JSON · shift-click to copy it instead' }, 'Export'),
      h('button', { class: 'btn', data: { action: 'clear' } }, 'Clear'),
    ),
    h('div', { class: 'body' }, empty, rows),
    grip,
  );
  shadow.appendChild(root);
  (opts.container ?? document.body).appendChild(host);

  let state: PanelState = createState({ maxCalls: opts.maxCalls });
  let isOpen = false;
  let destroyed = false;
  const rendered = new Map<string, RowRefs>();
  const ctx: RenderContext = { bus };

  /**
   * Keeps the panel on screen whatever it is mounted in (see ./fit). Runs after anything that
   * can change the panel's size or the viewport; cheap (one rect read), so no throttling
   * beyond the viewport watcher's rAF.
   */
  const fit = (): void => {
    if (destroyed) return;
    fitElement(root, undefined, dock);
  };
  const unwatchViewport = watchViewport(fit);
  const uninstallResize = installResize(root, grip, { dock: () => dock, onResize: fit });

  const updateBadge = (): void => {
    badge.textContent = String(state.calls.length);
    empty.hidden = state.calls.length > 0;
  };

  const appendRow = (call: CallView): void => {
    const summary = renderRowSummary(call);
    const row = h('li', { class: 'row' }, summary);
    rows.appendChild(row);
    rendered.set(call.id, { row, summary, details: null });
  };

  const patchRow = (refs: RowRefs, call: CallView): void => {
    const summary = renderRowSummary(call);
    refs.summary.replaceWith(summary);
    refs.summary = summary;
    if (refs.details) {
      const details = renderRowDetails(call, ctx);
      refs.details.replaceWith(details);
      refs.details = details;
    }
  };

  /** Drops rows for calls the reducer's cap has evicted. */
  const pruneRows = (): void => {
    for (const [id, refs] of rendered) {
      if (!state.byId.has(id)) {
        refs.row.remove();
        rendered.delete(id);
      }
    }
  };

  const renderPending = (): void => {
    for (const call of state.calls) {
      if (!rendered.has(call.id)) appendRow(call);
    }
  };

  const ingest = (ev: InspectorEvent): void => {
    const { call, change } = reduce(state, ev);
    updateBadge();
    if (!isOpen) return;
    if (change === 'new') pruneRows();
    const refs = rendered.get(call.id);
    if (refs) patchRow(refs, call);
    else appendRow(call);
    fit();
  };

  const toggleExpand = (id: string): void => {
    const refs = rendered.get(id);
    const call = state.byId.get(id);
    if (!refs || !call) return;
    if (refs.details) {
      refs.details.remove();
      refs.details = null;
      refs.row.classList.remove('expanded');
    } else {
      refs.details = renderRowDetails(call, ctx);
      refs.row.appendChild(refs.details);
      refs.row.classList.add('expanded');
    }
    fit();
  };

  /**
   * Resolves one `Load values` or `Preview` click: fetches the tensor through the bus into
   * the row's values cell and hands it to `render`; the button is disabled while the request
   * is in flight. Both buttons share the cell, so the last click wins.
   */
  const loadTensor = async (id: string, row: HTMLElement, button: HTMLButtonElement | null, render: (cell: HTMLElement, data: TensorData) => void): Promise<void> => {
    const cell = valuesCellFor(row);
    cell.textContent = '';
    cell.appendChild(h('div', { class: 'muted' }, 'loading…'));
    if (button) button.disabled = true;
    try {
      const data = await bus.request('tensor', { id });
      render(cell, data);
    } catch (e: unknown) {
      render(cell, { id, error: e instanceof Error ? e.message : String(e) });
    } finally {
      if (button) button.disabled = false;
    }
  };

  let statusTimer: ReturnType<typeof setTimeout> | null = null;
  /** Shows `text` in the header for `STATUS_FLASH_MS`; a new flash restarts the clock. */
  const flash = (text: ExportStatus): void => {
    status.textContent = text;
    status.dataset.exportStatus = text;
    if (statusTimer !== null) clearTimeout(statusTimer);
    statusTimer = setTimeout(() => {
      statusTimer = null;
      status.textContent = '';
      status.dataset.exportStatus = '';
    }, STATUS_FLASH_MS);
  };

  /**
   * Export: the bus history (not the reducer's rows, which are capped by `maxCalls` and
   * emptied by Clear) serialised and offered as a file download. The clipboard is used
   * instead when the click carried Shift or when a download is impossible here (no
   * `URL.createObjectURL`, or the anchor path threw).
   */
  const doExport = async (copyInstead: boolean): Promise<void> => {
    const doc = host.ownerDocument;
    const text = serializeExport(exportEvents(bus));
    if (!copyInstead && downloadJson(text, exportFilename(), doc)) {
      flash('exported');
      return;
    }
    const ok = await copyText(text, doc.defaultView?.navigator ?? globalThis.navigator);
    flash(ok ? 'copied' : 'failed');
  };

  const onClick = (e: Event): void => {
    const target = e.target;
    if (!(target instanceof Element)) return;
    const actionEl = target.closest<HTMLElement>('[data-action]');
    if (!actionEl) return;
    switch (actionEl.dataset.action) {
      case 'toggle':
        if (isOpen) panel.close();
        else panel.open();
        break;
      case 'expand': {
        const id = actionEl.closest<HTMLElement>('[data-call]')?.dataset.call;
        if (id !== undefined) toggleExpand(id);
        break;
      }
      case 'clear':
        panel.clear();
        break;
      case 'export':
        void doExport(e instanceof MouseEvent && e.shiftKey);
        break;
      case 'load':
      case 'preview': {
        const row = actionEl.closest<HTMLElement>('tr[data-tensor]');
        const id = actionEl.dataset.tensor ?? row?.dataset.tensor;
        const render = actionEl.dataset.action === 'preview' ? renderTensorImage : renderTensorValues;
        if (row && id) void loadTensor(id, row, actionEl instanceof HTMLButtonElement ? actionEl : null, render);
        break;
      }
      default:
        break;
    }
  };

  shadow.addEventListener('click', onClick);
  for (const ev of bus.history) ingest(ev);
  const unsubscribe = bus.on(ingest);

  const panel: InspectorPanel = {
    host,
    shadow,
    open() {
      if (isOpen || destroyed) return;
      isOpen = true;
      root.classList.remove('closed');
      chevron.textContent = '▾';
      renderPending();
      fit();
    },
    close() {
      if (!isOpen) return;
      isOpen = false;
      root.classList.add('closed');
      chevron.textContent = '▸';
      fit();
    },
    isOpen: () => isOpen,
    clear() {
      state = createState({ maxCalls: opts.maxCalls });
      rendered.clear();
      rows.textContent = '';
      updateBadge();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      unsubscribe();
      unwatchViewport();
      uninstallResize();
      if (statusTimer !== null) clearTimeout(statusTimer);
      statusTimer = null;
      shadow.removeEventListener('click', onClick);
      rendered.clear();
      host.remove();
    },
  };

  if (opts.open) panel.open();
  else fit();
  return panel;
}
