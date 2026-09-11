/**
 * The inspector panel: a shadow-DOM widget driven purely by an `InspectorBus`.
 *
 * Three levels of laziness: while closed only the reducer runs and the badge count is
 * written; an open panel renders one summary row per call; expanding a row renders its
 * sections. Tensor values are fetched only when "Load values" is clicked: that is the sole
 * caller of `bus.request('tensor')`. Details are re-rendered wholesale on every update of an
 * expanded call, so values loaded into a still-running call disappear and need another click.
 */
import type { InspectorBus } from '../bus';
import type { InspectorEvent } from '../events';
import { h } from './dom';
import type { CallView, PanelState } from './model';
import { createState, reduce } from './model';
import { renderRowDetails, renderRowSummary, renderTensorValues, valuesCellFor } from './render';
import type { RenderContext } from './render';
import { adoptStyles } from './styles';

export interface PanelOptions {
  /** Where the host `<div data-tjsi-panel>` is appended; `document.body` by default. */
  container?: Element;
  /** Start expanded; default `false` (collapsed badge only). */
  open?: boolean;
  title?: string;
  /** Rows kept; oldest dropped beyond this. Default 200. */
  maxCalls?: number;
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

type RowRefs = { row: HTMLElement; summary: HTMLElement; details: HTMLElement | null };

export function mountPanel(bus: InspectorBus, opts: PanelOptions = {}): InspectorPanel {
  const host = h('div', { data: { tjsiPanel: '' } });
  const shadow = host.attachShadow({ mode: 'open' });
  adoptStyles(shadow);

  const badge = h('span', { class: 'badge', data: { badge: '' } }, '0');
  const chevron = h('span', { class: 'chev' }, '▸');
  const rows = h('ol', { class: 'rows', data: { rows: '' } });
  const empty = h('div', { class: 'empty' }, 'No calls yet');
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
      h('button', { class: 'btn', data: { action: 'clear' } }, 'Clear'),
    ),
    h('div', { class: 'body' }, empty, rows),
  );
  shadow.appendChild(root);
  (opts.container ?? document.body).appendChild(host);

  let state: PanelState = createState({ maxCalls: opts.maxCalls });
  let isOpen = false;
  let destroyed = false;
  const rendered = new Map<string, RowRefs>();
  const ctx: RenderContext = { bus };

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
  };

  /** Resolves one `Load values` click; the button is disabled while the request is in flight. */
  const loadValues = async (id: string, row: HTMLElement, button: HTMLButtonElement | null): Promise<void> => {
    const cell = valuesCellFor(row);
    cell.textContent = '';
    cell.appendChild(h('div', { class: 'muted' }, 'loading…'));
    if (button) button.disabled = true;
    try {
      const data = await bus.request('tensor', { id });
      renderTensorValues(cell, data);
    } catch (e: unknown) {
      renderTensorValues(cell, { id, error: e instanceof Error ? e.message : String(e) });
    } finally {
      if (button) button.disabled = false;
    }
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
      case 'load': {
        const row = actionEl.closest<HTMLElement>('tr[data-tensor]');
        const id = actionEl.dataset.tensor ?? row?.dataset.tensor;
        if (row && id) void loadValues(id, row, actionEl instanceof HTMLButtonElement ? actionEl : null);
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
    },
    close() {
      if (!isOpen) return;
      isOpen = false;
      root.classList.add('closed');
      chevron.textContent = '▸';
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
      shadow.removeEventListener('click', onClick);
      rendered.clear();
      host.remove();
    },
  };

  if (opts.open) panel.open();
  return panel;
}
