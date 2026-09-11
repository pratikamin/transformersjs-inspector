// @vitest-environment happy-dom
import { afterEach, describe, expect, test } from 'vitest';
import { InspectorBus } from '../src/bus';
import type { InspectorEvent } from '../src/events';
import { fmtBytes, fmtDims, fmtMs, fmtNum, h } from '../src/panel/dom';
import type { InspectorPanel } from '../src/panel/panel';
import { mountPanel } from '../src/panel/panel';
import { adoptStyles, PANEL_CSS } from '../src/panel/styles';
import { fixtureEvents } from './fakes';

const panels: InspectorPanel[] = [];
const mount = (bus: InspectorBus, opts?: Parameters<typeof mountPanel>[1]): InspectorPanel => {
  const p = mountPanel(bus, opts);
  panels.push(p);
  return p;
};

afterEach(() => {
  for (const p of panels.splice(0)) p.destroy();
});

const badgeOf = (p: InspectorPanel): string => p.shadow.querySelector('[data-badge]')?.textContent ?? '';
const rowsOf = (p: InspectorPanel): HTMLElement[] => [...p.shadow.querySelectorAll<HTMLElement>('[data-call]')];
const click = (el: Element | null | undefined): void => {
  if (!(el instanceof HTMLElement)) throw new Error('nothing to click');
  el.click();
};

describe('dom helpers', () => {
  test('h() sets className, dataset and text without markup interpretation', () => {
    const el = h('div', { class: 'a b', data: { callId: 'c1' } }, '<b>x</b>', 42, null, false, [h('span', null, 'y')]);
    expect(el.className).toBe('a b');
    expect(el.getAttribute('data-call-id')).toBe('c1');
    expect(el.querySelector('b')).toBeNull();
    expect(el.textContent).toBe('<b>x</b>42y');
    expect(h('button', null, 'go').type).toBe('button');
  });

  test('formatters', () => {
    expect(fmtMs(null)).toBe('…');
    expect(fmtMs(0.4)).toBe('0.40 ms');
    expect(fmtMs(12.7)).toBe('12.7 ms');
    expect(fmtMs(250)).toBe('250 ms');
    expect(fmtMs(1250)).toBe('1.25 s');
    expect(fmtBytes(0)).toBe('0 B');
    expect(fmtBytes(56)).toBe('56 B');
    expect(fmtBytes(10752)).toBe('10.5 KiB');
    expect(fmtBytes(64 * 1024 * 1024)).toBe('64.0 MiB');
    expect(fmtNum(-0.03470001)).toBe('-0.0347');
    expect(fmtNum(128256)).toBe('128300');
    expect(fmtNum(1996)).toBe('1996');
    expect(fmtNum('[CLS]')).toBe('[CLS]');
    expect(fmtDims([1, 7, 384])).toBe('[1, 7, 384]');
    expect(fmtDims([])).toBe('[]');
  });
});

describe('adoptStyles', () => {
  test('uses a constructed stylesheet when available', () => {
    const shadow = document.body.appendChild(document.createElement('div')).attachShadow({ mode: 'open' });
    adoptStyles(shadow);
    expect(shadow.adoptedStyleSheets).toHaveLength(1);
    expect(shadow.querySelector('style')).toBeNull();
  });

  test('falls back to a <style> element without adoptedStyleSheets', () => {
    const shadow = document.body.appendChild(document.createElement('div')).attachShadow({ mode: 'open' });
    // A runtime without constructed stylesheets has no `adoptedStyleSheets` property on the root at all.
    const legacy = { prepend: (n: Node) => shadow.prepend(n) } as unknown as ShadowRoot;
    adoptStyles(legacy);
    const styleEl = shadow.querySelector('style');
    expect(styleEl?.textContent).toBe(PANEL_CSS);
    expect(shadow.adoptedStyleSheets).toHaveLength(0);
  });

  test('stylesheet docks the host and reserves the story-6 bar rule', () => {
    expect(PANEL_CSS).toMatch(/:host\s*\{[^}]*all:\s*initial/);
    expect(PANEL_CSS).toMatch(/position:\s*fixed/);
    expect(PANEL_CSS).toMatch(/z-index:\s*2147483647/);
    expect(PANEL_CSS).toMatch(/\.bar\s*\{/);
  });
});

describe('mountPanel', () => {
  test('mounts a <div data-tjsi-panel> with an open shadow root on document.body', () => {
    const p = mount(new InspectorBus());
    expect(p.host.hasAttribute('data-tjsi-panel')).toBe(true);
    expect(p.host.parentElement).toBe(document.body);
    expect(p.host.shadowRoot).toBe(p.shadow);
    expect(p.shadow.mode).toBe('open');
    expect(p.isOpen()).toBe(false);
    expect(badgeOf(p)).toBe('0');
    expect(p.shadow.querySelector('.panel')?.classList.contains('closed')).toBe(true);
  });

  test('honours container, title and open options', () => {
    const container = document.body.appendChild(document.createElement('section'));
    const p = mount(new InspectorBus(), { container, title: 'Mine', open: true });
    expect(p.host.parentElement).toBe(container);
    expect(p.shadow.querySelector('.title')?.textContent).toBe('Mine');
    expect(p.isOpen()).toBe(true);
  });

  test('while closed, fixture events create no rows and the badge reads the call count', () => {
    const bus = new InspectorBus();
    const p = mount(bus);
    for (const ev of fixtureEvents()) bus.emit(ev);
    expect(rowsOf(p)).toHaveLength(0);
    expect(badgeOf(p)).toBe('2');
  });

  test('open() renders one row per call, incrementally afterwards, and close() keeps them', () => {
    const bus = new InspectorBus();
    const p = mount(bus);
    const events = fixtureEvents();
    for (const ev of events.slice(0, 5)) bus.emit(ev); // c1 complete
    p.open();
    expect(p.isOpen()).toBe(true);
    expect(rowsOf(p)).toHaveLength(1);
    for (const ev of events.slice(5)) bus.emit(ev); // c2
    const rows = rowsOf(p);
    expect(rows).toHaveLength(2);
    expect(rows.map((r) => r.dataset.call)).toEqual(['c1', 'c2']);
    expect(rows[0].querySelector('.n')?.textContent).toBe('#1');
    expect(rows[0].querySelector('.label')?.textContent).toBe('feature-extraction');
    expect(rows[0].querySelector('.excerpt')?.textContent).toBe('the quick brown fox.');
    expect(rows[0].querySelector('.ms')?.textContent).toBe('14.2 ms');
    expect(rows[0].querySelector('.dot')?.classList.contains('ok')).toBe(true);
    p.close();
    expect(p.isOpen()).toBe(false);
    expect(rowsOf(p)).toHaveLength(2);
  });

  test('replays bus.history on mount', () => {
    const bus = new InspectorBus();
    for (const ev of fixtureEvents()) bus.emit(ev);
    const p = mount(bus, { open: true });
    expect(rowsOf(p)).toHaveLength(2);
    expect(badgeOf(p)).toBe('2');
  });

  test('updated events patch the summary of an open row', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    const [start, ...rest] = fixtureEvents().slice(0, 5);
    bus.emit(start);
    expect(rowsOf(p)[0].querySelector('.dot')?.classList.contains('pending')).toBe(true);
    expect(rowsOf(p)[0].querySelector('.ms')?.textContent).toBe('…');
    for (const ev of rest) bus.emit(ev);
    expect(rowsOf(p)).toHaveLength(1);
    expect(rowsOf(p)[0].querySelector('.dot')?.classList.contains('ok')).toBe(true);
    expect(rowsOf(p)[0].querySelector('.ms')?.textContent).toBe('14.2 ms');
  });

  test('clicking expand renders tokenizer chips, tensor rows and the result; clicking again collapses', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    const [c1] = rowsOf(p);
    click(c1);

    const details = p.shadow.querySelector<HTMLElement>('[data-details="c1"]');
    expect(details).not.toBeNull();
    const headings = [...(details?.querySelectorAll('h3') ?? [])].map((x) => x.textContent);
    expect(headings).toEqual(['Input', 'Tokenizer', 'Session runs', 'Result']);

    const chips = details?.querySelectorAll('.chip') ?? [];
    expect(chips).toHaveLength(7);
    expect(chips[0].querySelector('.chip-id')?.textContent).toBe('101');
    expect(chips[0].querySelector('.chip-str')?.textContent).toBe('[CLS]');
    expect(chips[1].querySelector('.chip-str')?.textContent).toBe('the');

    const tensorRows = [...(details?.querySelectorAll<HTMLElement>('[data-tensor]') ?? [])];
    expect(tensorRows.length).toBeGreaterThanOrEqual(4);
    const hidden = tensorRows.find((r) => r.dataset.tensor === 't4');
    expect(hidden).toBeDefined();
    const cells = [...(hidden?.querySelectorAll('td') ?? [])].map((td) => td.textContent);
    expect(cells[0]).toBe('last_hidden_state');
    expect(cells[1]).toBe('float32');
    expect(cells[2]).toBe('[1, 7, 384]');
    expect(cells[3]).toBe('cpu');
    expect(cells[4]).toBe('10.5 KiB');
    expect(cells[5]).toContain('-0.0347');
    expect(cells[5]).toContain('…');
    expect(hidden?.querySelector('button[data-action="load"]')?.textContent).toBe('Load values');

    const pre = details?.querySelector('section:last-child pre');
    expect(pre?.textContent).toContain('"$tensor"');
    expect(pre?.textContent).toContain('"last_hidden_state"');

    click(c1);
    expect(p.shadow.querySelector('[data-details="c1"]')).toBeNull();
  });

  test('an expanded row re-renders its details as its call updates', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    const events = fixtureEvents().slice(0, 5);
    bus.emit(events[0]);
    click(rowsOf(p)[0]);
    expect(p.shadow.querySelectorAll('.chip')).toHaveLength(0);
    expect(p.shadow.querySelector('[data-details="c1"] section:last-child')?.textContent).toContain('pending');
    for (const ev of events.slice(1)) bus.emit(ev);
    expect(p.shadow.querySelectorAll('[data-details="c1"]')).toHaveLength(1);
    expect(p.shadow.querySelectorAll('.chip')).toHaveLength(7);
    expect(p.shadow.querySelectorAll('[data-run="r1"]')).toHaveLength(1);
    expect(p.shadow.querySelector('[data-details="c1"] section:last-child pre')).not.toBeNull();
  });

  test('gpu-resident tensors show a dash for head and errors render in red', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[1]);
    const kv = p.shadow.querySelector('[data-details="c2"] [data-tensor="t8"] td.head');
    expect(kv?.textContent).toBe('—');

    const err: InspectorEvent = { type: 'result', callId: 'c3', result: undefined, ms: 1, error: 'boom', t: 5000 };
    bus.emit(err);
    const row = rowsOf(p)[2];
    expect(row.querySelector('.dot')?.classList.contains('err')).toBe(true);
    click(row);
    expect(p.shadow.querySelector('[data-details="c3"] .error')?.textContent).toBe('boom');
  });

  test('header toggles open/closed and the clear button empties the list', () => {
    const bus = new InspectorBus();
    const p = mount(bus);
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(p.shadow.querySelector('[data-action="toggle"]'));
    expect(p.isOpen()).toBe(true);
    expect(rowsOf(p)).toHaveLength(2);
    click(p.shadow.querySelector('[data-action="clear"]'));
    expect(p.isOpen()).toBe(true); // the button's own action wins over the header toggle
    expect(rowsOf(p)).toHaveLength(0);
    expect(badgeOf(p)).toBe('0');
    bus.emit(fixtureEvents()[0]);
    expect(rowsOf(p)).toHaveLength(1);
    expect(badgeOf(p)).toBe('1');
    click(p.shadow.querySelector('[data-action="toggle"]'));
    expect(p.isOpen()).toBe(false);
  });

  test('maxCalls drops the oldest row', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true, maxCalls: 1 });
    for (const ev of fixtureEvents()) bus.emit(ev);
    expect(rowsOf(p).map((r) => r.dataset.call)).toEqual(['c2']);
    expect(badgeOf(p)).toBe('1');
  });

  test('load button is inert for now (story 6) and issues no request', async () => {
    const bus = new InspectorBus();
    let requests = 0;
    bus.handle('tensor', ({ id }) => {
      requests++;
      return { id, error: 'unknown' };
    });
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[0]);
    click(p.shadow.querySelector('[data-tensor="t4"] [data-action="load"]'));
    await new Promise((r) => setTimeout(r, 0));
    expect(requests).toBe(0);
  });

  test('destroy() removes the host and stops listening', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    p.destroy();
    expect(document.querySelector('[data-tjsi-panel]')).toBeNull();
    expect(p.host.isConnected).toBe(false);
    for (const ev of fixtureEvents()) bus.emit(ev);
    expect(rowsOf(p)).toHaveLength(0);
    expect(badgeOf(p)).toBe('0');
    p.destroy(); // idempotent
  });
});
