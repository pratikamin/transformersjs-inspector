// @vitest-environment happy-dom
import { afterEach, describe, expect, test, vi } from 'vitest';
import { InspectorBus } from '../src/bus';
import type { InspectorEvent, TensorData } from '../src/events';
import { fmtBytes, fmtDims, fmtMs, fmtNum, h } from '../src/panel/dom';
import type { InspectorPanel } from '../src/panel/panel';
import { mountPanel } from '../src/panel/panel';
import { MAX_VALUES, renderTensorValues } from '../src/panel/render';
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
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
/** The `tr.values` directly under a tensor row, or null when nothing was loaded there. */
const valuesUnder = (row: Element | null | undefined): HTMLElement | null => {
  const next = row?.nextElementSibling;
  return next instanceof HTMLElement && next.classList.contains('values') ? next : null;
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

  test('dark theme: a media block that yields to data-theme="light", a data-theme="dark" block, and no stray colours', () => {
    expect(PANEL_CSS).toMatch(/:host\s*\{[^}]*color-scheme:\s*light/);
    expect(PANEL_CSS).toMatch(/@media \(prefers-color-scheme: dark\)\s*\{\s*:host\(:not\(\[data-theme="light"\]\)\)\s*\{[^}]*color-scheme: dark/);
    expect(PANEL_CSS).toMatch(/:host\(\[data-theme="dark"\]\)\s*\{[^}]*color-scheme: dark/);
    // The dark blocks are the last two rules; the explicit block comes after the media block so it wins.
    const media = PANEL_CSS.indexOf('@media (prefers-color-scheme: dark)');
    const explicit = PANEL_CSS.indexOf(':host([data-theme="dark"])');
    expect(media).toBeGreaterThan(0);
    expect(explicit).toBeGreaterThan(media);
    // Both blocks override every token the light theme declares, and nothing else.
    const lightTokens = [...PANEL_CSS.slice(0, media).matchAll(/(--tjsi-[a-z-]+):/g)].map((m) => m[1]).sort();
    expect(lightTokens.length).toBeGreaterThanOrEqual(15);
    for (const block of PANEL_CSS.slice(media).split(/\n(?=:host)/)) {
      const darkTokens = [...block.matchAll(/(--tjsi-[a-z-]+):/g)].map((m) => m[1]).sort();
      expect(darkTokens).toEqual(lightTokens);
    }
    // With the dark blocks stripped, every colour literal sits in a --tjsi-* declaration on :host.
    const light = PANEL_CSS.slice(0, media);
    for (const line of light.split('\n')) {
      if (/#[0-9a-f]{3,8}\b|rgba?\(/i.test(line)) expect(line).toMatch(/^\s*--tjsi-[a-z-]+:/);
    }
    // No rule other than :host declares a token; every usage goes through var().
    const usages = light.match(/var\(--tjsi-[a-z-]+\)/g) ?? [];
    expect(usages.length).toBeGreaterThan(30);
    for (const name of ['fg', 'muted', 'bg', 'bg-alt', 'bg-detail', 'border', 'border-soft', 'accent', 'on-accent', 'ok', 'warn', 'err', 'picked', 'shadow']) {
      expect(light).toContain(`var(--tjsi-${name})`);
    }
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

  test('data-theme is auto by default and follows the theme option', () => {
    expect(mount(new InspectorBus()).host.dataset.theme).toBe('auto');
    expect(mount(new InspectorBus(), { theme: 'dark' }).host.dataset.theme).toBe('dark');
    expect(mount(new InspectorBus(), { theme: 'light' }).host.dataset.theme).toBe('light');
    expect(mount(new InspectorBus(), { theme: 'auto' }).host.getAttribute('data-theme')).toBe('auto');
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
    expect(headings).toEqual(['Input', 'Tokenizer', 'Session runs', 'Result']); // no steps, no Generation

    const chips = details?.querySelectorAll('.chip') ?? [];
    expect(chips).toHaveLength(7);
    expect(chips[0].querySelector('.chip-id')?.textContent).toBe('101');
    expect(chips[0].querySelector('.chip-str')?.textContent).toBe('[CLS]');
    expect(chips[0].getAttribute('title')).toBe('id 101 · raw [CLS]');
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

  test('expanding the generation call renders a Generation section with 3 steps and top-k rows with bars', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    const events = fixtureEvents();
    for (const ev of events) bus.emit(ev);
    const perStep = events.find((e) => e.type === 'logits')?.topK.length ?? 0;
    expect(perStep).toBeGreaterThan(0);
    click(rowsOf(p)[1]);

    const details = p.shadow.querySelector<HTMLElement>('[data-details="c2"]');
    const headings = [...(details?.querySelectorAll('h3') ?? [])].map((x) => x.textContent);
    expect(headings).toEqual(['Input', 'Tokenizer', 'Session runs', 'Generation', 'Result']);

    const steps = [...(details?.querySelectorAll<HTMLElement>('.step') ?? [])];
    expect(steps).toHaveLength(3);
    expect(steps.map((s) => s.dataset.step)).toEqual(['0', '1', '2']);
    expect(steps[0].querySelector('.step-head')?.textContent).toContain('token 1996 "the"');
    expect(steps[0].querySelector('.step-head')?.textContent).toContain('logits t10');
    expect(steps[2].querySelector('.step-head')?.textContent).toContain('token 5927 "brown"');

    for (const step of steps) {
      const heads = [...step.querySelectorAll('table.topk th')].map((th) => th.textContent);
      expect(heads).toEqual(['token', 'id', 'logit', 'prob']);
      const rows = [...step.querySelectorAll<HTMLElement>('table.topk tbody tr')];
      expect(rows).toHaveLength(perStep);
      for (const r of rows) {
        expect(r.querySelector('.bar')).not.toBeNull();
        expect(r.querySelector('.prob-text')?.textContent).toMatch(/^0\.\d+$/);
      }
    }

    const first = steps[0].querySelector<HTMLElement>('table.topk tbody tr');
    const cells = [...(first?.querySelectorAll('td') ?? [])].map((td) => td.textContent);
    expect(cells[0]).toBe('the');
    expect(first?.querySelector('td.tok')?.getAttribute('title')).toBe('the');
    expect(cells[1]).toBe('1996');
    expect(cells[2]).toBe('8');
    expect(cells[3]).toContain('0.9971');
    expect(first?.classList.contains('picked')).toBe(true);
    const bar = first?.querySelector<HTMLElement>('.bar');
    expect(bar?.style.width).toBe('99.7%');
    const second = steps[0].querySelectorAll<HTMLElement>('table.topk tbody tr')[1];
    expect(second.classList.contains('picked')).toBe(false);
    expect(second.querySelector<HTMLElement>('.bar')?.style.width).toBe('0.1%');
  });

  test('decoded token text: whitespace runs become span.ws, the vocab string is the title, old events still render', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    bus.emit({ type: 'call:start', callId: 'c1', label: 'text-generation', task: 'text-generation', input: { kind: 'text', text: 'a film' }, t: 1 });
    bus.emit({ type: 'tokenize', callId: 'c1', text: 'a film', ids: [[64, 2143, 2075]], tokens: [['a', ' film', 'ing']], raw: [['a', 'Ġfilm', '##ing']], ms: 0.1, t: 2 });
    bus.emit({
      type: 'logits',
      callId: 'c1',
      step: 0,
      vocab: 8,
      topK: [
        { id: 2143, token: ' film', logit: 2, prob: 0.6, raw: 'Ġfilm' },
        { id: 7, token: null, logit: 1, prob: 0.4 }, // a v0.1 entry: no raw
      ],
      tensorId: null,
      t: 3,
    });
    bus.emit({ type: 'token', callId: 'c1', step: 0, ids: [2143], text: ' film', t: 4 }); // v0.1 shape: no raw
    click(rowsOf(p)[0]);

    const chips = [...p.shadow.querySelectorAll<HTMLElement>('.chip')];
    expect(chips).toHaveLength(3);
    const film = chips[1].querySelector<HTMLElement>('.chip-str');
    expect(film?.textContent).toBe(' film');
    const ws = film?.querySelectorAll('span.ws') ?? [];
    expect(ws).toHaveLength(1);
    expect(ws[0].textContent).toBe(' ');
    expect(chips[1].getAttribute('title')).toBe('id 2143 · raw Ġfilm');
    expect(chips[2].querySelector('.chip-str')?.textContent).toBe('ing');
    expect(chips[2].querySelectorAll('span.ws')).toHaveLength(0);
    expect(chips[2].getAttribute('title')).toBe('id 2075 · raw ##ing');

    const toks = [...p.shadow.querySelectorAll<HTMLElement>('table.topk td.tok')];
    expect(toks).toHaveLength(2);
    expect(toks[0].textContent).toBe(' film');
    expect(toks[0].getAttribute('title')).toBe('Ġfilm');
    expect(toks[0].querySelectorAll('span.ws')).toHaveLength(1);
    expect(toks[1].textContent).toBe('∅');
    expect(toks[1].getAttribute('title')).toBe('');
    expect(p.shadow.querySelector('.step-head')?.textContent).toContain('token 2143 " film"');
  });

  test('a token event alone renders a step without a top-k table', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    bus.emit(fixtureEvents()[5]); // c2 call:start
    bus.emit({ type: 'token', callId: 'c2', step: 0, ids: [7], text: 'x', t: 2001 });
    click(rowsOf(p)[0]);
    const step = p.shadow.querySelector<HTMLElement>('[data-details="c2"] .step');
    expect(step?.querySelector('.step-head')?.textContent).toContain('token 7 "x"');
    expect(step?.querySelector('table.topk')).toBeNull();
    expect(step?.textContent).toContain('no logits captured');
  });

  test('no tensor request is issued before the load button is clicked; the click requests exactly that id', async () => {
    const bus = new InspectorBus();
    bus.handle('tensor', ({ id }) => ({ id, dtype: 'float32', dims: [3], data: new Float32Array([1.5, -2, 3]) }));
    const request = vi.spyOn(bus, 'request');
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[0]);
    click(rowsOf(p)[1]);
    expect(request).not.toHaveBeenCalled();
    expect(p.shadow.querySelectorAll('tr.values')).toHaveLength(0);

    const button = p.shadow.querySelector<HTMLButtonElement>('[data-details="c1"] [data-run="r1"] [data-tensor="t4"] button[data-action="load"]');
    expect(button?.dataset.tensor).toBe('t4');
    click(button);
    expect(request).toHaveBeenCalledTimes(1);
    expect(request).toHaveBeenCalledWith('tensor', { id: 't4' });
    expect(button?.disabled).toBe(true);
    await tick();
    expect(button?.disabled).toBe(false);

    const row = p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t4"]');
    const values = valuesUnder(row);
    expect(values?.dataset.values).toBe('t4');
    expect(values?.querySelector('.values-list')?.textContent).toBe('1.5, -2, 3');
    expect(values?.textContent).toContain('float32 [3] · 3 values');
    expect(values?.querySelector('[data-more]')).toBeNull();
    expect(p.shadow.querySelectorAll('tr.values')).toHaveLength(1);
  });

  test('the same tensor id appearing in Session runs and Result loads independently', async () => {
    const bus = new InspectorBus();
    let calls = 0;
    bus.handle('tensor', ({ id }) => {
      calls++;
      return { id, dtype: 'float32', dims: [2], data: new Float32Array([7, 8]) };
    });
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[0]);
    const buttons = [...p.shadow.querySelectorAll<HTMLButtonElement>('[data-details="c1"] tr[data-tensor="t4"] button[data-action="load"]')];
    expect(buttons).toHaveLength(2); // once under the run outputs, once under Result
    click(buttons[1]);
    await tick();
    expect(calls).toBe(1);
    const rows = [...p.shadow.querySelectorAll('[data-details="c1"] tr[data-tensor="t4"]')];
    expect(valuesUnder(rows[0])).toBeNull();
    expect(valuesUnder(rows[1])?.querySelector('.values-list')?.textContent).toBe('7, 8');
    click(buttons[0]);
    await tick();
    expect(calls).toBe(2);
    expect(valuesUnder(rows[0])?.querySelector('.values-list')?.textContent).toBe('7, 8');
    // A second click on the same row reuses its values cell instead of stacking another.
    click(buttons[0]);
    await tick();
    expect(p.shadow.querySelectorAll('[data-details="c1"] tr.values')).toHaveLength(2);
  });

  test('a response over MAX_VALUES is truncated with a "… N more" note; bigint and string data render', async () => {
    const bus = new InspectorBus();
    const big = new Float32Array(MAX_VALUES + 904).map((_, i) => i);
    bus.handle('tensor', ({ id }) => {
      if (id === 't1') return { id, dtype: 'int64', dims: [1, 3], data: new BigInt64Array([101n, -5n, 102n]) };
      if (id === 't2') return { id, dtype: 'string', dims: [2], data: ['[CLS]', 'the'] };
      return { id, dtype: 'float32', dims: [1, big.length], data: big };
    });
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[0]);
    const load = (id: string): void => click(p.shadow.querySelector(`[data-details="c1"] [data-run="r1"] tr[data-tensor="${id}"] button[data-action="load"]`));
    load('t4');
    load('t1');
    load('t2');
    await tick();
    const under = (id: string): HTMLElement | null => valuesUnder(p.shadow.querySelector(`[data-details="c1"] [data-run="r1"] tr[data-tensor="${id}"]`));

    const list = under('t4')?.querySelector('.values-list')?.textContent ?? '';
    expect(list.split(', ')).toHaveLength(MAX_VALUES);
    expect(list.startsWith('0, 1, 2, ')).toBe(true);
    expect(list.endsWith(`, ${MAX_VALUES - 1}`)).toBe(true);
    expect(under('t4')?.querySelector('[data-more]')?.textContent).toBe('… 904 more');
    expect(under('t4')?.textContent).toContain(`${MAX_VALUES + 904} values`);

    expect(under('t1')?.querySelector('.values-list')?.textContent).toBe('101, -5, 102');
    expect(under('t1')?.textContent).toContain('int64 [1, 3]');
    expect(under('t2')?.querySelector('.values-list')?.textContent).toBe('[CLS], the');
  });

  test('an {error} response renders the error string, and a rejected request renders its message', async () => {
    const bus = new InspectorBus();
    bus.handle('tensor', ({ id }) => (id === 't4' ? { id, error: 'evicted' } : Promise.reject(new Error('readback failed'))));
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[0]);
    click(p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t4"] button[data-action="load"]'));
    click(p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t1"] button[data-action="load"]'));
    await tick();
    const evicted = valuesUnder(p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t4"]'));
    expect(evicted?.querySelector('[data-values-error]')?.textContent).toBe('evicted');
    expect(evicted?.querySelector('.values-list')).toBeNull();
    const rejected = valuesUnder(p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t1"]'));
    expect(rejected?.querySelector('.error')?.textContent).toBe('readback failed');
  });

  test('loading with no handler on the bus renders the bus error instead of hanging', async () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    for (const ev of fixtureEvents()) bus.emit(ev);
    click(rowsOf(p)[0]);
    const button = p.shadow.querySelector<HTMLButtonElement>('[data-details="c1"] [data-run="r1"] tr[data-tensor="t4"] button[data-action="load"]');
    click(button);
    await tick();
    const cell = valuesUnder(p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t4"]'));
    expect(cell?.querySelector('.error')?.textContent).toContain("no handler for request 'tensor'");
    expect(button?.disabled).toBe(false);
  });

  test('re-rendering details on an update drops loaded values (another click reloads them)', async () => {
    const bus = new InspectorBus();
    bus.handle('tensor', ({ id }) => ({ id, dtype: 'float32', dims: [1], data: new Float32Array([4]) }));
    const p = mount(bus, { open: true });
    const events = fixtureEvents().slice(0, 5);
    for (const ev of events.slice(0, 4)) bus.emit(ev); // through run:end, no result yet
    click(rowsOf(p)[0]);
    click(p.shadow.querySelector('[data-details="c1"] tr[data-tensor="t4"] button[data-action="load"]'));
    await tick();
    expect(p.shadow.querySelectorAll('[data-details="c1"] tr.values')).toHaveLength(1);
    bus.emit(events[4]);
    expect(p.shadow.querySelectorAll('[data-details="c1"] tr.values')).toHaveLength(0);
    click(p.shadow.querySelector('[data-details="c1"] [data-run="r1"] tr[data-tensor="t4"] button[data-action="load"]'));
    await tick();
    expect(p.shadow.querySelector('[data-details="c1"] tr.values .values-list')?.textContent).toBe('4');
  });

  test('renderTensorValues handles a DataView, an empty array and each error kind', () => {
    const el = h('div');
    renderTensorValues(el, { id: 'x', dtype: 'uint8', dims: [4], data: new DataView(new ArrayBuffer(4)) });
    expect(el.textContent).toContain('empty');
    renderTensorValues(el, { id: 'x', dtype: 'string', dims: [0], data: [] });
    expect(el.textContent).toContain('0 values');
    const err: TensorData = { id: 'x', error: 'disposed' };
    renderTensorValues(el, err);
    expect(el.querySelector('.error')?.textContent).toBe('disposed');
    expect(el.querySelector('.values-list')).toBeNull();
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
