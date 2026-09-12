// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { InspectorBus } from '../src/bus';
import type { InspectorEvent, TensorData } from '../src/events';
import { fmtBytes, fmtDims, fmtMs, fmtNum, h, svg } from '../src/panel/dom';
import type { InspectorPanel } from '../src/panel/panel';
import { STATUS_FLASH_MS, mountPanel } from '../src/panel/panel';
import { REVOKE_DELAY_MS, copyText, downloadJson } from '../src/panel/download';
import { MAX_VALUES, renderTensorImage, renderTensorValues, renderWaveform } from '../src/panel/render';
import { adoptStyles, PANEL_CSS } from '../src/panel/styles';
import { exportEvents } from '../src/export';
import type { InspectorExport } from '../src/export';
import { FAKE_THUMB, fixtureEvents, fixtureMediaEvents } from './fakes';

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

  test('media rules: waveform, thumbnail and tensor image, coloured through tokens only', () => {
    expect(PANEL_CSS).toMatch(/\.wave\s*\{[^}]*height: 40px/);
    expect(PANEL_CSS).toMatch(/\.wave-area\s*\{[^}]*fill: var\(--tjsi-accent\)/);
    expect(PANEL_CSS).toMatch(/img\.thumb\s*\{[^}]*max-width: 96px/);
    expect(PANEL_CSS).toMatch(/canvas\.tensor-image\s*\{[^}]*image-rendering: pixelated/);
  });

  test('dock insets on the host and a 16px grip on the corner opposite the anchor', () => {
    expect(PANEL_CSS).toMatch(/:host\(\[data-dock="bottom-left"\]\)\s*\{\s*right: auto; left: 16px;/);
    expect(PANEL_CSS).toMatch(/:host\(\[data-dock="top-right"\]\)\s*\{\s*bottom: auto; top: 16px;/);
    expect(PANEL_CSS).toMatch(/:host\(\[data-dock="top-left"\]\)\s*\{\s*right: auto; bottom: auto; left: 16px; top: 16px;/);
    expect(PANEL_CSS).toMatch(/\.panel\s*\{[^}]*position: relative/);
    expect(PANEL_CSS).toMatch(/\.body\s*\{[^}]*flex: 1 1 auto;[^}]*min-height: 0/);
    const grip = PANEL_CSS.match(/\n\.grip\s*\{([^}]*)\}/)?.[1] ?? '';
    expect(grip).toContain('position: absolute');
    expect(grip).toContain('width: 16px');
    expect(grip).toContain('height: 16px');
    expect(grip).toContain('touch-action: none');
    expect(grip).toContain('cursor: nwse-resize');
    expect(grip).toMatch(/top: 0;[^}]*left: 0;/);
    expect(PANEL_CSS).toMatch(/:host\(\[data-dock="bottom-left"\]\) \.grip\s*\{[^}]*right: 0;[^}]*cursor: nesw-resize/);
    expect(PANEL_CSS).toMatch(/:host\(\[data-dock="top-right"\]\) \.grip\s*\{[^}]*bottom: 0;[^}]*cursor: nesw-resize/);
    expect(PANEL_CSS).toMatch(/:host\(\[data-dock="top-left"\]\) \.grip\s*\{[^}]*right: 0;[^}]*bottom: 0;/);
    expect(PANEL_CSS).toMatch(/\.panel\.closed \.grip\s*\{\s*display: none/);
    // A dragged CSSOM width must not survive into the collapsed badge.
    expect(PANEL_CSS).toMatch(/\.panel\.closed\s*\{[^}]*width: auto !important;[^}]*height: auto !important/);
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

  test('data-dock is bottom-right by default and follows the dock option; the grip is the last child of the panel', () => {
    const p = mount(new InspectorBus());
    expect(p.host.dataset.dock).toBe('bottom-right');
    const root = p.shadow.querySelector<HTMLElement>('[data-panel]');
    const grip = root?.querySelector<HTMLElement>('[data-grip]');
    expect(grip?.classList.contains('grip')).toBe(true);
    expect(grip?.parentElement).toBe(root);
    expect(root?.lastElementChild).toBe(grip);
    expect(grip?.title).toContain('double-click to reset');
    expect(grip?.closest('[data-action]')).toBeNull();
    for (const dock of ['bottom-left', 'top-right', 'top-left'] as const) {
      expect(mount(new InspectorBus(), { dock }).host.getAttribute('data-dock')).toBe(dock);
    }
  });

  test('fit() reads the anchor corner named by the dock option', () => {
    const p = mount(new InspectorBus(), { dock: 'top-left' });
    const root = p.shadow.querySelector<HTMLElement>('[data-panel]');
    if (!root) throw new Error('panel root missing');
    // Top-left anchor off screen by 50px on both axes: the panel is nudged back inside.
    root.getBoundingClientRect = () => ({ x: -50, y: -50, top: -50, left: -50, width: 560, height: 400, right: 510, bottom: 350, toJSON: () => ({}) }) as DOMRect;
    p.open();
    expect(root.style.translate).toBe('66px 66px');
    // The same rect under the default dock is anchored at (510, 350), well inside: no nudge.
    const q = mount(new InspectorBus());
    const qroot = q.shadow.querySelector<HTMLElement>('[data-panel]');
    if (!qroot) throw new Error('panel root missing');
    qroot.getBoundingClientRect = root.getBoundingClientRect;
    q.open();
    expect(qroot.style.translate).toBe('');
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

describe('media previews (story 6)', () => {
  const details = (p: InspectorPanel, id: string): Element | null => p.shadow.querySelector(`[data-details="${id}"]`);
  const mountMedia = (bus: InspectorBus): InspectorPanel => {
    const p = mount(bus, { open: true });
    for (const ev of fixtureMediaEvents()) bus.emit(ev);
    return p;
  };
  /** A 2d-context stand-in for happy-dom, whose `getContext` returns null. */
  const fakeContext = () => {
    const puts: { data: Uint8ClampedArray; width: number; height: number; x: number; y: number }[] = [];
    const ctx = {
      createImageData: (width: number, height: number) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) }),
      putImageData: (image: { data: Uint8ClampedArray; width: number; height: number }, x: number, y: number) => {
        puts.push({ ...image, x, y });
      },
    };
    return { ctx, puts };
  };

  test('h() sets src only on <img> and only for data:image/ URLs; width/height only on img and canvas', () => {
    expect(h('img', { src: FAKE_THUMB }).getAttribute('src')).toBe(FAKE_THUMB);
    expect(h('img', { src: 'https://example.com/a.png' }).hasAttribute('src')).toBe(false);
    expect(h('img', { src: 'data:text/html,hi' }).hasAttribute('src')).toBe(false);
    expect(h('div', { src: FAKE_THUMB }).hasAttribute('src')).toBe(false);
    const canvas = h('canvas', { width: 12, height: 7 });
    expect(canvas.width).toBe(12);
    expect(canvas.height).toBe(7);
    expect(h('div', { width: 12 }).hasAttribute('width')).toBe(false);
  });

  test('svg() creates namespaced elements with attributes and no markup parsing', () => {
    const el = svg('svg', { viewBox: '0 0 200 40', class: 'wave' }, svg('path', { d: 'M0,20 L200,20 Z' }));
    expect(el.namespaceURI).toBe('http://www.w3.org/2000/svg');
    expect(el.getAttribute('viewBox')).toBe('0 0 200 40');
    expect(el.querySelector('path')?.namespaceURI).toBe('http://www.w3.org/2000/svg');
    expect(el.querySelector('path')?.getAttribute('d')).toBe('M0,20 L200,20 Z');
    expect(svg('rect', { width: '<b>' }).getAttribute('width')).toBe('<b>');
    expect(svg('rect', { width: '<b>' }).children).toHaveLength(0);
  });

  test('renderWaveform: maxes left to right at y = 20 - v*19, mins back, closed; clamped to [-1, 1]', () => {
    const d = renderWaveform([-1, 1, 0, 0, -0.5, 0.5]).querySelector('path')?.getAttribute('d');
    expect(d).toBe('M0,1 L100,20 L200,10.5 L200,29.5 L100,20 L0,39 Z');
    expect(renderWaveform([-5, 5]).querySelector('path')?.getAttribute('d')).toBe('M20,1 L20,39 Z');
    expect(renderWaveform([]).querySelector('path')?.getAttribute('d')).toBe('M0,20 L200,20 Z');
    const wave = renderWaveform([0, 0]);
    expect(wave.getAttribute('class')).toBe('wave');
    expect(wave.getAttribute('preserveAspectRatio')).toBe('none');
  });

  test('c3 (audio) renders samples, rate and duration text plus an svg.wave whose path starts at x = 0', () => {
    const p = mountMedia(new InspectorBus());
    click(rowsOf(p)[0]);
    const input = details(p, 'c3')?.querySelector('section');
    expect(input?.textContent).toContain('audio · 48000 samples @ 16000 Hz · 3 s');
    const wave = input?.querySelector('svg.wave');
    expect(wave?.namespaceURI).toBe('http://www.w3.org/2000/svg');
    expect(wave?.getAttribute('viewBox')).toBe('0 0 200 40');
    const d = wave?.querySelector('path.wave-area')?.getAttribute('d') ?? '';
    expect(d.startsWith('M0,')).toBe(true);
    expect(d.endsWith(' Z')).toBe(true);
    expect(d.split(' L')).toHaveLength(400); // 200 maxes out, 200 mins back
    expect(rowsOf(p)[0].querySelector('.excerpt')?.textContent).toBe('audio 48000 samples @ 16000 Hz');
  });

  test('c4 (image) renders img.thumb from the data: URL and the size/channels text', () => {
    const p = mountMedia(new InspectorBus());
    click(rowsOf(p)[1]);
    const input = details(p, 'c4')?.querySelector('section');
    const img = input?.querySelector<HTMLImageElement>('img.thumb');
    expect(img?.getAttribute('src')?.startsWith('data:image/jpeg;base64,')).toBe(true);
    expect(img?.title).toBe('input thumbnail');
    expect(input?.textContent).toContain('image 224×224×4');
  });

  test('old previews without peaks/thumb render as before, and a remote src is text, never an <img>', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    bus.emit({ type: 'call:start', callId: 'c1', label: 'asr', task: 'automatic-speech-recognition', input: { kind: 'audio', samples: 100, sampleRate: 16000 }, t: 1 });
    bus.emit({ type: 'call:start', callId: 'c2', label: 'img', task: 'image-classification', input: { kind: 'image', width: 10, height: 10, src: 'https://example.com/cat.png' }, t: 2 });
    bus.emit({ type: 'call:start', callId: 'c3', label: 'img', task: 'image-classification', input: { kind: 'image', width: 10, height: 10, thumb: 'https://example.com/cat.png' }, t: 3 });
    for (const row of rowsOf(p)) click(row);
    expect(details(p, 'c1')?.textContent).toContain('audio · 100 samples @ 16000 Hz');
    expect(details(p, 'c1')?.querySelector('svg')).toBeNull();
    expect(details(p, 'c2')?.querySelector('img')).toBeNull();
    expect(details(p, 'c2')?.textContent).toContain('https://example.com/cat.png');
    expect(details(p, 'c2')?.textContent).toContain('image 10×10');
    expect(details(p, 'c3')?.querySelector('img')).toBeNull();
    expect(p.shadow.querySelector('[src]')).toBeNull();
  });

  test('Preview buttons appear on image-shaped tensor rows only', () => {
    const bus = new InspectorBus();
    const p = mount(bus, { open: true });
    for (const ev of [...fixtureEvents(), ...fixtureMediaEvents()]) bus.emit(ev);
    for (const row of rowsOf(p)) click(row);
    const preview = (id: string): Element | null => p.shadow.querySelector(`tr[data-tensor="${id}"] button[data-action="preview"]`);
    expect(preview('t20')?.textContent).toBe('Preview'); // input_features [1, 80, 3000]
    expect(preview('t22')?.textContent).toBe('Preview'); // pixel_values [1, 3, 224, 224]
    expect(preview('t21')?.textContent).toBe('Preview'); // last_hidden_state [1, 1500, 384]: a [1,C,T] map
    expect(preview('t4')).toBeNull(); // last_hidden_state [1, 7, 384]: 7 rows is under MIN_IMAGE_SIDE
    expect(preview('t23')).toBeNull(); // logits [1, 1000]
    expect(p.shadow.querySelectorAll('button[data-action="preview"]')).toHaveLength(3);
    expect(p.shadow.querySelector('tr[data-tensor="t22"] button[data-action="load"]')).not.toBeNull();
  });

  test('clicking Preview requests exactly that id and, under happy-dom, renders the canvas-unavailable note with the mapping', async () => {
    const bus = new InspectorBus();
    const w = 3000;
    bus.handle('tensor', ({ id }) => ({ id, dtype: 'float32', dims: [1, 80, w], data: Float32Array.from({ length: 80 * w }, (_, i) => (i % w) / 1000 - 1.5) }));
    const request = vi.spyOn(bus, 'request');
    const p = mountMedia(bus);
    click(rowsOf(p)[0]);
    expect(request).not.toHaveBeenCalled();
    const button = p.shadow.querySelector<HTMLButtonElement>('[data-details="c3"] tr[data-tensor="t20"] button[data-action="preview"]');
    click(button);
    expect(request).toHaveBeenCalledTimes(1);
    expect(request).toHaveBeenCalledWith('tensor', { id: 't20' });
    expect(button?.disabled).toBe(true);
    await tick();
    expect(button?.disabled).toBe(false);
    const cell = valuesUnder(p.shadow.querySelector('[data-details="c3"] tr[data-tensor="t20"]'));
    expect(cell?.dataset.values).toBe('t20');
    expect(cell?.querySelector('canvas')).toBeNull();
    expect(cell?.querySelector('[data-canvas-unavailable]')?.textContent).toBe('canvas unavailable');
    const meta = cell?.querySelector('.meta')?.textContent ?? '';
    expect(meta).toContain('min');
    expect(meta).toBe('float32 [1, 80, 3000] · gray 80×3000 → 80×512 · min -1.5 · max 1.499');
    expect(p.shadow.querySelectorAll('tr.values')).toHaveLength(1);
  });

  test('with a 2d context, Preview paints the rasterised bytes onto canvas.tensor-image', async () => {
    const { ctx, puts } = fakeContext();
    const spy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => ctx as unknown as CanvasRenderingContext2D);
    try {
      const bus = new InspectorBus();
      const data = new Float32Array(3 * 224 * 224);
      for (let i = 0; i < 224 * 224; i++) {
        data[i] = i % 224; // R ramps across columns
        data[224 * 224 + i] = 2; // G flat
        data[2 * 224 * 224 + i] = -Math.floor(i / 224); // B ramps down rows
      }
      bus.handle('tensor', ({ id }) => ({ id, dtype: 'float32', dims: [1, 3, 224, 224], data }));
      const p = mountMedia(bus);
      click(rowsOf(p)[1]);
      click(p.shadow.querySelector('[data-details="c4"] tr[data-tensor="t22"] button[data-action="preview"]'));
      await tick();
      const cell = valuesUnder(p.shadow.querySelector('[data-details="c4"] tr[data-tensor="t22"]'));
      const canvas = cell?.querySelector<HTMLCanvasElement>('canvas.tensor-image');
      expect(canvas?.width).toBe(224);
      expect(canvas?.height).toBe(224);
      expect(cell?.querySelector('[data-canvas-unavailable]')).toBeNull();
      expect(puts).toHaveLength(1);
      expect(puts[0]).toMatchObject({ width: 224, height: 224, x: 0, y: 0 });
      const px = puts[0].data;
      expect(Array.from(px.slice(0, 4))).toEqual([0, 0, 255, 255]); // top-left: R min, G flat, B max
      expect(Array.from(px.slice(223 * 4, 224 * 4))).toEqual([255, 0, 255, 255]); // top-right
      expect(Array.from(px.slice(223 * 224 * 4, 223 * 224 * 4 + 4))).toEqual([0, 0, 0, 255]); // bottom-left
      expect(cell?.querySelector('.meta')?.textContent).toBe('float32 [1, 3, 224, 224] · rgb chw 224×224 · min…max R 0…223 · G 2…2 · B -223…0');
    } finally {
      spy.mockRestore();
    }
  });

  test('Preview: an {error} response, a rejected request and a non-image response render the error string', async () => {
    const bus = new InspectorBus();
    bus.handle('tensor', ({ id }) => (id === 't20' ? { id, error: 'evicted' } : Promise.reject(new Error('readback failed'))));
    const p = mountMedia(bus);
    for (const row of rowsOf(p)) click(row);
    click(p.shadow.querySelector('[data-details="c3"] tr[data-tensor="t20"] button[data-action="preview"]'));
    click(p.shadow.querySelector('[data-details="c4"] tr[data-tensor="t22"] button[data-action="preview"]'));
    await tick();
    const evicted = valuesUnder(p.shadow.querySelector('[data-details="c3"] tr[data-tensor="t20"]'));
    expect(evicted?.querySelector('[data-values-error]')?.textContent).toBe('evicted');
    expect(evicted?.querySelector('.meta')).toBeNull();
    const rejected = valuesUnder(p.shadow.querySelector('[data-details="c4"] tr[data-tensor="t22"]'));
    expect(rejected?.querySelector('.error')?.textContent).toBe('readback failed');

    const el = h('div');
    renderTensorImage(el, { id: 'x', dtype: 'float32', dims: [1, 7, 384], data: new Float32Array(7 * 384) });
    expect(el.querySelector('.error')?.textContent).toBe('not an image: float32 [1, 7, 384]');
    renderTensorImage(el, { id: 'x', dtype: 'string', dims: [8, 8], data: new Array<string>(64).fill('a') });
    expect(el.querySelector('.error')?.textContent).toContain('not an image');
    renderTensorImage(el, { id: 'x', dtype: 'float32', dims: [8, 8], data: new Float32Array(3) });
    expect(el.querySelector('.error')?.textContent).toContain('needs 64');
  });

  test('Preview and Load values share the cell; the last click wins', async () => {
    const bus = new InspectorBus();
    bus.handle('tensor', ({ id }) => ({ id, dtype: 'float32', dims: [1, 80, 3000], data: new Float32Array(80 * 3000) }));
    const p = mountMedia(bus);
    click(rowsOf(p)[0]);
    const row = p.shadow.querySelector('[data-details="c3"] tr[data-tensor="t20"]');
    click(row?.querySelector('button[data-action="preview"]'));
    await tick();
    expect(valuesUnder(row)?.querySelector('.meta')?.textContent).toContain('gray');
    click(row?.querySelector('button[data-action="load"]'));
    await tick();
    expect(valuesUnder(row)?.querySelector('.values-list')).not.toBeNull();
    expect(valuesUnder(row)?.querySelector('[data-canvas-unavailable]')).toBeNull();
    expect(p.shadow.querySelectorAll('tr.values')).toHaveLength(1);
  });
});

/**
 * Story 8: Export. Under happy-dom on Node 22 `URL.createObjectURL` is Node's own (inherited
 * static, `blob:nodedata:` URLs nothing here can read back), so the download path shadows it
 * with an own-property stub that captures the Blob plus a spy on
 * `HTMLAnchorElement.prototype.click`, and the fallback tests shadow it with `undefined`;
 * `delete` restores the inherited original. The clipboard path uses a stubbed `navigator.clipboard`.
 */
describe('export (story 8)', () => {
  const FILENAME = /^transformersjs-inspector-\d{8}-\d{6}\.json$/;
  type UrlStatics = { createObjectURL?: (b: Blob) => string; revokeObjectURL?: (u: string) => void };
  const urlStatics = URL as unknown as UrlStatics;
  let blobs: Blob[];
  let revoked: string[];
  let anchors: HTMLAnchorElement[];
  let clipboardTexts: string[];

  const stubObjectUrl = (): void => {
    urlStatics.createObjectURL = (b: Blob) => {
      blobs.push(b);
      return `blob:stub/${blobs.length}`;
    };
    urlStatics.revokeObjectURL = (u: string) => {
      revoked.push(u);
    };
  };
  /** Makes object URLs unavailable, as in a runtime without them. */
  const removeObjectUrl = (): void => {
    urlStatics.createObjectURL = undefined;
    urlStatics.revokeObjectURL = undefined;
  };
  const unstubObjectUrl = (): void => {
    delete urlStatics.createObjectURL;
    delete urlStatics.revokeObjectURL;
  };
  const stubClipboard = (impl: (t: string) => Promise<void>): void => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: impl } });
  };
  const unstubClipboard = (): void => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
  };
  const statusOf = (p: InspectorPanel): HTMLElement => {
    const el = p.shadow.querySelector<HTMLElement>('[data-export-status]');
    if (!el) throw new Error('no status element');
    return el;
  };
  const exportButton = (p: InspectorPanel): HTMLElement => {
    const el = p.shadow.querySelector<HTMLElement>('[data-action="export"]');
    if (!el) throw new Error('no export button');
    return el;
  };
  const filledPanel = (): { bus: InspectorBus; p: InspectorPanel } => {
    const bus = new InspectorBus();
    for (const ev of fixtureEvents()) bus.emit(ev);
    return { bus, p: mount(bus, { open: true }) };
  };
  /** Waits for the async export handler (clipboard path) to settle. */
  const settle = async (): Promise<void> => {
    await tick();
    await tick();
  };

  beforeEach(() => {
    blobs = [];
    revoked = [];
    anchors = [];
    clipboardTexts = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      anchors.push(this);
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    unstubObjectUrl();
    unstubClipboard();
  });

  test('the header has an Export button before Clear and an empty, hidden status span', () => {
    const { p } = filledPanel();
    const buttons = [...p.shadow.querySelectorAll<HTMLButtonElement>('header .btn')].map((b) => b.dataset.action);
    expect(buttons).toEqual(['export', 'clear']);
    expect(exportButton(p).title).toMatch(/shift-click to copy/i);
    expect(statusOf(p).textContent).toBe('');
    expect(statusOf(p).dataset.exportStatus).toBe('');
    expect(PANEL_CSS).toContain('.status:empty { display: none; }');
  });

  test('click: downloads a Blob whose JSON events equal bus.history, names it by the pattern, flashes "exported" and does not toggle the panel', async () => {
    vi.useFakeTimers();
    stubObjectUrl();
    const { bus, p } = filledPanel();
    click(exportButton(p));
    expect(p.isOpen()).toBe(true);
    expect(anchors).toHaveLength(1);
    expect(anchors[0].download).toMatch(FILENAME);
    expect(anchors[0].href).toBe('blob:stub/1');
    expect(anchors[0].isConnected).toBe(false);
    expect(blobs).toHaveLength(1);
    expect(blobs[0].type).toBe('application/json');
    const parsed = JSON.parse(await blobs[0].text()) as InspectorExport;
    expect(parsed.events).toEqual([...bus.history]);
    expect(parsed.events).toHaveLength(fixtureEvents().length);
    expect(parsed.version).toBe(exportEvents(bus).version);
    expect(Number.isNaN(Date.parse(parsed.exportedAt))).toBe(false);
    expect(statusOf(p).textContent).toBe('exported');
    expect(statusOf(p).dataset.exportStatus).toBe('exported');
    // The object URL is revoked after a delay, the status a little later.
    expect(revoked).toEqual([]);
    vi.advanceTimersByTime(REVOKE_DELAY_MS);
    expect(revoked).toEqual(['blob:stub/1']);
    vi.advanceTimersByTime(STATUS_FLASH_MS - REVOKE_DELAY_MS - 1);
    expect(statusOf(p).textContent).toBe('exported');
    vi.advanceTimersByTime(1);
    expect(statusOf(p).textContent).toBe('');
    expect(statusOf(p).dataset.exportStatus).toBe('');
  });

  test('fallback: without URL.createObjectURL the JSON is copied to the clipboard and "copied" flashes', async () => {
    removeObjectUrl();
    stubClipboard(async (t) => {
      clipboardTexts.push(t);
    });
    const { bus, p } = filledPanel();
    click(exportButton(p));
    await settle();
    expect(anchors).toHaveLength(0);
    expect(clipboardTexts).toHaveLength(1);
    const parsed = JSON.parse(clipboardTexts[0]) as InspectorExport;
    expect(parsed.events).toEqual([...bus.history]);
    expect(clipboardTexts[0].startsWith('{\n  "version": ')).toBe(true);
    expect(statusOf(p).textContent).toBe('copied');
  });

  test('shift-click copies even when a download is possible', async () => {
    stubObjectUrl();
    stubClipboard(async (t) => {
      clipboardTexts.push(t);
    });
    const { p } = filledPanel();
    exportButton(p).dispatchEvent(new MouseEvent('click', { bubbles: true, composed: true, shiftKey: true }));
    await settle();
    expect(anchors).toHaveLength(0);
    expect(blobs).toHaveLength(0);
    expect(clipboardTexts).toHaveLength(1);
    expect(statusOf(p).textContent).toBe('copied');
  });

  test('with neither object URLs nor a clipboard the status is "failed"; a rejected clipboard write also fails', async () => {
    removeObjectUrl();
    const { p } = filledPanel();
    click(exportButton(p));
    await settle();
    expect(statusOf(p).textContent).toBe('failed');
    stubClipboard(() => Promise.reject(new Error('denied')));
    click(exportButton(p));
    await settle();
    expect(statusOf(p).textContent).toBe('failed');
  });

  test('the bus history, not the panel rows, is exported: Clear leaves the export intact and maxCalls does not cap it', async () => {
    stubObjectUrl();
    const bus = new InspectorBus();
    for (const ev of fixtureEvents()) bus.emit(ev);
    const p = mount(bus, { open: true, maxCalls: 1 });
    expect(rowsOf(p)).toHaveLength(1);
    click(p.shadow.querySelector('[data-action="clear"]'));
    expect(rowsOf(p)).toHaveLength(0);
    click(exportButton(p));
    const parsed = JSON.parse(await blobs[0].text()) as InspectorExport;
    expect(parsed.events).toEqual([...bus.history]);
    expect(parsed.events.filter((e) => e.type === 'call:start')).toHaveLength(2);
  });

  test('destroy() clears a pending status timer', () => {
    vi.useFakeTimers();
    stubObjectUrl();
    const { p } = filledPanel();
    click(exportButton(p));
    const before = vi.getTimerCount();
    p.destroy();
    expect(vi.getTimerCount()).toBe(before - 1); // the revoke timer stays; the flash timer is gone
    vi.runAllTimers();
    expect(revoked).toEqual(['blob:stub/1']);
  });

  test('downloadJson: false without object URLs; false and revoked when the anchor click throws; copyText guards', async () => {
    removeObjectUrl();
    expect(downloadJson('{}', 'x.json', document)).toBe(false);
    stubObjectUrl();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(downloadJson('{}', 'x.json', document)).toBe(false);
    expect(blobs).toHaveLength(1);
    expect(revoked).toEqual(['blob:stub/1']);
    expect(await copyText('t', undefined)).toBe(false);
    expect(await copyText('t', {} as Navigator)).toBe(false);
    expect(await copyText('t', { clipboard: { writeText: () => Promise.reject(new Error('no')) } } as unknown as Navigator)).toBe(false);
    expect(await copyText('t', { clipboard: { writeText: async () => undefined } } as unknown as Navigator)).toBe(true);
  });
});
