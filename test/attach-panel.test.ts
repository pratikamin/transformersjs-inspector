// @vitest-environment happy-dom
import { afterEach, describe, expect, test } from 'vitest';
import { attach } from '../src/attach';
import { InspectorBus } from '../src/bus';
import { ensurePanel, getDefaultBus } from '../src/default';
import { fakePipeline } from './fakes';

const hosts = (): HTMLElement[] => [...document.querySelectorAll<HTMLElement>('[data-tjsi-panel]')];

afterEach(() => {
  for (const host of hosts()) host.remove();
});

describe('ensurePanel (happy-dom)', () => {
  test('mounts exactly once per bus and returns the same panel; a second bus gets its own', () => {
    const bus = new InspectorBus();
    const panel = ensurePanel(bus, { title: 'first' });
    expect(panel).not.toBeNull();
    expect(ensurePanel(bus)).toBe(panel);
    expect(ensurePanel(bus, { open: true })).toBe(panel);
    expect(hosts()).toHaveLength(1);
    expect(panel!.host).toBe(hosts()[0]);
    expect(panel!.isOpen()).toBe(false); // later options do not re-mount or re-open

    const other = ensurePanel(new InspectorBus());
    expect(other).not.toBe(panel);
    expect(hosts()).toHaveLength(2);
  });

  test('a destroyed panel is mounted again on the next call', () => {
    const bus = new InspectorBus();
    const first = ensurePanel(bus)!;
    first.destroy();
    expect(hosts()).toHaveLength(0);
    const second = ensurePanel(bus)!;
    expect(second).not.toBe(first);
    expect(hosts()).toHaveLength(1);
  });

  test('attach() mounts the panel by default, once across attaches, honours PanelOptions, and skips it with panel:false', async () => {
    const bus = new InspectorBus();
    const a = attach(fakePipeline(), { bus });
    const b = attach(fakePipeline(), { bus });
    expect(hosts()).toHaveLength(1);
    a.detach();
    b.detach();

    const openBus = new InspectorBus();
    const pipe = fakePipeline();
    const c = attach(pipe, { bus: openBus, panel: { open: true, title: 'opened' } });
    expect(hosts()).toHaveLength(2);
    const panel = ensurePanel(openBus)!;
    expect(panel.isOpen()).toBe(true);
    await pipe('hello world');
    expect(panel.shadow.querySelector('[data-badge]')?.textContent).toBe('1');
    expect(panel.shadow.querySelectorAll('[data-call]')).toHaveLength(1);
    c.detach();

    const d = attach(fakePipeline(), { bus: new InspectorBus(), panel: false });
    expect(hosts()).toHaveLength(2);
    d.detach();
  });

  test('the default bus gets one panel no matter how many pipelines attach', () => {
    const a = attach(fakePipeline(), {});
    const b = attach(fakePipeline({ task: 'text-generation' }));
    expect(a.bus).toBe(getDefaultBus());
    expect(b.bus).toBe(getDefaultBus());
    expect(hosts()).toHaveLength(1);
    a.detach();
    b.detach();
  });
});
