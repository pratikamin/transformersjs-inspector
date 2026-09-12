import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { InspectorBus } from '../src/bus';
import { isInspectorEvent } from '../src/events';
import { exportEvents, exportFilename, serializeExport } from '../src/export';
import type { InspectorExport } from '../src/export';
import { VERSION } from '../src/version';
import { fixtureEvents } from './fakes';

const FILENAME = /^transformersjs-inspector-\d{8}-\d{6}\.json$/;

const filledBus = (opts?: { maxHistory?: number }): InspectorBus => {
  const bus = new InspectorBus(opts);
  for (const ev of fixtureEvents()) bus.emit(ev);
  return bus;
};

describe('exportEvents', () => {
  test('shape: version from src/version (and package.json), ISO exportedAt, events equal to bus.history', () => {
    const bus = filledBus();
    const before = Date.now();
    const x = exportEvents(bus);
    expect(Object.keys(x).sort()).toEqual(['events', 'exportedAt', 'version']);
    expect(x.version).toBe(VERSION);
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { version: string };
    expect(x.version).toBe(pkg.version);
    expect(x.exportedAt).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
    const t = Date.parse(x.exportedAt);
    expect(t).toBeGreaterThanOrEqual(before - 1);
    expect(t).toBeLessThanOrEqual(Date.now() + 1);
    expect(x.events).toEqual([...bus.history]);
    expect(x.events).toHaveLength(fixtureEvents().length);
  });

  test('the events are copied into a new array but not cloned; the bus history is left alone', () => {
    const bus = filledBus();
    const n = bus.history.length;
    const x = exportEvents(bus);
    expect(x.events).not.toBe(bus.history);
    expect(x.events[0]).toBe(bus.history[0]);
    x.events.length = 0;
    expect(bus.history).toHaveLength(n);
  });

  test('honours the bus maxHistory cap', () => {
    const bus = filledBus({ maxHistory: 3 });
    const x = exportEvents(bus);
    expect(x.events).toHaveLength(3);
    expect(x.events).toEqual(fixtureEvents().slice(-3));
  });

  test('filter keeps matching events; limit keeps the last N after filtering', () => {
    const bus = filledBus();
    const all = fixtureEvents();
    const c1 = exportEvents(bus, { filter: (ev) => ev.callId === 'c1' });
    expect(c1.events.map((e) => e.type)).toEqual(['call:start', 'tokenize', 'run:start', 'run:end', 'result']);
    expect(exportEvents(bus, { limit: 2 }).events).toEqual(all.slice(-2));
    expect(exportEvents(bus, { limit: 0 }).events).toEqual([]);
    expect(exportEvents(bus, { limit: 10_000 }).events).toEqual(all);
    expect(exportEvents(bus, { limit: -1 }).events).toEqual(all);
    const both = exportEvents(bus, { filter: (ev) => ev.type === 'token', limit: 1 });
    expect(both.events).toHaveLength(1);
    expect(both.events[0]).toEqual(all.filter((e) => e.type === 'token').at(-1));
  });
});

describe('exportFilename', () => {
  test('transformersjs-inspector-YYYYMMDD-HHMMSS.json in local time, zero-padded', () => {
    const d = new Date(2026, 8, 11, 9, 5, 7);
    const name = exportFilename(d);
    expect(name).toMatch(FILENAME);
    expect(name).toBe('transformersjs-inspector-20260911-090507.json');
    expect(exportFilename(new Date(Date.UTC(2026, 8, 11, 9, 5, 7)))).toMatch(FILENAME);
    expect(exportFilename()).toMatch(FILENAME);
    expect(exportFilename(new Date(2026, 11, 31, 23, 59, 59))).toBe('transformersjs-inspector-20261231-235959.json');
  });
});

describe('serializeExport', () => {
  test('two-space JSON with a trailing newline that round-trips to the same export', () => {
    const x = exportEvents(filledBus());
    const text = serializeExport(x);
    expect(text.endsWith('}\n')).toBe(true);
    expect(text.startsWith('{\n  "version": ')).toBe(true);
    const back = JSON.parse(text) as InspectorExport;
    expect(back).toEqual(x);
    expect(back.events.every(isInspectorEvent)).toBe(true);
    // Every event is JSON-safe by contract, so a structuredClone agrees too.
    expect(structuredClone(x)).toEqual(back);
  });
});
