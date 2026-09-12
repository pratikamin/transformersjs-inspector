/**
 * Export of the captured event history: a plain, JSON-serialisable snapshot of
 * `bus.history`. No DOM here; the panel's download/clipboard glue lives in
 * `panel/download.ts`. The bus keeps at most `maxHistory` events (500 by default):
 * construct the bus with a larger `maxHistory` for longer exports. Tensor bytes are
 * never included, only the summaries already in the events.
 */
import type { InspectorBus } from './bus';
import type { InspectorEvent } from './events';
import { VERSION } from './version';

export interface InspectorExport {
  /** Library version that wrote the file. */
  version: string;
  /** ISO 8601 timestamp of the export. */
  exportedAt: string;
  events: InspectorEvent[];
}

export interface ExportOptions {
  /** Keep only events for which this returns `true`. */
  filter?: (ev: InspectorEvent) => boolean;
  /** Keep only the last `limit` events (after `filter`). */
  limit?: number;
}

/** Snapshot of the bus history; the events are the same objects, copied into a new array. */
export function exportEvents(bus: InspectorBus, opts: ExportOptions = {}): InspectorExport {
  let events: InspectorEvent[] = opts.filter ? bus.history.filter(opts.filter) : [...bus.history];
  if (opts.limit !== undefined && opts.limit >= 0 && events.length > opts.limit) {
    events = events.slice(events.length - opts.limit);
  }
  return { version: VERSION, exportedAt: new Date().toISOString(), events };
}

const pad = (n: number, width = 2): string => String(n).padStart(width, '0');

/** `transformersjs-inspector-YYYYMMDD-HHMMSS.json`, in the local time zone. */
export function exportFilename(d: Date = new Date()): string {
  const stamp =
    `${pad(d.getFullYear(), 4)}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return `transformersjs-inspector-${stamp}.json`;
}

/** The file contents: JSON, two-space indented, trailing newline. */
export function serializeExport(x: InspectorExport): string {
  return JSON.stringify(x, null, 2) + '\n';
}
