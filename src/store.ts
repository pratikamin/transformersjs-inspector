/**
 * Strong-ref, byte-budgeted tensor store. `put` summarises a tensor and keeps a reference
 * so its full contents can be fetched later by id (`read`, or `bus.request('tensor')` once
 * `attachTo` has registered the handler). Oldest insertions are evicted first once the
 * nominal byte total (`summary.bytes`) exceeds `maxBytes`.
 */
import type { InspectorBus } from './bus';
import type { TensorData, TensorSummary } from './events';
import type { TensorLike } from './types';
import { DEFAULT_HEAD, readTensor, summarizeTensor } from './summarize';

export const DEFAULT_MAX_BYTES = 64 * 1024 * 1024;

type Entry = { tensor: TensorLike; summary: TensorSummary };

export class TensorStore {
  /** Store namespace also routes lazy readback to the owning page or worker. */
  readonly id = `s${crypto.randomUUID()}`;
  readonly maxBytes: number;
  readonly head: number;
  /** Live entries in insertion order (Map preserves it); the first key is the eviction candidate. */
  private readonly entries = new Map<string, Entry>();
  private readonly evicted = new Set<string>();
  private readonly ids = new WeakMap<object, string>();
  private seq = 0;
  private _bytes = 0;

  constructor(opts: { maxBytes?: number; head?: number } = {}) {
    this.maxBytes = Math.max(0, opts.maxBytes ?? DEFAULT_MAX_BYTES);
    this.head = Math.max(0, opts.head ?? DEFAULT_HEAD);
  }

  /** Nominal bytes of the live entries (element count × dtype width, not copied memory). */
  get bytes(): number {
    return this._bytes;
  }

  get count(): number {
    return this.entries.size;
  }

  /**
   * Summarise `t` and keep it. The same tensor object always yields the same id, even
   * after eviction; a re-`put` of an evicted tensor makes it live again.
   */
  put(t: TensorLike, name: string): TensorSummary {
    let id = this.ids.get(t);
    if (id === undefined) {
      id = `${this.id}/t${++this.seq}`;
      this.ids.set(t, id);
    }
    const live = this.entries.get(id);
    if (live) return live.summary.name === name ? live.summary : { ...live.summary, name };

    const summary = summarizeTensor(t, name, id, this.head);
    this.evicted.delete(id);
    this.entries.set(id, { tensor: t, summary });
    this._bytes += summary.bytes;
    this.evictToBudget();
    return summary;
  }

  has(id: string): boolean {
    return this.entries.has(id);
  }

  async read(id: string): Promise<TensorData> {
    const entry = this.entries.get(id);
    if (!entry) return { id, error: this.evicted.has(id) ? 'evicted' : 'unknown' };
    try {
      const data = await readTensor(entry.tensor);
      return { id, dtype: entry.summary.dtype, dims: entry.summary.dims, data };
    } catch (e: unknown) {
      return { id, error: e instanceof Error ? e.message : String(e) };
    }
  }

  /** Drops every live entry; their ids answer `evicted` afterwards. */
  clear(): void {
    for (const id of this.entries.keys()) this.evicted.add(id);
    this.entries.clear();
    this._bytes = 0;
  }

  /** Registers this store as the bus's `'tensor'` request handler; returns the unregister function. */
  attachTo(bus: InspectorBus): () => void {
    return bus.handle('tensor', ({ id }) => this.read(id), { scope: this.id });
  }

  private evictToBudget(): void {
    for (const [id, entry] of this.entries) {
      if (this._bytes <= this.maxBytes) return;
      this.entries.delete(id);
      this.evicted.add(id);
      this._bytes -= entry.summary.bytes;
    }
  }
}
