import { describe, expect, test } from 'vitest';
import { InspectorBus, InspectorError, loopbackPair } from '../src/bus';
import type { TensorData } from '../src/events';
import { DEFAULT_MAX_BYTES, TensorStore } from '../src/store';
import { fakeTensor } from './fakes';

const flush = () => new Promise<void>((r) => setTimeout(r, 0));
const dataOf = (res: TensorData) => ('data' in res ? res.data : undefined);
const f32 = (n: number, location?: string) => fakeTensor({ type: 'float32', dims: [1, n], location }); // 4n bytes

describe('TensorStore', () => {
  test('defaults: 64 MiB budget, head 8', () => {
    const store = new TensorStore();
    expect(store.maxBytes).toBe(64 * 1024 * 1024);
    expect(DEFAULT_MAX_BYTES).toBe(67_108_864);
    expect(store.head).toBe(8);
    expect(store.bytes).toBe(0);
    expect(store.count).toBe(0);
  });

  test('put/read round trip returns the typed array and the summary metadata', async () => {
    const store = new TensorStore();
    const ids = [101, 1996, 7742, 5927, 2673, 1012, 102];
    const t = fakeTensor({ type: 'int64', dims: [1, 7], data: ids });
    const s = store.put(t, 'input_ids');
    expect(s).toMatchObject({ id: `${store.id}/t1`, name: 'input_ids', dtype: 'int64', dims: [1, 7], size: 7, bytes: 56, head: ids });
    expect(store.has(`${store.id}/t1`)).toBe(true);
    expect(store.count).toBe(1);
    expect(store.bytes).toBe(56);
    const res = await store.read(`${store.id}/t1`);
    expect(res).toMatchObject({ id: `${store.id}/t1`, dtype: 'int64', dims: [1, 7] });
    const data = dataOf(res);
    expect(data).toBeInstanceOf(BigInt64Array);
    expect(Array.from(data as BigInt64Array, Number)).toEqual(ids);
  });

  test('the same tensor object gets the same id; a new object gets a new one', () => {
    const store = new TensorStore();
    const t = f32(4);
    const a = store.put(t, 'logits');
    const b = store.put(t, 'logits');
    expect(b.id).toBe(a.id);
    expect(store.count).toBe(1);
    expect(store.bytes).toBe(16);
    const c = store.put(t, 'result');
    expect(c.id).toBe(a.id);
    expect(c.name).toBe('result');
    expect(store.put(f32(4), 'logits').id).not.toBe(a.id);
    expect(store.count).toBe(2);
  });

  test('gpu tensors are stored without reading their data and read back via getData', async () => {
    const store = new TensorStore();
    const t = f32(96, 'gpu-buffer');
    const s = store.put(t, 'present.0.key');
    expect(s.head).toBeNull();
    expect(t.dataReads).toBe(0);
    const res = await store.read(s.id);
    expect(dataOf(res)).toBeInstanceOf(Float32Array);
    expect(t.dataReads).toBe(0);
  });

  test('head option controls the summary head length', () => {
    const store = new TensorStore({ head: 2 });
    expect(store.put(f32(10), 'x').head).toHaveLength(2);
  });

  test('unknown ids answer { error: "unknown" }', async () => {
    const store = new TensorStore();
    expect(await store.read('t99')).toEqual({ id: 't99', error: 'unknown' });
    expect(store.has('t99')).toBe(false);
  });

  test('oldest insertions are evicted once the byte budget is exceeded', async () => {
    const store = new TensorStore({ maxBytes: 32 }); // two [1,4] float32 tensors
    const a = store.put(f32(4), 'a');
    const b = store.put(f32(4), 'b');
    expect(store.bytes).toBe(32);
    const c = store.put(f32(4), 'c');
    expect(store.count).toBe(2);
    expect(store.bytes).toBe(32);
    expect(store.has(a.id)).toBe(false);
    expect(store.has(b.id)).toBe(true);
    expect(store.has(c.id)).toBe(true);
    expect(await store.read(a.id)).toEqual({ id: a.id, error: 'evicted' });
    expect(dataOf(await store.read(b.id))).toBeInstanceOf(Float32Array);
    // a bigger tensor pushes out as many old entries as needed
    const d = store.put(f32(7), 'd'); // 28 bytes: evicts b and c
    expect(store.count).toBe(1);
    expect(store.bytes).toBe(28);
    expect(await store.read(b.id)).toEqual({ id: b.id, error: 'evicted' });
    expect(await store.read(c.id)).toEqual({ id: c.id, error: 'evicted' });
    expect(store.has(d.id)).toBe(true);
  });

  test('a tensor larger than the whole budget is summarised but immediately evicted', async () => {
    const store = new TensorStore({ maxBytes: 8 });
    const s = store.put(f32(4), 'huge');
    expect(s.head).toHaveLength(4);
    expect(store.count).toBe(0);
    expect(store.bytes).toBe(0);
    expect(await store.read(s.id)).toEqual({ id: s.id, error: 'evicted' });
  });

  test('re-putting an evicted tensor keeps its id and makes it live again', async () => {
    const store = new TensorStore({ maxBytes: 16 });
    const t = f32(4);
    const first = store.put(t, 'a');
    store.put(f32(4), 'b');
    expect(await store.read(first.id)).toMatchObject({ error: 'evicted' });
    const again = store.put(t, 'a');
    expect(again.id).toBe(first.id);
    expect(dataOf(await store.read(first.id))).toBeInstanceOf(Float32Array);
  });

  test('a disposed tensor answers { error: "disposed" }', async () => {
    const store = new TensorStore();
    const t = f32(4);
    const s = store.put(t, 'x');
    t.dispose();
    expect(await store.read(s.id)).toEqual({ id: s.id, error: 'disposed' });
    expect(store.has(s.id)).toBe(true);
  });

  test('clear() drops everything; cleared ids answer evicted', async () => {
    const store = new TensorStore();
    const s = store.put(f32(4), 'x');
    store.clear();
    expect(store.count).toBe(0);
    expect(store.bytes).toBe(0);
    expect(store.has(s.id)).toBe(false);
    expect(await store.read(s.id)).toEqual({ id: s.id, error: 'evicted' });
  });

  test('attachTo(bus) answers bus.request("tensor") with the typed array; detaching removes the handler', async () => {
    const store = new TensorStore();
    const bus = new InspectorBus();
    const off = store.attachTo(bus);
    const s = store.put(fakeTensor({ type: 'float32', dims: [1, 3], data: Float32Array.of(0.5, -1, 2) }), 'logits');
    const res = await bus.request('tensor', { id: s.id });
    expect(res).toMatchObject({ id: s.id, dtype: 'float32', dims: [1, 3] });
    expect(dataOf(res)).toEqual(Float32Array.of(0.5, -1, 2));
    expect(await bus.request('tensor', { id: 'nope' })).toEqual({ id: 'nope', error: 'unknown' });
    off();
    await expect(bus.request('tensor', { id: s.id })).rejects.toBeInstanceOf(InspectorError);
  });

  test('a store attached on one side of a loopbackPair serves requests from the other side', async () => {
    const store = new TensorStore();
    const [a, b] = loopbackPair();
    const host = new InspectorBus();
    const panel = new InspectorBus();
    host.connect(a);
    panel.connect(b);
    store.attachTo(host);
    const s = store.put(f32(4, 'gpu-buffer'), 'present.0.value');
    await flush();
    const res = await panel.request('tensor', { id: s.id });
    expect(res).toMatchObject({ id: s.id, dims: [1, 4] });
    expect(dataOf(res)).toBeInstanceOf(Float32Array);
  });
});
