import { describe, expect, test, vi } from 'vitest';
import { InspectorError } from '../src/bus';
import {
  DEFAULT_HEAD,
  bytesPerElement,
  elementCount,
  headOf,
  isCpuResident,
  isTensorLike,
  readTensor,
  summarizeTensor,
} from '../src/summarize';
import { fakeTensor } from './fakes';

describe('dtype table', () => {
  test.each([
    ['float32', 4],
    ['float16', 2],
    ['float64', 8],
    ['int64', 8],
    ['uint64', 8],
    ['int32', 4],
    ['uint32', 4],
    ['int16', 2],
    ['uint16', 2],
    ['int8', 1],
    ['uint8', 1],
    ['bool', 1],
    ['string', 0],
  ])('%s is %i bytes per element', (dtype, bytes) => {
    expect(bytesPerElement(dtype)).toBe(bytes);
  });

  test('unknown dtypes are 0 bytes rather than a throw', () => {
    expect(bytesPerElement('int4')).toBe(0);
  });

  test('elementCount is the product of dims; a scalar has one element', () => {
    expect(elementCount([1, 7, 384])).toBe(2688);
    expect(elementCount([1, 1, 128256])).toBe(128256);
    expect(elementCount([1, 2, 0, 16])).toBe(0);
    expect(elementCount([])).toBe(1);
  });

  test('DEFAULT_HEAD is 8', () => {
    expect(DEFAULT_HEAD).toBe(8);
  });
});

describe('isCpuResident / isTensorLike', () => {
  test.each([
    [undefined, true],
    ['cpu', true],
    ['cpu-pinned', true],
    ['gpu-buffer', false],
    ['ml-tensor', false],
    ['texture', false],
  ])('location %s → %s', (location, expected) => {
    expect(isCpuResident({ type: 'float32', dims: [1], location, data: new Float32Array(1) })).toBe(expected);
  });

  test('isTensorLike accepts ORT-shaped objects without touching data', () => {
    const gpu = fakeTensor({ type: 'float32', dims: [1, 2, 3, 16], location: 'gpu-buffer' });
    expect(isTensorLike(gpu)).toBe(true);
    expect(gpu.dataReads).toBe(0);
    expect(isTensorLike({ type: 'float32', dims: [1] })).toBe(false);
    expect(isTensorLike({ dims: [1], data: [] })).toBe(false);
    expect(isTensorLike(null)).toBe(false);
    expect(isTensorLike('float32')).toBe(false);
  });
});

describe('headOf / summarizeTensor', () => {
  test('int64 head is plain numbers, not bigint', () => {
    const ids = [101, 1996, 7742, 5927, 2673, 1012, 102];
    const t = fakeTensor({ type: 'int64', dims: [1, 7], data: ids });
    expect(t.data).toBeInstanceOf(BigInt64Array);
    const head = headOf(t, 8);
    expect(head).toEqual(ids);
    for (const v of head ?? []) expect(typeof v).toBe('number');
  });

  test('head is capped at n and at the element count', () => {
    const big = fakeTensor({ type: 'float32', dims: [1, 7, 384] });
    expect(headOf(big, 8)).toHaveLength(8);
    expect(headOf(big, 3)).toHaveLength(3);
    expect(headOf(big, 0)).toEqual([]);
    const small = fakeTensor({ type: 'int64', dims: [1, 3], data: [1, 2, 3] });
    expect(headOf(small, 8)).toEqual([1, 2, 3]);
  });

  test('string tensors have 0 bytes and string heads', () => {
    const t = fakeTensor({ type: 'string', dims: [2], data: ['a', 'b'] });
    const s = summarizeTensor(t, 'labels', 't9');
    expect(s).toMatchObject({ dtype: 'string', size: 2, bytes: 0, head: ['a', 'b'] });
  });

  test('summary bytes follow the dtype table', () => {
    const f32 = summarizeTensor(fakeTensor({ type: 'float32', dims: [1, 7, 384] }), 'last_hidden_state', 't4');
    expect(f32).toMatchObject({ id: 't4', name: 'last_hidden_state', dtype: 'float32', dims: [1, 7, 384], location: 'cpu', size: 2688, bytes: 2688 * 4 });
    expect(f32.head).toHaveLength(8);
    const i64 = summarizeTensor(fakeTensor({ type: 'int64', dims: [1, 7] }), 'input_ids', 't1');
    expect(i64.bytes).toBe(56);
    const bool = summarizeTensor(fakeTensor({ type: 'bool', dims: [4] }), 'mask', 't2');
    expect(bool.bytes).toBe(4);
  });

  test('gpu-buffer tensor is summarised with head null and its data getter is never invoked', () => {
    const t = fakeTensor({ type: 'float32', dims: [1, 2, 5, 16], location: 'gpu-buffer' });
    const s = summarizeTensor(t, 'present.0.key', 't7');
    expect(s).toEqual({ id: 't7', name: 'present.0.key', dtype: 'float32', dims: [1, 2, 5, 16], location: 'gpu-buffer', size: 160, bytes: 640, head: null });
    expect(t.dataReads).toBe(0);
    expect(headOf(t, 8)).toBeNull();
    expect(t.dataReads).toBe(0);
  });

  test('a tensor without a location field counts as cpu', () => {
    const s = summarizeTensor({ type: 'int32', dims: [2], data: Int32Array.of(4, 5) }, 'x', 't1');
    expect(s.location).toBe('cpu');
    expect(s.head).toEqual([4, 5]);
  });

  test('a disposed cpu tensor gets head null instead of throwing', () => {
    const t = fakeTensor({ type: 'float32', dims: [3] });
    t.dispose();
    expect(headOf(t, 8)).toBeNull();
    expect(summarizeTensor(t, 'x', 't1').head).toBeNull();
  });

  test('summary dims are a fresh plain array', () => {
    const t = fakeTensor({ type: 'float32', dims: [1, 3] });
    const s = summarizeTensor(t, 'x', 't1');
    expect(s.dims).toEqual([1, 3]);
    expect(s.dims).not.toBe(t.dims);
  });
});

describe('readTensor', () => {
  test('cpu tensor returns .data without calling getData', async () => {
    const t = fakeTensor({ type: 'float32', dims: [1, 4] });
    const getData = vi.spyOn(t, 'getData');
    const data = await readTensor(t);
    expect(data).toBeInstanceOf(Float32Array);
    expect((data as Float32Array).length).toBe(4);
    expect(getData).not.toHaveBeenCalled();
    expect(t.dataReads).toBe(1);
  });

  test('gpu tensor uses getData() and never reads .data', async () => {
    const t = fakeTensor({ type: 'float32', dims: [1, 2, 3, 16], location: 'gpu-buffer' });
    const getData = vi.spyOn(t, 'getData');
    const data = await readTensor(t);
    expect(data).toBeInstanceOf(Float32Array);
    expect((data as Float32Array).length).toBe(96);
    expect(getData).toHaveBeenCalledTimes(1);
    expect(t.dataReads).toBe(0);
  });

  test('string tensors read back as string[]', async () => {
    const t = fakeTensor({ type: 'string', dims: [2], data: ['x', 'y'] });
    expect(await readTensor(t)).toEqual(['x', 'y']);
  });

  test('disposed tensors reject with InspectorError("disposed") on either path', async () => {
    const cpu = fakeTensor({ type: 'int64', dims: [2], data: [1, 2] });
    cpu.dispose();
    await expect(readTensor(cpu)).rejects.toThrow(InspectorError);
    await expect(readTensor(cpu)).rejects.toThrow('disposed');
    const gpu = fakeTensor({ type: 'float32', dims: [2], location: 'gpu-buffer' });
    gpu.dispose();
    await expect(readTensor(gpu)).rejects.toThrow(InspectorError);
    await expect(readTensor(gpu)).rejects.toThrow('disposed');
  });

  test('a non-cpu tensor without getData rejects with an InspectorError', async () => {
    const t = { type: 'float32', dims: [2], location: 'ml-tensor', data: new Float32Array(2) };
    await expect(readTensor(t)).rejects.toThrow(InspectorError);
  });
});
