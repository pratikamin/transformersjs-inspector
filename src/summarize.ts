/**
 * Eager, cheap tensor summaries. `summarizeTensor` reads at most `head` values and
 * only when the tensor is CPU-resident; GPU / WebNN tensors are never read back here
 * (that is `readTensor`, called lazily by the `TensorStore` on request).
 */
import type { TensorSummary } from './events';
import type { TensorLike } from './types';
import { InspectorError } from './bus';

export const DEFAULT_HEAD = 8;

const BYTES_PER_ELEMENT: Readonly<Record<string, number>> = {
  float32: 4,
  float16: 2,
  float64: 8,
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
  bool: 1,
  string: 0,
};

/** Bytes per element for an ORT dtype string; `string` and unknown dtypes are 0. */
export function bytesPerElement(dtype: string): number {
  return BYTES_PER_ELEMENT[dtype] ?? 0;
}

/** Product of `dims`; a scalar (`[]`) has one element. */
export function elementCount(dims: readonly number[]): number {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

const CPU_LOCATIONS: ReadonlySet<string> = new Set(['cpu', 'cpu-pinned']);

/** True when `.data` can be read synchronously without a device round trip. */
export function isCpuResident(t: TensorLike): boolean {
  return t.location === undefined || CPU_LOCATIONS.has(t.location);
}

/** Duck-typed check used by `toCloneSafe` and the wrappers; never invokes the `data` getter. */
export function isTensorLike(x: unknown): x is TensorLike {
  if (typeof x !== 'object' || x === null) return false;
  const o = x as Record<string, unknown>;
  return typeof o.type === 'string' && Array.isArray(o.dims) && ('data' in o || typeof o.getData === 'function');
}

function toHeadValue(v: unknown): number | string {
  switch (typeof v) {
    case 'number':
      return v;
    case 'bigint':
      return Number(v);
    case 'string':
      return v;
    case 'boolean':
      return v ? 1 : 0;
    default:
      return String(v);
  }
}

/**
 * `.data` of a CPU-resident tensor; `null` when it is device-resident (the getter is never
 * touched) or when reading it throws (disposed). The one sanctioned way for a wrapper to
 * look at tensor contents synchronously.
 */
export function cpuData(t: TensorLike): ArrayLike<unknown> | null {
  if (!isCpuResident(t)) return null;
  try {
    return t.data;
  } catch {
    return null;
  }
}

/**
 * First `n` values as plain numbers / strings (bigint → Number). `null` when the tensor is
 * not CPU-resident (the `data` getter is never touched) or when reading it throws (disposed).
 */
export function headOf(t: TensorLike, n: number = DEFAULT_HEAD): (number | string)[] | null {
  const data = cpuData(t);
  if (data === null) return null;
  const len = Math.min(Math.max(0, n), data.length);
  const out: (number | string)[] = new Array<number | string>(len);
  for (let i = 0; i < len; i++) out[i] = toHeadValue(data[i]);
  return out;
}

export function summarizeTensor(t: TensorLike, name: string, id: string, head: number = DEFAULT_HEAD): TensorSummary {
  const dims = Array.from(t.dims, Number);
  const size = elementCount(dims);
  return {
    id,
    name,
    dtype: t.type,
    dims,
    location: t.location ?? 'cpu',
    size,
    bytes: size * bytesPerElement(t.type),
    head: headOf(t, head),
  };
}

function asReadable(d: ArrayLike<unknown>): ArrayBufferView | string[] {
  if (ArrayBuffer.isView(d)) return d;
  const arr = Array.from(d);
  if (arr.every((v) => typeof v === 'string')) return arr as string[];
  if (arr.every((v) => typeof v === 'bigint')) return BigInt64Array.from(arr as bigint[]);
  return Float64Array.from(arr, Number);
}

/**
 * Full tensor contents: `.data` when CPU-resident, otherwise `await getData()` (a device
 * readback). Rejects with `InspectorError('disposed')` when either throws.
 */
export async function readTensor(t: TensorLike): Promise<ArrayBufferView | string[]> {
  let data: ArrayLike<unknown>;
  if (isCpuResident(t)) {
    try {
      data = t.data;
    } catch {
      throw new InspectorError('disposed');
    }
  } else {
    if (typeof t.getData !== 'function') throw new InspectorError('unreadable');
    try {
      data = await t.getData();
    } catch {
      throw new InspectorError('disposed');
    }
  }
  return asReadable(data);
}
