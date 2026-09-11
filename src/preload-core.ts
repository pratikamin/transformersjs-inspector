/**
 * Zero-touch interception, as verified in `spike/index.html`: Transformers.js reads
 * `globalThis[Symbol.for('onnxruntime')]` at module evaluation and, when present, uses it
 * instead of its bundled onnxruntime-web. `installPreload` hands it a module-shaped shim of
 * the caller's ORT whose `InferenceSession.create` wraps every session it returns with
 * `wrapSession`, so an unmodified `pipeline()` is observed at the `session.run` boundary.
 * Everything else (`env`, `Tensor`, backends) is the original object: Transformers.js
 * configures `env.wasm` through the shim and checks `x instanceof Tensor`.
 *
 * Caveat (docs/01-research.md): with the symbol set, Transformers.js's device list is
 * empty, so the host must pass `device: 'auto'` (which hands ORT an empty provider list
 * and lets it pick its own default) or `pipeline()` throws `Unsupported device: "wasm"`.
 */
import type { WrapContext } from './context';
import type { SessionLike } from './types';
import { wrapSession } from './wrap/session';

export interface OrtLike {
  InferenceSession: { create(...args: unknown[]): Promise<SessionLike> };
  env: unknown;
  Tensor: unknown;
  [k: string]: unknown;
}

/** The key Transformers.js 4.x checks for an injected ONNX runtime. */
export const ORT_SYMBOL: unique symbol = Symbol.for('onnxruntime');

/** Marks a shim built here so a second install (or a differently bundled copy) is a no-op. */
const SHIM_MARK: unique symbol = Symbol.for('transformersjs-inspector.preload');

type GlobalWithOrt = Record<typeof ORT_SYMBOL, unknown>;

function currentGlobal(): unknown {
  return (globalThis as unknown as GlobalWithOrt)[ORT_SYMBOL];
}

function isShim(x: unknown): x is OrtLike {
  return (typeof x === 'object' || typeof x === 'function') && x !== null && (x as Record<symbol, unknown>)[SHIM_MARK] === true;
}

/** Whether `globalThis[ORT_SYMBOL]` currently holds a shim made by `installPreload`. */
export function isInstalled(): boolean {
  return isShim(currentGlobal());
}

/**
 * Session label for the panel: the file name of a string/URL argument without `.onnx`
 * (`.../onnx/model_quantized.onnx` → `model_quantized`), else — Transformers.js in the
 * browser passes a `Uint8Array` — `session#<n> → <first output name>`.
 */
export function nameFor(args: readonly unknown[], session: SessionLike, n: number): string {
  const first = args[0];
  const path = typeof first === 'string' ? first : first instanceof URL ? first.pathname : null;
  if (path !== null) {
    const last = path.split(/[\\/]/).filter(Boolean).pop() ?? path;
    const stripped = last.replace(/\.onnx$/i, '');
    if (stripped.length > 0) return stripped;
  }
  const out = Array.isArray(session.outputNames) ? session.outputNames[0] : undefined;
  return `session#${n} → ${typeof out === 'string' ? out : '?'}`;
}

/**
 * Builds the shim, sets `globalThis[ORT_SYMBOL]` and returns it. Calling it again while a
 * shim is installed returns that shim unchanged (whatever `ort` is passed).
 */
export function installPreload(ort: OrtLike, ctx: WrapContext): OrtLike {
  const existing = currentGlobal();
  if (isShim(existing)) return existing;

  const Real = ort.InferenceSession;
  let created = 0;
  const create = async (...args: unknown[]): Promise<SessionLike> => {
    const session = await Real.create(...args);
    created += 1;
    wrapSession(session, nameFor(args, session, created), ctx);
    return session;
  };
  // Verified shape: the wrapped factory inherits from the real one, so every other static
  // member (and `instanceof` checks against it) still resolves to the original.
  const InferenceSession = Object.assign(Object.create(Real) as OrtLike['InferenceSession'], Real, { create });
  const shim: OrtLike = { ...ort, InferenceSession };
  Object.defineProperty(shim, SHIM_MARK, { value: true, enumerable: false });
  (globalThis as unknown as GlobalWithOrt)[ORT_SYMBOL] = shim;
  return shim;
}
