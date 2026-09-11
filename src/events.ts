/**
 * Event schema. Every `InspectorEvent` must survive `structuredClone` and
 * `JSON.stringify`: plain objects, strings, finite numbers, booleans, null and
 * arrays of those. No bigint, no typed arrays, no DOM nodes. Only request
 * *responses* (`TensorData`) may carry typed arrays.
 */

export type TensorSummary = {
  id: string; // handle for lazy readback via bus.request('tensor', { id })
  name: string; // feed / output name, e.g. 'input_ids', 'logits', 'present.0.key'
  dtype: string; // ORT type string: 'float32' | 'int64' | 'bool' | ...
  dims: number[];
  location: string; // 'cpu' | 'cpu-pinned' | 'gpu-buffer' | 'ml-tensor' | ...
  size: number; // element count
  bytes: number; // size * bytesPerElement(dtype); 0 for 'string'
  head: (number | string)[] | null; // first `head` values; null when not CPU-resident
};

export type InputPreview =
  | { kind: 'text'; text: string } // truncated to 2000 chars
  | { kind: 'texts'; texts: string[] }
  | { kind: 'image'; width?: number; height?: number; channels?: number; src?: string }
  | { kind: 'audio'; samples: number; sampleRate?: number }
  | { kind: 'other'; json: unknown }; // toCloneSafe() of the value

/** `token` is the tokenizer-decoded text; `raw` (optional, v1.1) is the vocab string (`id_to_token`). */
export type TopKEntry = { id: number; token: string | null; logit: number; prob: number; raw?: string | null };

export type InspectorEvent =
  | { type: 'call:start'; callId: string; label: string; task: string | null; input: InputPreview; t: number }
  // `tokens` are decoded per id; `raw` (optional, v1.1) holds the vocab strings in the same shape.
  | { type: 'tokenize'; callId: string | null; text: string | string[]; ids: number[][]; tokens: (string | null)[][]; raw?: (string | null)[][]; ms: number; t: number }
  | { type: 'run:start'; callId: string | null; runId: string; session: string; inputs: TensorSummary[]; t: number }
  | { type: 'run:end'; callId: string | null; runId: string; session: string; outputs: TensorSummary[]; ms: number; error: string | null; t: number }
  | { type: 'logits'; callId: string | null; step: number; vocab: number; topK: TopKEntry[]; tensorId: string | null; t: number }
  // `text` is decoded; `raw` (optional, v1.1) is the vocab strings of `ids` joined.
  | { type: 'token'; callId: string | null; step: number; ids: number[]; text: string | null; raw?: string | null; t: number }
  | { type: 'result'; callId: string; result: unknown; ms: number; error: string | null; t: number };

export type InspectorEventType = InspectorEvent['type'];

// Request/response (not events); responses may carry typed arrays.
export interface RequestMap {
  tensor: { req: { id: string }; res: TensorData };
}

export type TensorData =
  | { id: string; dtype: string; dims: number[]; data: ArrayBufferView | string[] }
  | { id: string; error: 'unknown' | 'evicted' | 'disposed' | string };

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null;
const isNum = (x: unknown): x is number => typeof x === 'number';
const isStr = (x: unknown): x is string => typeof x === 'string';
const isStrOrNull = (x: unknown): x is string | null => x === null || isStr(x);
const isNumArray = (x: unknown): x is number[] => Array.isArray(x) && x.every(isNum);
const isStrOrNullRows = (x: unknown): x is (string | null)[][] => Array.isArray(x) && x.every((row) => Array.isArray(row) && row.every(isStrOrNull));
/** Optional fields are only checked when present, so v0.1 producers still validate. */
const isAbsent = (x: unknown): x is undefined => x === undefined;
const isTensorSummary = (x: unknown): x is TensorSummary =>
  isObj(x) &&
  isStr(x.id) &&
  isStr(x.name) &&
  isStr(x.dtype) &&
  isNumArray(x.dims) &&
  isStr(x.location) &&
  isNum(x.size) &&
  isNum(x.bytes) &&
  (x.head === null || (Array.isArray(x.head) && x.head.every((v) => isNum(v) || isStr(v))));

/** Structural check for a wire-received value; enough to trust `ev.type` and ids. */
export function isInspectorEvent(x: unknown): x is InspectorEvent {
  if (!isObj(x) || !isStr(x.type) || !isNum(x.t)) return false;
  switch (x.type) {
    case 'call:start':
      return isStr(x.callId) && isStr(x.label) && isStrOrNull(x.task) && isObj(x.input) && isStr(x.input.kind);
    case 'tokenize':
      return (
        isStrOrNull(x.callId) &&
        (isStr(x.text) || (Array.isArray(x.text) && x.text.every(isStr))) &&
        Array.isArray(x.ids) &&
        x.ids.every(isNumArray) &&
        isStrOrNullRows(x.tokens) &&
        (isAbsent(x.raw) || isStrOrNullRows(x.raw)) &&
        isNum(x.ms)
      );
    case 'run:start':
      return isStrOrNull(x.callId) && isStr(x.runId) && isStr(x.session) && Array.isArray(x.inputs) && x.inputs.every(isTensorSummary);
    case 'run:end':
      return (
        isStrOrNull(x.callId) &&
        isStr(x.runId) &&
        isStr(x.session) &&
        Array.isArray(x.outputs) &&
        x.outputs.every(isTensorSummary) &&
        isNum(x.ms) &&
        isStrOrNull(x.error)
      );
    case 'logits':
      return (
        isStrOrNull(x.callId) &&
        isNum(x.step) &&
        isNum(x.vocab) &&
        Array.isArray(x.topK) &&
        x.topK.every((e) => isObj(e) && isNum(e.id) && isStrOrNull(e.token) && isNum(e.logit) && isNum(e.prob) && (isAbsent(e.raw) || isStrOrNull(e.raw))) &&
        isStrOrNull(x.tensorId)
      );
    case 'token':
      return isStrOrNull(x.callId) && isNum(x.step) && isNumArray(x.ids) && isStrOrNull(x.text) && (isAbsent(x.raw) || isStrOrNull(x.raw));
    case 'result':
      return isStr(x.callId) && 'result' in x && isNum(x.ms) && isStrOrNull(x.error);
    default:
      return false;
  }
}
