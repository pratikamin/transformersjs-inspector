/**
 * Structural types for the Transformers.js / onnxruntime-web objects the
 * inspector touches. `src/` never imports either package at runtime; anything
 * that quacks like these shapes can be wrapped.
 */

export interface TensorLike {
  type: string;
  dims: readonly number[];
  location?: string;
  /** Throws on non-CPU-resident tensors (gpu-buffer, ml-tensor). Use `getData()` there. */
  readonly data: ArrayLike<unknown>;
  getData?(): Promise<ArrayLike<unknown>>;
}

export interface SessionLike {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, TensorLike>, ...rest: unknown[]): Promise<Record<string, TensorLike>>;
}

export interface TokenizerLike {
  _call(text: unknown, opts?: unknown): { input_ids: TensorLike; [k: string]: unknown };
  _tokenizer?: { id_to_token?(id: number): string | undefined };
  decode?(ids: number[]): string;
}

export interface StreamerLike {
  put(value: unknown): void;
  end(): void;
}

export type LogitsProcessorLike = (input_ids: unknown, logits: TensorLike) => TensorLike;

export interface GenerateLike {
  generate(opts: Record<string, unknown>): Promise<unknown>;
}

export interface PipelineLike {
  task?: string;
  model: {
    sessions: Record<string, SessionLike>;
    config?: Record<string, unknown>;
    generate?: GenerateLike['generate'];
  };
  tokenizer?: TokenizerLike;
  _call(...args: unknown[]): Promise<unknown>;
}
