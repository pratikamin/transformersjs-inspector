import { describe, expect, test, vi } from 'vitest';
import { InspectorBus } from '../src/bus';
import { WrapContext } from '../src/context';
import type { InspectorEvent } from '../src/events';
import { isInspectorEvent } from '../src/events';
import { TensorStore } from '../src/store';
import type { TensorLike, TokenizerLike } from '../src/types';
import { isWrapped } from '../src/wrap/session';
import { idsFrom, wrapTokenizer } from '../src/wrap/tokenizer';
import { fakeTensor, fakeTokenizer } from './fakes';

function setup() {
  const bus = new InspectorBus();
  const ctx = new WrapContext(bus, new TensorStore());
  const tok = fakeTokenizer();
  return { bus, ctx, tok };
}

const tokenizeEvents = (bus: InspectorBus) => bus.history.filter((e): e is Extract<InspectorEvent, { type: 'tokenize' }> => e.type === 'tokenize');

const FOX_IDS = [101, 1996, 7742, 5927, 2673, 1012, 102];
const FOX_TOKENS = ['[CLS]', 'the', 'quick', 'brown', 'fox', '.', '[SEP]'];

describe('idsFrom', () => {
  test('int64 [1,n] and [2,n] tensors become number rows', () => {
    expect(idsFrom(fakeTensor({ type: 'int64', dims: [1, 3], data: [5, 6, 7] }))).toEqual([[5, 6, 7]]);
    expect(idsFrom(fakeTensor({ type: 'int64', dims: [2, 2], data: [1, 2, 3, 4] }))).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  test('int32 data and 1-D dims work too', () => {
    expect(idsFrom(fakeTensor({ type: 'int32', dims: [2, 3], data: Int32Array.from([1, 2, 3, 4, 5, 6]) }))).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(idsFrom(fakeTensor({ type: 'int32', dims: [3], data: Int32Array.from([9, 8, 7]) }))).toEqual([[9, 8, 7]]);
    expect(idsFrom(fakeTensor({ type: 'int64', dims: [1, 0], data: [] }))).toEqual([[]]);
  });

  test('never reads a non-CPU tensor and tolerates a disposed one', () => {
    const gpu = fakeTensor({ type: 'int64', dims: [1, 2], location: 'gpu-buffer' });
    expect(idsFrom(gpu)).toEqual([]);
    expect(gpu.dataReads).toBe(0);
    const gone = fakeTensor({ type: 'int64', dims: [1, 2] });
    gone.dispose();
    expect(idsFrom(gone)).toEqual([]);
  });
});

describe('wrapTokenizer', () => {
  test('[1,n] int64: tokenize carries ids as number[][], tokens from _tokenizer.id_to_token, text and ms', () => {
    const { bus, ctx, tok } = setup();
    const original = tok._call;
    wrapTokenizer(tok, ctx);
    expect(ctx.tokenizer).toBe(tok);
    const result = tok._call('the quick brown fox.', { padding: true });
    expect(result.input_ids.dims).toEqual([1, 7]);
    expect(tok.calls).toEqual([{ text: 'the quick brown fox.', opts: { padding: true } }]);

    const events = tokenizeEvents(bus);
    expect(events).toHaveLength(1);
    const [ev] = events;
    expect(ev).toMatchObject({ callId: null, text: 'the quick brown fox.', ids: [FOX_IDS], tokens: [FOX_TOKENS], raw: [FOX_TOKENS] });
    expect(ev.ms).toBeGreaterThanOrEqual(0);
    expect(ev.ids[0].every((id) => typeof id === 'number')).toBe(true);
    expect(isInspectorEvent(ev)).toBe(true);
    expect(structuredClone(ev)).toEqual(ev);
    expect(JSON.parse(JSON.stringify(ev))).toEqual(ev);
    expect(tok._call).not.toBe(original);
  });

  test('batch [2,n]: one row per text, padding ids resolved to [PAD]', () => {
    const { bus, ctx, tok } = setup();
    wrapTokenizer(tok, ctx);
    ctx.currentCallId = 'c3';
    const result = tok._call(['hi world', 'hello']);
    expect(result.input_ids.dims).toEqual([2, 4]);
    const [ev] = tokenizeEvents(bus);
    expect(ev).toMatchObject({
      callId: 'c3',
      text: ['hi world', 'hello'],
      ids: [
        [101, 7632, 2088, 102],
        [101, 7592, 102, 0],
      ],
      tokens: [
        ['[CLS]', 'hi', 'world', '[SEP]'],
        ['[CLS]', 'hello', '[SEP]', '[PAD]'],
      ],
      raw: [
        ['[CLS]', 'hi', 'world', '[SEP]'],
        ['[CLS]', 'hello', '[SEP]', '[PAD]'],
      ],
    });
    expect(isInspectorEvent(ev)).toBe(true);
    expect(structuredClone(ev)).toEqual(ev);
  });

  test('a tokenizer without _call warns once per instance and returns a no-op restore', () => {
    const { bus, ctx } = setup();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const broken = { _tokenizer: { id_to_token: () => 'x' } } as unknown as TokenizerLike;
      const restore = wrapTokenizer(broken, ctx);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0][0])).toContain('_call');
      expect(() => wrapTokenizer(broken, ctx)).not.toThrow();
      expect(warn).toHaveBeenCalledTimes(1);
      expect(() => restore()).not.toThrow();
      expect(isWrapped(broken)).toBe(false);
      expect(ctx.tokenizer).toBeNull();
      expect(bus.history).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  test('wrapping twice is a no-op; restore reinstates the original _call identity and stops emitting', () => {
    const { bus, ctx, tok } = setup();
    const original = tok._call;
    const restore1 = wrapTokenizer(tok, ctx);
    const wrapped = tok._call;
    const restore2 = wrapTokenizer(tok, ctx);
    expect(tok._call).toBe(wrapped);
    tok._call('hi');
    expect(tokenizeEvents(bus)).toHaveLength(1);

    restore2();
    expect(tok._call).toBe(wrapped);
    restore1();
    expect(tok._call).toBe(original);
    expect(Object.prototype.hasOwnProperty.call(tok, '_call')).toBe(true); // the fake's own property is back
    expect(isWrapped(tok)).toBe(false);
    expect(ctx.tokenizer).toBeNull();
    tok._call('hi');
    expect(tokenizeEvents(bus)).toHaveLength(1);
    expect(tok.calls).toHaveLength(2);
  });

  test('restore of a prototype _call (the real Callable shape) deletes the shadowing own property', () => {
    const { bus, ctx } = setup();
    const inner = fakeTokenizer();
    class Callable {
      _tokenizer = inner._tokenizer;
      _call(text: unknown, opts?: unknown): { input_ids: TensorLike } {
        return inner._call(text, opts);
      }
    }
    const tok = new Callable();
    const restore = wrapTokenizer(tok, ctx);
    expect(Object.prototype.hasOwnProperty.call(tok, '_call')).toBe(true);
    tok._call('fox');
    expect(tokenizeEvents(bus)[0]).toMatchObject({ ids: [[101, 2673, 102]], tokens: [['[CLS]', 'fox', '[SEP]']] });
    restore();
    expect(Object.prototype.hasOwnProperty.call(tok, '_call')).toBe(false);
    expect(tok._call).toBe(Callable.prototype._call);
  });

  test('does not replace a tokenizer already set on the context', () => {
    const { ctx, tok } = setup();
    const other = fakeTokenizer();
    ctx.tokenizer = other;
    const restore = wrapTokenizer(tok, ctx);
    expect(ctx.tokenizer).toBe(other);
    restore();
    expect(ctx.tokenizer).toBe(other);
  });

  test('a result without an input_ids tensor yields empty ids and no throw', () => {
    const { bus, ctx } = setup();
    const odd = { _call: (text: unknown) => ({ input_ids: text as TensorLike, raw: true }) } as TokenizerLike;
    wrapTokenizer(odd, ctx);
    expect(odd._call('x')).toEqual({ input_ids: 'x', raw: true });
    expect(tokenizeEvents(bus)[0]).toMatchObject({ text: 'x', ids: [], tokens: [] });
  });
});

describe('WrapContext.tokenStrings', () => {
  test('raw is the vocab string, text is decode([id]) with specials kept and no clean-up', () => {
    const { ctx, tok } = setup();
    ctx.tokenizer = tok;
    const decode = vi.spyOn(tok, 'decode');
    expect(ctx.tokenStrings(1996)).toEqual({ raw: 'the', text: 'the' });
    expect(ctx.tokenStrings(101)).toEqual({ raw: '[CLS]', text: '[CLS]' });
    expect(ctx.tokenStrings(2075)).toEqual({ raw: '##ing', text: 'ing' });
    expect(ctx.tokenStrings(2143)).toEqual({ raw: 'Ġfilm', text: ' film' });
    expect(decode).toHaveBeenLastCalledWith([2143], { skip_special_tokens: false, clean_up_tokenization_spaces: false });
    // memoised per id: a repeat lookup does not decode again
    const calls = decode.mock.calls.length;
    expect(ctx.tokenStrings(2143)).toEqual({ raw: 'Ġfilm', text: ' film' });
    expect(decode.mock.calls.length).toBe(calls);
  });

  test('without decode text === raw; a throwing decode yields raw only; swapping the tokenizer drops the memo', () => {
    const { ctx, tok } = setup();
    const noDecode = { _call: tok._call, _tokenizer: tok._tokenizer } as TokenizerLike;
    ctx.tokenizer = noDecode;
    expect(ctx.tokenStrings(2075)).toEqual({ raw: '##ing', text: '##ing' });

    const throwingDecode = {
      _call: tok._call,
      _tokenizer: tok._tokenizer,
      decode: (): string => {
        throw new Error('no decoder');
      },
    } as TokenizerLike;
    ctx.tokenizer = throwingDecode;
    expect(ctx.tokenStrings(2075)).toEqual({ raw: '##ing', text: '##ing' });
    expect(ctx.tokenStrings(424242)).toEqual({ raw: null, text: null });

    ctx.tokenizer = tok;
    expect(ctx.tokenStrings(2075)).toEqual({ raw: '##ing', text: 'ing' });
    ctx.tokenizer = null;
    expect(ctx.tokenStrings(2075)).toEqual({ raw: null, text: null });
  });

  test('tokenToString prefers _tokenizer.id_to_token, falls back to decode([id]), then null; never throws', () => {
    const { ctx, tok } = setup();
    ctx.tokenizer = tok;
    expect(ctx.tokenToString(1996)).toBe('the');
    expect(ctx.tokenToString(0)).toBe('[PAD]');

    const decodeOnly = { _call: tok._call, decode: (ids: number[]) => `<${ids.join(',')}>` } as TokenizerLike;
    ctx.tokenizer = decodeOnly;
    expect(ctx.tokenToString(5)).toBe('<5>');

    const unknownThenDecode = {
      _call: tok._call,
      _tokenizer: { id_to_token: () => undefined },
      decode: (ids: number[]) => `d${ids[0]}`,
    } as TokenizerLike;
    ctx.tokenizer = unknownThenDecode;
    expect(ctx.tokenToString(9)).toBe('d9');

    const throwing = {
      _call: tok._call,
      _tokenizer: {
        id_to_token: (): string => {
          throw new Error('moved');
        },
      },
    } as TokenizerLike;
    ctx.tokenizer = throwing;
    expect(ctx.tokenToString(1)).toBeNull();

    ctx.tokenizer = { _call: tok._call } as TokenizerLike;
    expect(ctx.tokenToString(1)).toBeNull();
    ctx.tokenizer = null;
    expect(ctx.tokenToString(1)).toBeNull();
  });
});
