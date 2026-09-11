/**
 * Transformers.js for the demo comes from the jsDelivr CDN at runtime (the verified spike
 * path); the npm package is only a dev dependency, so the cast borrows its types. The
 * `@vite-ignore` hint and the https specifier keep Vite from touching the import, so the
 * browser fetches the module directly (and its ORT/WASM assets alongside it). The type-only
 * namespace import is erased by `verbatimModuleSyntax`; nothing from npm reaches the bundle.
 */
import type * as Transformers from '@huggingface/transformers';

export const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

export const tf = (await import(/* @vite-ignore */ TRANSFORMERS_URL)) as typeof Transformers;
