/**
 * The preload's CDN pin must be the onnxruntime-web version the installed Transformers.js
 * depends on (docs/01-research.md, "ORT version coupling"). Offline: filesystem only.
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { ORT_CDN_URL, ORT_SPECIFIER, ORT_VERSION, TRANSFORMERS_VERSION } from '../vite.shared';

const readPkg = (rel: string): { version: string; dependencies?: Record<string, string> } =>
  JSON.parse(readFileSync(new URL(rel, import.meta.url), 'utf8'));

describe('ORT version pin', () => {
  test('ORT_VERSION equals node_modules/@huggingface/transformers/package.json dependencies["onnxruntime-web"]', () => {
    const pkg = readPkg('../node_modules/@huggingface/transformers/package.json');
    expect(typeof ORT_VERSION).toBe('string');
    expect(ORT_VERSION.length).toBeGreaterThan(0);
    expect(ORT_VERSION).toBe(pkg.dependencies?.['onnxruntime-web']);
    expect(TRANSFORMERS_VERSION).toBe(pkg.version);
  });

  test('the installed onnxruntime-web (transitive) is that exact version and ships the WebGPU bundle', () => {
    expect(readPkg('../node_modules/onnxruntime-web/package.json').version).toBe(ORT_VERSION);
    expect(existsSync(new URL('../node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs', import.meta.url))).toBe(true);
  });

  test('ORT_CDN_URL is the verified spike URL for that version', () => {
    expect(ORT_CDN_URL).toBe(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.bundle.min.mjs`);
    expect(ORT_SPECIFIER).toBe('onnxruntime-web/webgpu');
  });

  test('src/preload.ts is the only file under src/ importing onnxruntime-web, and it uses the bare specifier', () => {
    const root = new URL('../src/', import.meta.url);
    const importers = readdirSync(root, { recursive: true, encoding: 'utf8' })
      .filter((f) => f.endsWith('.ts'))
      .filter((f) => /from\s+['"]onnxruntime-web/.test(readFileSync(new URL(f, root), 'utf8')));
    expect(importers).toEqual(['preload.ts']);
    expect(readFileSync(new URL('preload.ts', root), 'utf8')).toContain(`from '${ORT_SPECIFIER}'`);
  });

  test('package.json sideEffects lists the side-effect-only source modules, or Vite drops the preload from build:demo', () => {
    // Vite applies the project's own `sideEffects` to source files in build mode; a module
    // imported for its side effects alone is tree-shaken unless it is listed here.
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { sideEffects: string[] };
    expect(pkg.sideEffects).toEqual(expect.arrayContaining(['./dist/preload.js', './src/preload.ts', './demo/preload-entry.ts']));
  });
});
