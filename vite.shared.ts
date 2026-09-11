/**
 * Build-time constants shared by `vite.config.ts` (library) and `vite.demo.config.ts` (demo).
 * The ORT pin is derived from the installed `@huggingface/transformers`, never hand-copied:
 * the preload loads its own copy of onnxruntime-web from the CDN and it must be the exact
 * version Transformers.js was built against (docs/01-research.md, "ORT version coupling").
 * The package's `exports` map does not expose `./package.json`, so the file is located by
 * walking up from the resolved entry point.
 */
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

const TRANSFORMERS = '@huggingface/transformers';

function transformersPackageJson(): { version: string; dependencies?: Record<string, string> } {
  let dir = dirname(createRequire(import.meta.url).resolve(TRANSFORMERS));
  for (;;) {
    const candidate = join(dir, 'package.json');
    if (existsSync(candidate)) {
      const pkg = JSON.parse(readFileSync(candidate, 'utf8')) as { name?: string; version: string; dependencies?: Record<string, string> };
      if (pkg.name === TRANSFORMERS) return pkg;
    }
    const parent = dirname(dir);
    if (parent === dir) throw new Error(`vite.shared: could not find ${TRANSFORMERS}/package.json`);
    dir = parent;
  }
}

const pkg = transformersPackageJson();

/** Installed Transformers.js version (informational). */
export const TRANSFORMERS_VERSION: string = pkg.version;

/** The onnxruntime-web version `@huggingface/transformers` depends on. */
export const ORT_VERSION: string = (() => {
  const v = pkg.dependencies?.['onnxruntime-web'];
  if (typeof v !== 'string' || v.length === 0) throw new Error(`vite.shared: ${TRANSFORMERS} has no onnxruntime-web dependency`);
  return v;
})();

/** The verified spike URL: the WebGPU bundle of exactly that version on jsDelivr. */
export const ORT_CDN_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.bundle.min.mjs`;

/** The bare specifier `src/preload.ts` imports; both configs redirect it to `ORT_CDN_URL`. */
export const ORT_SPECIFIER = 'onnxruntime-web/webgpu';
