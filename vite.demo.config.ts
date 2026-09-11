import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { ORT_CDN_URL, ORT_SPECIFIER } from './vite.shared.ts';

const page = (name: string): string => fileURLToPath(new URL(`demo/${name}`, import.meta.url));

export default defineConfig({
  root: 'demo',
  base: './',
  resolve: {
    // `src/preload.ts` imports the bare specifier; the dev server and `build:demo` both send
    // the browser to the same CDN file the library build rewrites it to (Vite leaves
    // `https://` specifiers to the browser).
    alias: { [ORT_SPECIFIER]: ORT_CDN_URL },
  },
  build: {
    outDir: '../dist-demo',
    emptyOutDir: true,
    rollupOptions: {
      input: { index: page('index.html'), preload: page('preload.html'), worker: page('worker.html') },
    },
  },
  // `demo/inference.worker.ts` imports `./tf.ts`, whose top-level `await import(CDN)` only
  // works in a *module* worker; the default IIFE worker format cannot carry it.
  worker: { format: 'es' },
  server: {
    port: 5173,
    strictPort: true,
  },
});
