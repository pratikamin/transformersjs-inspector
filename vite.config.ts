import { defineConfig } from 'vitest/config';
import { ORT_CDN_URL, ORT_SPECIFIER } from './vite.shared.ts';

export default defineConfig({
  build: {
    lib: {
      entry: { index: 'src/index.ts', preload: 'src/preload.ts' },
      formats: ['es'],
    },
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      // The preload's ORT import is left external and rewritten to the CDN build of the
      // exact version Transformers.js pins, so dist/preload.js does not bundle ~2 MB of ORT.
      external: ['@huggingface/transformers', ORT_SPECIFIER],
      output: {
        paths: { [ORT_SPECIFIER]: ORT_CDN_URL },
      },
    },
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
});
