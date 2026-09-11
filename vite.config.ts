import { defineConfig } from 'vitest/config';

export default defineConfig({
  build: {
    lib: {
      entry: { index: 'src/index.ts' },
      formats: ['es'],
    },
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      external: ['@huggingface/transformers'],
    },
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
});
