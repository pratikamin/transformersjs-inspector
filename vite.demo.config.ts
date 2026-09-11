import { defineConfig } from 'vite';

export default defineConfig({
  root: 'demo',
  base: './',
  build: {
    outDir: '../dist-demo',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
