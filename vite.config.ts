import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the static build also works when hosted from a sub-path.
  base: './',
  server: {
    port: 5173,
    host: true,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});
