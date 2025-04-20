import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy all /api calls to test_api.py Flask server for search and query endpoints
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true
        // no rewrite: backend routes handle /api/search, /api/queries, etc.
      }
    },
  },
  configureServer: (server) => {
    server.middlewares.use((req, res, next) => {
      console.log(`[Proxy] ${req.method} ${req.url}`);
      next();
    });
  },
});
