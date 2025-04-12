import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://i3tiny1.local:7020',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/wikipedia/_search'),
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log(`[ProxyReq] ${req.method} ${req.url}`);
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log(`[ProxyRes] ${req.method} ${req.url} -> ${proxyRes.statusCode}`);
          });
        },
      },
    },
  },
  configureServer: (server) => {
    server.middlewares.use((req, res, next) => {
      console.log(`[Proxy] ${req.method} ${req.url}`);
      next();
    });
  },
});
