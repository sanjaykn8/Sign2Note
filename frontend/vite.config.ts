import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import fs from "fs";
import { componentTagger } from "lovable-tagger";

// Optional HTTPS for the dev server. Browsers only allow camera and screen
// capture on secure pages (https:// or localhost), so OTHER devices that
// open http://<main-laptop-ip>:8080 can upload videos but cannot use the
// live modes. Point these at a certificate for the laptop's LAN address
// (e.g. generated with mkcert) to serve https:// -- see SETUP.md.
function httpsOptions() {
  const key = process.env.SIGN2NOTES_HTTPS_KEY;
  const cert = process.env.SIGN2NOTES_HTTPS_CERT;
  if (!key || !cert) return undefined;
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    https: httpsOptions(),
    hmr: {
      overlay: false,
    },
  },
  preview: {
    host: "::",
    port: 8080,
    https: httpsOptions(),
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime", "@tanstack/react-query", "@tanstack/query-core"],
  },
}));
