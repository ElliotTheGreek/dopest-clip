import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import electron from "vite-plugin-electron";
import renderer from "vite-plugin-electron-renderer";

// Vite + Electron config.
//   `npm run dev`   -> Vite dev server for the renderer + builds & launches Electron
//                      main/preload, which spawns the python sidecar.
//   `npm run build` -> bundles the renderer to dist/ and (separately, via tsc) the
//                      electron main/preload to dist-electron/.
// Electron loads the built renderer over file://, where a `<script type=module
// crossorigin>` tag (which Vite emits by default) is fetched in CORS mode against an
// opaque origin and silently blocked — the module never runs and #root stays empty.
// Strip the crossorigin attributes so the same-origin file:// module loads.
function stripCrossorigin() {
  return {
    name: "strip-crossorigin",
    transformIndexHtml(html: string) {
      return html.replace(/\s+crossorigin/g, "");
    },
  };
}

export default defineConfig({
  plugins: [
    stripCrossorigin(),
    react(),
    electron([
      {
        // Main process.
        entry: "electron/main.ts",
        onstart(args) {
          args.startup();
        },
        vite: {
          build: {
            outDir: "dist-electron",
            rollupOptions: { external: ["electron"] },
          },
        },
      },
      {
        // Preload script.
        entry: "electron/preload.ts",
        onstart(args) {
          // notify the renderer process to reload the page when preload changes
          args.reload();
        },
        vite: {
          build: {
            outDir: "dist-electron",
            rollupOptions: { external: ["electron"] },
          },
        },
      },
    ]),
    renderer(),
  ],
  build: {
    outDir: "dist",
  },
});
