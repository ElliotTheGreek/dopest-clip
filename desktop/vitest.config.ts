import { defineConfig } from "vitest/config";

// Renderer logic tests only — no Electron, no Python sidecar required.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
