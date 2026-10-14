import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: { entry: "src/index.tsx", formats: ["es"], fileName: () => "filesystem.js", cssFileName: "filesystem" },
    outDir: "../../src/juena_core/ui/filesystem_dist",
    emptyOutDir: true,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
});
