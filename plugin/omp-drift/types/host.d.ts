/**
 * Typing shim for the host import. At runtime the OMP loader remaps the bare
 * `@oh-my-pi/pi-coding-agent` specifier onto the running host, so the plugin
 * has no dependency to install; this shim only serves `bun run typecheck`.
 *
 * omp-drift currently reaches the host only through the extension API
 * object handed to its entry point (see src/types.ts), so nothing here is
 * imported yet. The declaration keeps the same layout as omp-local-duo so a
 * future host import typechecks the same way.
 */
declare module "@oh-my-pi/pi-coding-agent" {}
