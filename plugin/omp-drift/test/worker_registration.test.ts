import { test, expect } from "bun:test";
import { validateWorkerConfig } from "../src/worker_registration";
test("owner configuration requires explicit experimental qualification and command pins", () => {
  expect(() => validateWorkerConfig({ version: 1, workers: [] })).toThrow();
});

test("owner config refuses live sessions labelled fixture and missing memory mode", () => {
  const entry = { identity: { session: "s", worker: "w", model_id: "m", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) }, limits: { max_input_bytes: 8192, max_output_tokens: 128, max_session_tokens: 512, max_turns: 2, deadline_ms: 1000 }, context_window: 8192, expected: { backend: "live", nativeStates: ["fixture"] }, command: {}, artifacts: [] };
  expect(() => validateWorkerConfig({ version: 1, experimental: true, workers: [entry] })).toThrow("CONFIG");
  expect(() => validateWorkerConfig({ version: 1, experimental: true, workers: [{ ...entry, memory_mode: "no-link" }] })).toThrow("CONFIG");
});

test("owner pins reject tampered backend artifacts before dispatch", async () => {
  const { mkdtempSync, writeFileSync, readFileSync, rmSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { createHash } = await import("node:crypto");
  const root = mkdtempSync(tmpdir() + "/drift-owner-"); const path = root + "/backend.py";
  const hash = (value: Buffer) => createHash("sha256").update(value).digest("hex");
  writeFileSync(path, "fixture");
  const config = { version: 1, experimental: true, workers: [{ identity: { session: "s", worker: "w", model_id: "m", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) }, limits: { max_input_bytes: 8192, max_output_tokens: 128, max_session_tokens: 512, max_turns: 2, deadline_ms: 1000 }, context_window: 8192, memory_mode: "no-link", expected: { backend: "live", nativeStates: ["reconstructed"] }, command: { executable: "/bin/echo", sha256: hash(readFileSync("/bin/echo")), args: [], cwd: root, env: {} }, artifacts: [{ path, sha256: hash(readFileSync(path)) }] }] };
  try { expect(validateWorkerConfig(config).workers.length).toBe(1); writeFileSync(path, "tampered"); expect(() => validateWorkerConfig(config)).toThrow("PIN"); } finally { rmSync(root, { recursive: true, force: true }); }
});
