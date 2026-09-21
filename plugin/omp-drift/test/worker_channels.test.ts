import { expect, test } from "bun:test";
import { validateChannels, textDuoAllowed } from "../src/worker_channels";
import { WorkerRoutes } from "../src/worker_routes";
import { WorkerContext } from "../src/worker_context";
import type { WorkerClient } from "../src/worker_client";
const entry = { experimental_multi_turn: true, memory_mode: "linked", communication_mode: "text_and_artifacts", session_binding: "fixed", identity: { session: "owner-fixed", worker: "glm", model_id: "glm", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) } } as any;
test("fixed session requires the owner's explicit linked communication opt-in", () => {
  expect(() => validateChannels(entry)).not.toThrow();
  for (const patch of [{ experimental_multi_turn: false }, { memory_mode: "no-link" }, { communication_mode: undefined }, { communication_mode: "peer-owned" }, { session_binding: "model-chosen" }]) expect(() => validateChannels({ ...entry, ...patch })).toThrow("CONFIG");
  const routes = new WorkerRoutes();
  expect(routes.bind(entry, "omp-parent").identity.session).toBe("owner-fixed");
  expect(routes.bind(entry, "omp-parent").identity.session).toBe("owner-fixed");
  expect(() => routes.bind(entry, "second-parent")).toThrow("SESSION_LIMIT");
  expect(new WorkerRoutes().bind({ ...entry, communication_mode: undefined, session_binding: undefined }, "omp-parent").identity.session).not.toBe("owner-fixed");
  expect(new WorkerRoutes().bind({ ...entry, experimental_multi_turn: false, communication_mode: undefined, session_binding: undefined }, undefined).identity.session).toBe("owner-fixed");
});
test("declared text uses counted own input in both arms and strict linked stays blocked", async () => {
  for (const memory_mode of ["no-link", "linked"] as const) {
    const operations: unknown[] = [];
    const client = { open: async (system: string[]) => operations.push(["open", system]), ownPrompt: async (text: string) => operations.push(["prompt", text]), ownControl: async (system: string[]) => operations.push(["own_control", system]), ownControlAppend: async (system: string[]) => operations.push(["own_control_append", system]) } as unknown as WorkerClient;
    const allowed = textDuoAllowed({ ...entry, memory_mode, session_binding: undefined });
    const state = new WorkerContext(true, allowed), user = { role: "user", content: "own task" };
    await state.ingest(client, { messages: [user] });
    const answer = { role: "assistant", content: [{ type: "text", text: "ready" }] }; state.remember(answer);
    await state.ingest(client, { messages: [user, answer, { role: "developer", content: "declared peer Hub text" }] });
    expect(operations).toEqual([["open", []], ["prompt", "own task"], ["own_control_append", ["declared peer Hub text"]]]);
  }
  const strict = { ...entry, communication_mode: undefined, session_binding: undefined };
  expect(textDuoAllowed(strict)).toBe(false);
  await expect(new WorkerContext(true, textDuoAllowed(strict)).ingest({} as WorkerClient, { messages: [{ role: "developer", content: "peer text" }] })).rejects.toThrow("CAPABILITY");
});

test("communication authorization is covered by the owner hash and invalid modes fail registration", async () => {
  const { mkdtempSync, writeFileSync, readFileSync, rmSync } = await import("node:fs");
  const { tmpdir } = await import("node:os"); const { createHash } = await import("node:crypto");
  const { loadWorkerConfig, validateWorkerConfig } = await import("../src/worker_registration");
  const root = mkdtempSync(tmpdir()+"/declared-channels-"), path = root+"/owner.json", artifact = root+"/artifact";
  const hash = (value: Buffer) => createHash("sha256").update(value).digest("hex"); writeFileSync(artifact,"fixture");
  const config = { version: 1, experimental: true, workers: [{ ...entry, limits: { max_input_bytes: 8192, max_output_tokens: 128, max_session_tokens: 512, max_turns: 2, deadline_ms: 1000 }, context_window: 8192, expected: { backend: "fixture", nativeStates: ["fixture"] }, command: { executable: "/bin/echo", sha256: hash(readFileSync("/bin/echo")), args: [], cwd: root, env: {} }, artifacts: [{ path: artifact, sha256: hash(readFileSync(artifact)) }] }] };
  try {
    writeFileSync(path,JSON.stringify(config)); const pin = hash(readFileSync(path)); expect(loadWorkerConfig(path,pin).workers[0].session_binding).toBe("fixed");
    for (const patch of [{ communication_mode: "implicit" }, { session_binding: "derived" }, { experimental_multi_turn: false }]) expect(() => validateWorkerConfig({ ...config, workers: [{ ...config.workers[0], ...patch }] })).toThrow("CONFIG");
    config.workers[0].communication_mode = undefined; writeFileSync(path,JSON.stringify(config)); expect(() => loadWorkerConfig(path,pin)).toThrow("PIN");
  } finally { rmSync(root,{recursive:true,force:true}); }
});
