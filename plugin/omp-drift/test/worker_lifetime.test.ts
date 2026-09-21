import { expect, test } from "bun:test";
import { WorkerContext } from "../src/worker_context";
import type { WorkerClient } from "../src/worker_client";
test("opt-in own system updates travel once on the private channel", async () => {
  const calls: string[] = [];
  const client = { open: async () => calls.push("open"), ownPrompt: async () => calls.push("prompt"), ownControl: async () => calls.push("control"), ownControlAppend: async () => calls.push("append") } as unknown as WorkerClient;
  const state = new WorkerContext(true);
  const first = { role: "user", content: "first" };
  await state.ingest(client, { systemPrompt: "old", messages: [first] });
  const answer = { role: "assistant", content: [{ type: "text", text: "answer" }] }; state.remember(answer);
  await state.ingest(client, { systemPrompt: "new", messages: [first, answer, { role: "user", content: "next" }] });
  expect(calls).toEqual(["open", "prompt", "control", "prompt"]);
});

test("different OMP harnesses cannot share a worker session", async () => {
  const { WorkerRoutes } = await import("../src/worker_routes");
  const routes = new WorkerRoutes();
  const entry = { experimental_multi_turn: true, identity: { session: "owner", worker: "glm", model_id: "glm", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) } } as any;
  expect(() => routes.bind(entry, undefined)).toThrow("SESSION");
  const first = routes.bind(entry, "parent-omp"); expect(routes.bind(entry, "parent-omp").key).toBe(first.key);
  expect(() => routes.bind(entry, "another-omp")).toThrow("SESSION_LIMIT");
  expect(routes.bind({ ...entry, identity: { ...entry.identity, worker: "qwen" } }, "child-omp").key).not.toBe(first.key);
});

test("Duo developer text uses the declared private input channel only when opted in", async () => {
  const opened: string[][] = []; const prompts: string[] = [];
  const client = { open: async (system: string[]) => opened.push(system), ownPrompt: async (text: string) => prompts.push(text) } as unknown as WorkerClient;
  const context = { systemPrompt: "base", messages: [{ role: "developer", content: "declared room text" }, { role: "user", content: "goal" }] };
  await new WorkerContext(true, true).ingest(client, context);
  expect(opened).toEqual([["base", "declared room text"]]); expect(prompts).toEqual(["goal"]);
  await expect(new WorkerContext().ingest(client, context)).rejects.toThrow("CAPABILITY");
});
