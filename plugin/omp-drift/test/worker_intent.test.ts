import { expect, test } from "bun:test";
import { WorkerContext } from "../src/worker_context";
import type { WorkerClient } from "../src/worker_client";
async function ingestChanged(change: (part: any) => void) {
  const state = new WorkerContext();
  const client = { open: async () => {}, ownPrompt: async () => {}, toolResult: async () => {} } as unknown as WorkerClient;
  const user = { role: "user", content: "fixture" };
  await state.ingest(client, { messages: [user] });
  const answer = { role: "assistant", content: [{ type: "text", text: "prefix" }, { type: "toolCall", id: "c", name: "echo", arguments: { i: "fixture intent", value: 7 } }] };
  state.remember(answer); change(answer.content[1]);
  return state.ingest(client, { messages: [user, answer, { role: "toolResult", toolCallId: "c", content: "ok" }] });
}
test("verified OMP intent metadata does not alter generated assistant semantics", async () => {
  await expect(ingestChanged(part => { part.intent = "fixture intent"; })).resolves.toBeUndefined();
});
test("the original intent argument and every tool semantic field remain pinned", async () => {
  for (const change of [(part: any) => { part.arguments.i = "changed"; }, (part: any) => { part.arguments.value = 8; }, (part: any) => { part.id = "other"; }, (part: any) => { part.name = "other"; }]) {
    await expect(ingestChanged(change)).rejects.toThrow("CONTEXT");
  }
});
test("unknown content mutations are rejected rather than broadly discarded", async () => {
  await expect(ingestChanged(part => { part.unrecognized = "changed"; })).rejects.toThrow();
});
