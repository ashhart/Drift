import { expect, test } from "bun:test";
import { WorkerContext } from "../src/worker_context";
import type { WorkerClient } from "../src/worker_client";
const user = { role: "user", content: "public task" };
const assistant = { role: "assistant", content: [{ type: "text", text: "public result" }] };
function fixture() {
  const calls: Array<[string, unknown]> = [];
  const client = { open: async (value: unknown) => calls.push(["open", value]), ownPrompt: async () => {}, ownControl: async (value: unknown) => calls.push(["snapshot", value]), ownControlAppend: async (value: unknown) => calls.push(["append", value]), toolResult: async (...value: unknown[]) => calls.push(["result", value]) } as unknown as WorkerClient;
  return { context: new WorkerContext(true, true), client, calls };
}
test("unchanged base appends only fresh developer controls once", async () => {
  const { context, client, calls } = fixture();
  const initial = [user, { role: "developer", content: "initial control" }];
  await context.ingest(client, { systemPrompt: "stable base", messages: initial }); context.remember(assistant);
  const next = [...initial, assistant, { role: "developer", content: "new control" }];
  await context.ingest(client, { systemPrompt: "stable base", messages: next }); context.remember(assistant);
  await context.ingest(client, { systemPrompt: "stable base", messages: [...next, assistant, { role: "user", content: "continue" }] });
  expect(calls).toEqual([["open", ["stable base", "initial control"]], ["append", ["new control"]]]);
});
test("a real base change keeps the full snapshot fallback then establishes a new append baseline", async () => {
  const { context, client, calls } = fixture();
  await context.ingest(client, { systemPrompt: "old", messages: [user] }); context.remember(assistant);
  const next = [user, assistant, { role: "developer", content: "peer text" }];
  await context.ingest(client, { systemPrompt: "changed", messages: next }); context.remember(assistant);
  await context.ingest(client, { systemPrompt: "changed", messages: [...next, assistant, { role: "developer", content: "later" }] });
  expect(calls).toEqual([["open", ["old"]], ["snapshot", ["changed", "peer text"]], ["append", ["later"]]]);
});
test("append sends fresh developer text then exact receipt amendment before tool results", async () => {
  const { context, client, calls } = fixture();
  await context.ingest(client, { systemPrompt: "base", messages: [user] });
  const generated = { role: "assistant", content: [{ type: "toolCall", id: "call", name: "task", arguments: { i: "intent", task: "work" } }] };
  context.remember(generated); context.recordExecution("call", "task", { task: "work with controller suffix" });
  await context.ingest(client, { systemPrompt: "base", messages: [user, generated, { role: "developer", content: "new advisory" }, { role: "toolResult", toolCallId: "call", content: "done" }] });
  expect(calls.map(call => call[0])).toEqual(["open", "append", "result"]);
  expect((calls[1][1] as string[])[0]).toBe("new advisory");
  expect((calls[1][1] as string[])[1]).toContain('"call_id":"call"');
  expect(generated.content[0].arguments).toEqual({ i: "intent", task: "work" });
});
test("append cannot hide changed historical controls or tools", async () => {
  for (const tamper of ["history", "tools"]) {
    const { context, client, calls } = fixture();
    const initial = [user, { role: "developer", content: "original" }];
    await context.ingest(client, { systemPrompt: "base", messages: initial }); context.remember(assistant);
    const messages = [user, { role: "developer", content: tamper === "history" ? "tampered" : "original" }, assistant, { role: "developer", content: "new" }];
    await expect(context.ingest(client, { systemPrompt: "base", messages, tools: tamper === "tools" ? [{ name: "other", description: "x", parameters: {} }] : [] })).rejects.toThrow(tamper === "history" ? "CONTEXT" : "CAPABILITY");
    expect(calls.length).toBe(1);
  }
});
