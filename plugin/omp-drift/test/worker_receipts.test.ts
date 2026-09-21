import { expect, test } from "bun:test";
import { WorkerContext } from "../src/worker_context";
import type { WorkerClient } from "../src/worker_client";
const user = { role: "user", content: "goal" };
const generated = { role: "assistant", content: [{ type: "toolCall", id: "call", name: "task", arguments: { context: "original", i: "intent" } }] };
const result = { role: "toolResult", toolCallId: "call", content: "done" };
function fixture() {
  const calls: Array<[string, unknown]> = [];
  const client = { open: async () => {}, ownPrompt: async () => {}, ownControl: async (value: unknown) => calls.push(["control", value]), ownControlAppend: async (value: unknown) => calls.push(["control", value]), toolResult: async (...value: unknown[]) => calls.push(["result", value]) } as unknown as WorkerClient;
  return { calls, client, context: new WorkerContext(true, true) };
}
test("observed controller arguments are accounted before the tool result without rewriting generated history", async () => {
  const { context, client, calls } = fixture();
  await context.ingest(client, { messages: [user] }); context.remember(generated);
  const args = { context: "original plus actual stock Duo todo" };
  context.recordExecution("call", "task", args);
  await context.ingest(client, { messages: [user, { ...generated, content: [{ ...generated.content[0], arguments: args }] }, result] });
  expect(calls.map(call => call[0])).toEqual(["control", "result"]);
  expect(JSON.stringify(calls[0][1])).toContain("Controller-observed tool execution amendment");
  expect(generated.content[0].arguments).toEqual({ context: "original", i: "intent" });
});
test("unobserved semantic rewrites and forged execution receipts fail closed", async () => {
  const { context, client } = fixture();
  await context.ingest(client, { messages: [user] }); context.remember(generated);
  expect(() => context.recordExecution("unknown", "task", {})).toThrow("CONTEXT");
  expect(() => context.recordExecution("call", "wrong", {})).toThrow("CONTEXT");
  await expect(context.ingest(client, { messages: [user, { ...generated, content: [{ ...generated.content[0], arguments: {} }] }, result] })).rejects.toThrow("CONTEXT");
});
test("execution receipt permits only the exact observed arguments", async () => {
  const { context, client } = fixture();
  await context.ingest(client, { messages: [user] }); context.remember(generated);
  context.recordExecution("call", "task", { context: "observed" });
  expect(() => context.recordExecution("call", "task", { context: "different" })).toThrow("CONTEXT");
  await expect(context.ingest(client, { messages: [user, { ...generated, content: [{ ...generated.content[0], arguments: { context: "forged" } }] }, result] })).rejects.toThrow("CONTEXT");
});
test("mixed multicall history keeps unchanged original arguments while reconciling the amended sibling", async () => {
  const { context, client, calls } = fixture();
  const todo = { type: "toolCall", id: "todo", name: "todo", arguments: { op: "view", i: "intent" } };
  const message = { role: "assistant", content: [todo, generated.content[0]] };
  await context.ingest(client, { messages: [user] }); context.remember(message);
  context.recordExecution("todo", "todo", { op: "view" });
  const amended = { context: "observed task update" };
  context.recordExecution("call", "task", amended);
  await context.ingest(client, { messages: [user, { ...message, content: [todo, { ...generated.content[0], arguments: amended }] }, { ...result, toolCallId: "todo" }, result] });
  expect(calls.map(call => call[0])).toEqual(["control", "result", "result"]);
  expect(JSON.stringify(calls[0][1])).toContain('observed task update');
  expect(todo.arguments.i).toBe("intent");
});
test("multicall reconciliation still rejects an unobserved sibling mutation", async () => {
  const { context, client } = fixture();
  const todo = { type: "toolCall", id: "todo", name: "todo", arguments: { op: "view", i: "intent" } };
  const message = { role: "assistant", content: [todo, generated.content[0]] };
  await context.ingest(client, { messages: [user] }); context.remember(message);
  context.recordExecution("todo", "todo", { op: "view" });
  context.recordExecution("call", "task", { context: "observed" });
  await expect(context.ingest(client, { messages: [user, { ...message, content: [{ ...todo, arguments: { op: "delete" } }, { ...generated.content[0], arguments: { context: "observed" } }] }, result] })).rejects.toThrow("CONTEXT");
});
test("unchanged multicall history remains original while both execution receipts are accounted", async () => {
  const { context, client, calls } = fixture();
  const todo = { type: "toolCall", id: "todo", name: "todo", arguments: { op: "view", i: "intent" } };
  const message = { role: "assistant", content: [todo, generated.content[0]] };
  await context.ingest(client, { messages: [user] }); context.remember(message);
  context.recordExecution("todo", "todo", { op: "view" });
  context.recordExecution("call", "task", { context: "observed" });
  await context.ingest(client, { messages: [user, message, { ...result, toolCallId: "todo" }, result] });
  expect(calls.map(call => call[0])).toEqual(["control", "result", "result"]);
  expect((calls[0][1] as string[]).length).toBe(2);
  expect(message.content[1].arguments).toEqual({ context: "original", i: "intent" });
});
