import { expect, test } from "bun:test";
import { lifecycleBoundary, kvTask, kvYield } from "../src/worker_kv_lifecycle";
import { installKvTools } from "../src/worker_kv_tools";
import { kvTools, validateKvContext } from "../src/worker_kv_policy";

test("KV context validates wire schemas without serializing host tool wrappers", () => {
  const tools = kvTools("parent").map(tool => ({ ...tool, runner: { execute() {} }, renderCall: undefined }));
  const context = { messages: [{ role: "user", content: "own task" }], tools };
  expect(() => validateKvContext(context, "parent")).not.toThrow();
  const withIntent = tools.map(tool => ({ ...tool, parameters: {
    type: "object", properties: { i: { type: "string" } }, required: ["i"], additionalProperties: false,
  } }));
  expect(() => validateKvContext({ ...context, tools: withIntent }, "parent")).toThrow("CAPABILITY");
});

const boundary = (name?: string, args = {}) => ({ identity: {} as any, signal: new AbortController().signal,
  toolNames: name ? [name] : [], toolCalls: name ? [{ name, arguments: args }] : [] });

test("subagent lifecycle rejects payloads, repeated admission and child terminal text", async () => {
  const admitted: string[] = [];
  const parent = lifecycleBoundary(async value => { admitted.push(...value.toolNames); }, "parent");
  await expect(parent(boundary("drift_sync"))).rejects.toThrow("CAPABILITY");
  await expect(parent(boundary("task", { task: "peer secret" }))).rejects.toThrow("CAPABILITY");
  await parent(boundary("task"));
  await expect(parent(boundary("task"))).rejects.toThrow("CAPABILITY");
  await parent(boundary("drift_sync")); await parent(boundary());
  expect(admitted).toEqual(["task", "drift_sync"]);
  await expect(parent(boundary("drift_sync"))).rejects.toThrow("CAPABILITY");
  const child = lifecycleBoundary(async () => {}, "child");
  await expect(child(boundary())).rejects.toThrow("CAPABILITY");
  await expect(child(boundary("task"))).rejects.toThrow("CAPABILITY");
  await child(boundary("drift_sync")); await child(boundary("yield"));
});

test("native task delegation receives only fixed control and discards its result", async () => {
  const tools = new Map<string, any>(), sent: unknown[] = [];
  installKvTools({ registerTool: tool => { tools.set(tool.name, tool); }, on() {}, setActiveTools() {}, getActiveTools: () => [] },
    { workers: ["parent", "child"].map(role => ({ identity: { model_id: role }, subagent_role: role })) } as any);
  const context = (role: string) => ({ model: { provider: "drift-experimental", id: role }, sessionManager: { getSessionId: () => role },
    invokeTool: async (args: unknown) => { sent.push(args); return { content: [{ type: "text", text: "PRIVATE_PEER_SENTINEL" }] }; } });
  const execute = (name: string, role: string, args = {}) => tools.get(name).execute("id", args, new AbortController().signal, undefined, context(role));
  await expect(execute("task", "child")).rejects.toThrow("CAPABILITY");
  await expect(execute("task", "parent", { prompt: "peer secret" })).rejects.toThrow("CAPABILITY");
  expect(await execute("task", "parent")).toEqual({ content: [{ type: "text", text: "ready" }], details: {} });
  await expect(execute("task", "parent")).rejects.toThrow("CAPABILITY");
  expect(await execute("yield", "child")).toEqual({ content: [{ type: "text", text: "ready" }], details: {} });
  expect(sent).toEqual([kvTask, kvYield]);
});
