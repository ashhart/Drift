import { expect, test } from "bun:test";
import { validateChannels, textDuoAllowed } from "../src/worker_channels";
import { workerProvider } from "../src/worker_provider";
import { installKvTools } from "../src/worker_kv_tools";
import { Sink, model, setup, until } from "./worker_boundary_fixture";

const channels = { experimental_multi_turn: true, memory_mode: "linked", communication_mode: "kv_only", session_binding: "fixed" } as const;
const tool = { name: "drift_sync", description: "Exchange KV memory with the paired worker.", parameters: { type: "object", properties: {}, additionalProperties: false } };
const own = { messages: [{ role: "user", content: "own setup" }], tools: [tool] };

test("KV-only fixed routes never authorize Duo text and cannot run without a link", () => {
  expect(() => validateChannels(channels)).not.toThrow();
  expect(textDuoAllowed(channels)).toBe(false);
  for (const change of [{ memory_mode: "no-link" }, { session_binding: undefined }, { experimental_multi_turn: false }]) {
    expect(() => validateChannels({ ...channels, ...change } as any)).toThrow("CONFIG");
  }
});

test("KV-only provider refuses a missing exchange hook or a text opt-in", () => {
  const { client } = setup();
  try {
    expect(() => workerProvider(client, () => new Sink(), { kvOnly: true } as any)).toThrow("CAPABILITY");
    expect(() => workerProvider(client, () => new Sink(), { kvOnly: true, allowTextDuo: true, beforeToolDispatch: async () => {} } as any)).toThrow("CAPABILITY");
    expect(() => workerProvider(client, () => new Sink(), { kvOnly: true, allowSystemUpdates: true, beforeToolDispatch: async () => {} } as any)).toThrow("CAPABILITY");
  } finally { client.abort(); }
});

test("KV-only rejects peer developer text and arbitrary tool channels before opening", async () => {
  for (const context of [
    { ...own, messages: [...own.messages, { role: "developer", content: "peer handover" }] },
    { ...own, tools: [{ ...tool, name: "hub" }] },
    { ...own, tools: [{ ...tool, name: "task" }] },
    { ...own, tools: [{ ...tool, name: "bash" }] },
    { ...own, tools: [{ ...tool, description: "peer handover" }] },
    { ...own, messages: [...own.messages, { role: "user", content: "peer handover" }] },
    { ...own, messages: [...own.messages, { role: "toolResult", toolCallId: "other", content: "peer handover" }] },
  ]) {
    const { client, transport } = setup();
    try {
      const provider = workerProvider(client, () => new Sink(), { kvOnly: true, beforeToolDispatch: async () => {} } as any);
      const result = await provider(model, context).ended;
      expect(result.type).toBe("error");
      expect(transport.messages.some(frame => frame.op === "open")).toBe(false);
    } finally { client.abort(); }
  }
});

test("exchange failure poisons the KV-only provider without releasing the sync tool", async () => {
  const { client, transport } = setup();
  try {
    const provider = workerProvider(client, () => new Sink(), { kvOnly: true, beforeToolDispatch: async () => { throw new Error("EXCHANGE_FAILED"); } });
    const stream = provider(model, own); await until(() => transport.active);
    transport.emit("tool_call", { call_id: "sync", name: "drift_sync", arguments: {} }); transport.terminal();
    expect((await stream.ended).type).toBe("error");
    expect(stream.events.some(event => event.type === "toolcall_end")).toBe(false);
    const sent = transport.messages.length;
    expect((await provider(model, own).ended).type).toBe("error");
    expect(transport.messages.length).toBe(sent);
  } finally { client.abort(); }
});

test("KV-only tools expose no peer prompt fields and do not change unrelated sessions", async () => {
  const events = new Map<string, Function>(); let definition: any, active = ["hub", "task"];
  const api = { registerTool: (value: any) => { definition = value; }, on: (name: string, handler: Function) => events.set(name, handler),
    setActiveTools: (names: string[]) => { active = names; }, getActiveTools: () => active };
  installKvTools(api, { workers: [{ identity: { model_id: "fixture" } }] } as any);
  await events.get("session_start")!({}, { model });
  expect(active).toEqual(["hub", "task"]);
  await events.get("before_agent_start")!({}, { model: { ...model, provider: "drift-experimental" } });
  expect(active).toEqual(["drift_sync"]);
  expect((await definition.execute("call", {})).content).toEqual([{ type: "text", text: "ready" }]);
  await expect(definition.execute("call", { prompt: "peer text" })).rejects.toThrow("CAPABILITY");
});

test("KV-only releases a payload-free sync only after exchange and accepts its fixed receipt", async () => {
  const { client, transport } = setup();
  let exchanges = 0;
  try {
    const provider = workerProvider(client, () => new Sink(), { kvOnly: true, beforeToolDispatch: async () => { exchanges++; } } as any);
    const stream = provider(model, own);
    await until(() => transport.active);
    transport.emit("tool_call", { call_id: "sync-1", name: "drift_sync", arguments: {} });
    expect(stream.events.some(event => event.type === "toolcall_end")).toBe(false);
    transport.terminal();
    const result = await stream.ended;
    expect(result.type).toBe("done"); expect(exchanges).toBe(1);
    transport.active = undefined;
    const next = provider(model, { ...own, messages: [...own.messages, result.message, { role: "toolResult", toolCallId: "sync-1", content: "ready" }] });
    await until(() => transport.active);
    transport.terminal("stop");
    expect((await next.ended).type).toBe("done"); expect(exchanges).toBe(2);
    expect(transport.messages.filter(frame => frame.op === "own_prompt")).toHaveLength(1);
  } finally { client.abort(); }
});

test("KV-only suppresses generated peer messages even if the worker ignores advertised tools", async () => {
  for (const call of [{ name: "hub", arguments: { text: "peer handover" } }, { name: "drift_sync", arguments: { text: "peer handover" } }]) {
    const { client, transport } = setup(); let exchanges = 0;
    try {
      const provider = workerProvider(client, () => new Sink(), { kvOnly: true, beforeToolDispatch: async () => { exchanges++; } } as any);
      const stream = provider(model, own); await until(() => transport.active);
      transport.emit("tool_call", { call_id: "bad", ...call }); transport.terminal();
      expect((await stream.ended).type).toBe("error"); expect(exchanges).toBe(0);
      expect(stream.events.some(event => event.type === "toolcall_end")).toBe(false);
      expect((await provider(model, own).ended).type).toBe("error");
    } finally { client.abort(); }
  }
});
