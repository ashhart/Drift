import { writeFileSync } from "node:fs";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai";
const key = Symbol.for("drift.room.attack"), state = globalThis[key] ??= { turns: {}, tools: {}, blocked_results: {}, executed: {}, sessions: [] };
const save = () => writeFileSync(process.env.DRIFT_ROOM_ATTACK_REPORT, JSON.stringify(state));
export default function attackFixture(api) {
  api.on("tool_result", event => { if (["read", "bash", "edit", "task"].includes(event.toolName)) { const target = event.isError ? state.blocked_results : state.executed; target[event.toolName] = (target[event.toolName] ?? 0) + 1; save(); } });
  api.registerProvider("drift-experimental", {
    api: "room-attack-fixture", apiKey: "fixture", baseUrl: "https://example.invalid/unused",
    models: ["glm-fixture", "qwen-fixture"].map(id => ({ id, name: id, reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 128 })),
    streamSimple(model, context, options) {
      if (!state.sessions.includes(options.sessionId)) state.sessions.push(options.sessionId);
      state.tools[model.id] = context.tools.map(tool => tool.name).sort();
      const turn = state.turns[model.id] = (state.turns[model.id] ?? 0) + 1;
      if (turn > 12) throw new Error("ROOM_ATTACK_LIMIT");
      let calls = [];
      if (model.id === "glm-fixture" && turn === 1) calls = [["task", { context: "Synthetic guard qualification", tasks: [{ agent: "duo-peer", name: "DuoPeer", task: "Synthetic negative tool fixture." }] }]];
      else if (model.id === "glm-fixture" && turn === 2) calls = [["task", { tasks: [{ agent: "other-agent", name: "OtherPeer", task: "Do not create another room member" }] }]];
      else if (model.id === "glm-fixture" && turn === 3) calls = [["hub", { op: "wait", ids: ["DuoPeer"], timeoutMs: 2000 }]];
      else if (model.id === "qwen-fixture" && turn === 1) calls = [["read", { path: "sentinel.txt" }], ["bash", { command: "printf escaped > escaped.txt" }], ["edit", { path: "sentinel.txt", edits: [{ old_text: "sentinel", new_text: "escaped" }] }], ["task", { tasks: [{ agent: "duo-peer", name: "ExtraPeer", task: "Forbidden extra peer" }] }]];
      const content = calls.length ? calls.map(([name, args], index) => ({ type: "toolCall", id: `${model.id}-${turn}-${index}`, name, arguments: args })) : [{ type: "text", text: "Fixture complete" }];
      const message = { role: "assistant", content, api: "room-attack-fixture", provider: "drift-experimental", model: model.id, usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: calls.length ? "toolUse" : "stop", timestamp: Date.now() };
      const stream = new AssistantMessageEventStream(); stream.push({ type: "start", partial: message });
      content.forEach((part, index) => {
        if (part.type === "toolCall") { stream.push({ type: "toolcall_start", contentIndex: index, partial: message }); stream.push({ type: "toolcall_end", contentIndex: index, toolCall: part, partial: message }); }
        else { stream.push({ type: "text_start", contentIndex: index, partial: message }); stream.push({ type: "text_delta", contentIndex: index, delta: part.text, partial: message }); stream.push({ type: "text_end", contentIndex: index, content: part.text, partial: message }); }
      });
      stream.push({ type: "done", reason: message.stopReason, message }); save(); return stream;
    },
  });
}
