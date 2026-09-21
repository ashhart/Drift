import { writeFileSync } from "node:fs";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai";

const provider = "drift-contract-fixture";
const apiName = "drift-contract-fixture-api";
const toolName = "drift_contract_echo";
const facts = { registered: false, stream_calls: 0, tool_calls: 0, tool_results_seen: 0, tool_events: 0, other_tools: 0 };
const save = () => writeFileSync(process.env.DRIFT_CONTRACT_REPORT, JSON.stringify(facts));

export default function contractExtension(api) {
  api.registerTool({
    name: toolName,
    label: "Synthetic contract echo",
    description: "Return a fixed fixture value without I/O.",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    async execute() {
      facts.tool_calls += 1;
      save();
      return { content: [{ type: "text", text: "fixture-ok" }], details: {} };
    },
  });
  api.on("tool_call", event => {
    if (event.toolName === toolName) facts.tool_events += 1;
    else facts.other_tools += 1;
    save();
  });
  api.on("session_start", async () => {
    await api.setActiveTools([toolName]);
  });
  api.registerProvider(provider, {
    api: apiName,
    baseUrl: "https://example.invalid/never-called",
    apiKey: "synthetic-not-a-credential",
    models: [{
      id: "fixture", name: "Deterministic fixture", reasoning: false, input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 128,
    }],
    streamSimple(model, context) {
      facts.stream_calls += 1;
      facts.tool_results_seen = context.messages.filter(message => message.role === "toolResult" && message.toolName === toolName).length;
      save();
      if (facts.stream_calls > 2) throw new Error("Fixture exceeded its two-call budget");
      const stream = new AssistantMessageEventStream();
      const first = facts.stream_calls === 1;
      const part = first ? { type: "toolCall", id: "fixture-call-1", name: toolName, arguments: {} } : { type: "text", text: "fixture-complete" };
      const message = {
        role: "assistant", content: [part], api: apiName, provider, model: model.id,
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: first ? "toolUse" : "stop", timestamp: Date.now(),
      };
      stream.push({ type: "start", partial: message });
      if (first) {
        stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
        stream.push({ type: "toolcall_end", contentIndex: 0, toolCall: part, partial: message });
      } else {
        stream.push({ type: "text_start", contentIndex: 0, partial: message });
        stream.push({ type: "text_delta", contentIndex: 0, delta: part.text, partial: message });
        stream.push({ type: "text_end", contentIndex: 0, content: part.text, partial: message });
      }
      stream.push({ type: "done", reason: message.stopReason, message });
      return stream;
    },
  });
  facts.registered = true;
  save();
}
