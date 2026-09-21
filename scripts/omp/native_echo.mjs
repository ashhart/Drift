import { writeFileSync } from "node:fs";
const facts = { tool_calls: 0, other_tools: 0, assistant_turns: 0, output_tokens: 0, input_tokens: 0, diagnostics: [] };
const save = () => writeFileSync(process.env.DRIFT_NATIVE_REPORT, JSON.stringify(facts));
export default function nativeEcho(api) {
  api.registerTool({ name: "drift_native_echo", label: "Fixed non-I/O echo", description: "Return the fixed string echo-ok without arguments or external I/O", parameters: { type: "object", properties: {}, additionalProperties: false },
    async execute() { facts.tool_calls++; save(); if (facts.tool_calls > 1) throw new Error("DRIFT_PROBE_LIMIT"); return { content: [{ type: "text", text: "echo-ok" }], details: {} }; } });
  api.on("session_start", async () => api.setActiveTools(["drift_native_echo"]));
  api.on("tool_call", event => { if (event.toolName !== "drift_native_echo") facts.other_tools++; save(); });
  api.on("turn_end", event => { if (event.message?.role === "assistant") { facts.assistant_turns++; facts.output_tokens += event.message.usage?.output ?? 0; facts.input_tokens += event.message.usage?.input ?? 0;
    const match = /\[(context|stream|close|unknown):((?:REMOTE_)?(?:PROTOCOL|LIMIT|CAPABILITY|WORKER)|CONTEXT|SESSION|SESSION_LIMIT|TIMEOUT|CANCELLED|CLOSED|TRANSPORT|UNKNOWN)\]$/.exec(event.message.errorMessage ?? "");
    if (match) facts.diagnostics.push({ phase: match[1], code: match[2] }); } save(); });
  save();
}
