import { writeFileSync } from "node:fs";
const facts = { tool_calls: 0, tool_events: 0, other_tools: 0 };
const save = () => writeFileSync(process.env.DRIFT_CONTRACT_REPORT, JSON.stringify(facts));
export default function fixture(api) {
  api.registerTool({ name: "drift_contract_echo", label: "Fixture", description: "Return fixture value", parameters: { type: "object", properties: {} },
    async execute() { facts.tool_calls++; save(); return { content: [{ type: "text", text: "fixture-ok" }], details: {} }; } });
  api.on("session_start", async () => api.setActiveTools(["drift_contract_echo"]));
  api.on("tool_call", event => { if (event.toolName === "drift_contract_echo") facts.tool_events++; else facts.other_tools++; save(); });
  save();
}
