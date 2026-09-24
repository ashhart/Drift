import { textContent, type OwnContext } from "./worker_context";
import { canonical, fail, type Tool } from "./worker_protocol";
import type { BeforeToolDispatch } from "./worker_boundary";
import { kvLifecycleTool, type KvRole } from "./worker_kv_lifecycle";

const sync: Tool = {
  name: "drift_sync", description: "Exchange KV memory with the paired worker.",
  parameters: { type: "object", properties: {}, additionalProperties: false },
};

export function kvSyncTool(): Tool { return structuredClone(sync); }

export function kvTools(role?: KvRole): Tool[] { return role ? [kvSyncTool(), kvLifecycleTool(role)] : [kvSyncTool()]; }

export function validateKvContext(context: OwnContext, role?: KvRole): void {
  const ordered = (tools: Tool[]) => tools.map(tool => ({ name: tool.name, description: tool.description,
    parameters: JSON.parse(JSON.stringify(tool.parameters)) })).sort((a, b) => a.name.localeCompare(b.name));
  if (!Array.isArray(context.tools) || canonical(ordered(context.tools)) !== canonical(ordered(kvTools(role)))) fail("CAPABILITY");
  if (context.messages[0]?.role !== "user") fail("CAPABILITY");
  for (const message of context.messages.slice(1)) {
    if (message.role === "assistant") continue;
    if (message.role !== "toolResult" || message.isError === true || textContent(message.content) !== "ready") fail("CAPABILITY");
  }
}

export function kvBoundary(exchange: BeforeToolDispatch): BeforeToolDispatch {
  const run: BeforeToolDispatch = async boundary => {
    if (boundary.toolNames.length > 1 || boundary.toolNames.length !== boundary.toolCalls.length) fail("CAPABILITY");
    for (const call of boundary.toolCalls) {
      if (call.name !== sync.name || canonical(call.arguments) !== "{}") fail("CAPABILITY");
    }
    if (boundary.toolNames.some(name => name !== sync.name)) fail("CAPABILITY");
    await exchange(boundary);
  };
  run.abort = () => exchange.abort?.();
  return run;
}
