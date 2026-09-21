import { fail, record } from "./worker_protocol";
export function assistantSemantics(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) return fail("CONTEXT");
  return value.map(raw => {
    if (!record(raw)) return fail("CONTEXT");
    const part = Object.fromEntries(Object.entries(raw).filter(([, value]) => value !== undefined));
    if (part.type === "text") {
      if (Object.keys(part).sort().join() !== "text,type" || typeof part.text !== "string") return fail("CONTEXT");
      return { type: "text", text: part.text };
    }
    if (part.type !== "toolCall" || typeof part.id !== "string" || typeof part.name !== "string" || !record(part.arguments)) return fail("CONTEXT");
    if (Object.keys(part).some(key => !["type", "id", "name", "arguments", "intent"].includes(key))) return fail("CONTEXT");
    if (part.intent !== undefined && typeof part.intent !== "string") return fail("CONTEXT");
    return { type: "toolCall", id: part.id, name: part.name, arguments: part.arguments };
  });
}
