const known = new Set(["tasks", "context", "i", "agent", "name", "task", "description", "model", "cwd", "tools", "isolated", "effort", "outputSchema", "schemaMode", "op", "to", "message", "await", "ids", "timeoutMs", "from", "status", "limit"]);
export function roomArgumentShape(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) return { shape: "other", keys: [], unknown_keys: 0 };
  const names = Object.keys(input);
  return { shape: Object.hasOwn(input, "tasks") ? "batch" : Object.hasOwn(input, "task") ? "flat" : "other", keys: names.filter(name => known.has(name)).sort(), unknown_keys: names.filter(name => !known.has(name)).length };
}
export function roomTerminalCode(message) {
  const match = /\[(context|stream|close|unknown):((?:REMOTE_)?(?:PROTOCOL|LIMIT|CAPABILITY|WORKER)|CONTEXT|SESSION|SESSION_LIMIT|TIMEOUT|CANCELLED|CLOSED|TRANSPORT|UNKNOWN)\]$/.exec(message ?? "");
  return match ? { phase: match[1], code: match[2] } : { phase: "unknown", code: "UNKNOWN" };
}
