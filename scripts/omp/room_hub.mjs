const keys = (input, allowed) => Object.keys(input).every(key => allowed.includes(key));
const waitBound = value => value === undefined || Number.isInteger(value) && value >= 0 && value <= 2000;
export function roomHubFailure(role, input) {
  const peer = role === "parent" ? "DuoPeer" : "Main";
  if (input.op === "list") return keys(input, ["op", "i", "limit", "status"]) && (input.limit === undefined || Number.isInteger(input.limit) && input.limit >= 1 && input.limit <= 2) && (input.status === undefined || ["running", "idle"].includes(input.status)) ? null : "HUB_LIST";
  if (input.op === "send") return keys(input, ["op", "to", "message", "await", "i"]) && input.to === peer && typeof input.message === "string" && Buffer.byteLength(input.message) <= 4096 && (input.await === undefined || typeof input.await === "boolean") ? null : "HUB_SEND";
  if (input.op === "wait") return keys(input, ["op", "ids", "from", "timeoutMs", "i"]) && (input.from === undefined || input.from === peer) && (input.ids === undefined || Array.isArray(input.ids) && input.ids.length === 1 && input.ids[0] === peer) && waitBound(input.timeoutMs) ? null : "HUB_TARGET";
  if (role === "parent" && input.op === "cancel") return keys(input, ["op", "ids", "i"]) && Array.isArray(input.ids) && input.ids.length === 1 && input.ids[0] === peer ? null : "HUB_TARGET";
  return "HUB_OPERATION";
}
export function boundedRoomHub(role, input) {
  if (input.op === "list") return { ...input, limit: input.limit ?? 2 };
  if (input.op === "wait") return { ...input, from: role === "parent" ? "DuoPeer" : "Main", timeoutMs: input.timeoutMs || 2000 };
  return input;
}
