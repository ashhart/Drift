import { roomHubFailure } from "./room_hub.mjs";
const record = value => value && typeof value === "object" && !Array.isArray(value);
const keys = (value, allowed) => record(value) && Object.keys(value).every(key => allowed.includes(key));
const itemKeys = ["agent", "name", "task", "description"];
export const roomTools = role => role === "parent" ? ["task", "hub", "todo"] : ["hub", "yield"];
export function roomRole(scope, context) { return context.cwd === scope.root ? context.model === scope.parent ? "parent" : context.model === scope.child ? "child" : null : null; }
export function roomFailure(scope, context, name, input, spawned) {
  const role = roomRole(scope, context);
  if (!role) return "ROOM_SCOPE";
  if (!roomTools(role).includes(name)) return "TOOL_SCOPE";
  if (!record(input) || Buffer.byteLength(JSON.stringify(input)) > 8192) return "PAYLOAD_LIMIT";
  if (name === "task") {
    if (spawned) return "TASK_ALREADY_DISPATCHED";
    if (input.context !== undefined && typeof input.context !== "string") return "TASK_CONTEXT";
    const batch = Object.hasOwn(input, "tasks");
    if (!keys(input, batch ? ["tasks", "context", "i", "agent", "name"] : [...itemKeys, "context", "i"])) return "TASK_KEYS";
    if (batch && ((input.agent !== undefined && input.agent !== "duo-peer") || (input.name !== undefined && input.name !== "DuoPeer"))) return "TASK_IDENTITY";
    if (batch && (!Array.isArray(input.tasks) || input.tasks.length !== 1)) return "TASK_SHAPE";
    const task = batch ? input.tasks[0] : input;
    if (batch && !keys(task, itemKeys)) return "TASK_ITEM_KEYS";
    if (task.agent !== "duo-peer" || task.name !== "DuoPeer") return "TASK_IDENTITY";
    return typeof task.task === "string" && task.task.length > 0 ? null : "TASK_BODY";
  }
  if (name !== "hub") return null;
  return roomHubFailure(role, input);
}
export const roomDecision = (...args) => roomFailure(...args) === null;
