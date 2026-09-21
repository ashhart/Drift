import { boundedRoomHub } from "./room_hub.mjs";
import { realpathSync } from "node:fs";
import { roomFailure, roomRole, roomTools } from "./room_policy.mjs";
export const developmentTools = role => [...roomTools(role), "drift_task_read", "drift_task_write", "drift_task_verify"];
export function developmentFailure(scope, context, name, input, spawned) {
  if (!roomRole(scope, context)) return "ROOM_SCOPE";
  if (!input || typeof input !== "object" || Array.isArray(input)) return "PAYLOAD_SHAPE";
  const keys = Object.keys(input).filter(key => key !== "i");
  if (name === "drift_task_read") return keys.length === 1 && keys[0] === "path" && ["api.py", "verify.py"].includes(input.path) ? null : "TASK_READ_SCOPE";
  if (name === "drift_task_write") return keys.length === 2 && keys.includes("path") && keys.includes("text") && input.path === "api.py" && typeof input.text === "string" && Buffer.byteLength(input.text) <= 16384 ? null : "TASK_WRITE_SCOPE";
  if (name === "drift_task_verify") return keys.length === 0 ? null : "TASK_VERIFY_SCOPE";
  return roomFailure(scope, context, name, input, spawned);
}
export const developmentDecision = (...args) => developmentFailure(...args) === null;
const categories = new Map([["task", "task"], ["hub", "hub"], ["todo", "todo"], ["yield", "yield"], ["drift_task_read", "api_read"], ["drift_task_write", "api_write"], ["drift_task_verify", "api_verify"], ["bash", "shell"], ["read", "filesystem"], ["write", "filesystem"], ["edit", "filesystem"]]);
const category = name => categories.get(name) ?? "other";
export function installDevelopmentGuard(api, scope, record = () => {}) {
  if (realpathSync(scope.root) !== scope.root || scope.parent === scope.child) throw new Error("DEVELOPMENT_SCOPE");
  const spawned = new Set();
  const own = context => { const model = context.models.current() ?? context.model; return { cwd: realpathSync(context.cwd), model: `${model?.provider}/${model?.id}` }; };
  const activate = async (_event, context) => {
    const role = roomRole(scope, own(context));
    if (!role) { await api.setActiveTools([]); throw new Error("DEVELOPMENT_SCOPE"); }
    const tools = developmentTools(role); await api.setActiveTools(tools);
    if (JSON.stringify([...api.getActiveTools()].sort()) !== JSON.stringify([...tools].sort())) throw new Error("DEVELOPMENT_TOOLS");
    record({ kind: "advertised", role, tools });
  };
  api.on("session_start", activate); api.on("before_agent_start", activate);
  api.on("tool_call", (event, context) => {
    const session = context.sessionManager.getSessionId();
    const reason = developmentFailure(scope, own(context), event.toolName, event.input, spawned.has(session));
    if (reason) {
      record({ kind: "blocked", reason, tool_category: category(event.toolName) }); return { block: true, reason: "Restricted development allows only pinned peer coordination and public API task tools." };
    }
    if (event.toolName === "task") spawned.add(session);
    if (event.toolName === "hub") return { input: boundedRoomHub(roomRole(scope, own(context)), event.input) };
  });
}
