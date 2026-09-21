import { boundedRoomHub } from "./room_hub.mjs";
import { roomArgumentShape } from "./room_diagnostics.mjs";
import { realpathSync } from "node:fs";
import { roomFailure, roomRole, roomTools } from "./room_policy.mjs";
export function installRoomGuard(api, scope, record = () => {}) {
  if (realpathSync(scope.root) !== scope.root || scope.parent === scope.child) throw new Error("ROOM_SCOPE");
  const spawned = new Set();
  const contextOf = context => {
    const model = context.models.current() ?? context.model;
    return { model: `${model?.provider}/${model?.id}`, cwd: realpathSync(context.cwd) };
  };
  const activate = async (_event, context) => {
    const own = contextOf(context), role = roomRole(scope, own);
    if (!role) { await api.setActiveTools([]); throw new Error("ROOM_SCOPE"); }
    const allowed = roomTools(role); await api.setActiveTools(allowed);
    if (JSON.stringify([...api.getActiveTools()].sort()) !== JSON.stringify([...allowed].sort())) throw new Error("ROOM_TOOL_SCOPE");
    record({ kind: "advertised", role, tools: allowed });
  };
  api.on("session_start", activate);
  api.on("before_agent_start", activate);
  api.on("tool_call", (event, context) => {
    const own = contextOf(context), id = context.sessionManager.getSessionId();
    const reason = roomFailure(scope, own, event.toolName, event.input, spawned.has(id));
    if (reason) {
      record({ kind: "blocked", reason, ...roomArgumentShape(event.input), role: roomRole(scope, own), tool: ["read", "bash", "edit", "task"].includes(event.toolName) ? event.toolName : "other" });
      return { block: true, reason: "Restricted room qualification permits only pinned peer coordination." };
    }
    if (event.toolName === "task") spawned.add(id);
    if (event.toolName === "hub") return { input: boundedRoomHub(roomRole(scope, own), event.input) };
  });
}
