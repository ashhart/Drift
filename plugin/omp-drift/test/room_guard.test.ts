import { expect, test } from "bun:test";
import { roomDecision } from "../../../scripts/omp/room_policy.mjs";
const scope = { parent: "drift-experimental/glm", child: "drift-experimental/qwen", root: "/scratch" };
const parent = { model: scope.parent, cwd: scope.root };
const child = { model: scope.child, cwd: scope.root };
const valid = { context: "room", tasks: [{ agent: "duo-peer", name: "DuoPeer", task: "coordinate" }] };
test("room qualification admits only one pinned peer and rejects dispatch overrides", () => {
  expect(roomDecision(scope, parent, "task", valid, false)).toBe(true);
  for (const input of [{ ...valid, model: "other" }, { tasks: [{ ...valid.tasks[0], cwd: "/elsewhere" }] }, { tasks: [{ ...valid.tasks[0], agent: "other" }] }, { tasks: [...valid.tasks, ...valid.tasks] }]) expect(roomDecision(scope, parent, "task", input, false)).toBe(false);
  expect(roomDecision(scope, parent, "task", valid, true)).toBe(false);
  expect(roomDecision(scope, child, "task", valid, false)).toBe(false);
});
test("child cannot use filesystem/shell tools and hub cannot address outside the room", () => {
  for (const tool of ["read", "bash", "edit", "write", "ast_edit"]) expect(roomDecision(scope, child, tool, {}, false)).toBe(false);
  expect(roomDecision(scope, child, "hub", { op: "send", to: "Main", message: "hi" }, false)).toBe(true);
  expect(roomDecision(scope, child, "hub", { op: "send", to: "all", message: "hi" }, false)).toBe(false);
  expect(roomDecision(scope, { ...child, cwd: "/elsewhere" }, "yield", {}, false)).toBe(false);
});

test("execution hooks recheck model, scratch scope and one-dispatch budget", async () => {
  const { installRoomGuard } = await import("../../../scripts/omp/room_guard.mjs");
  const { realpathSync } = await import("node:fs");
  const root = realpathSync(process.cwd()), hooks = new Map<string, Function>();
  let active: string[] = ["read", "bash", "edit", "task"], model = "glm";
  const api = { on: (name: string, fn: Function) => hooks.set(name, fn), setActiveTools: async (value: string[]) => { active = value; }, getActiveTools: () => active };
  const context = { models: { current: () => ({ provider: "drift-experimental", id: model }) }, cwd: root, sessionManager: { getSessionId: () => model } };
  installRoomGuard(api, { ...scope, root });
  await hooks.get("session_start")!({}, context);
  expect(active).toEqual(["task", "hub", "todo"]);
  const call = hooks.get("tool_call")!;
  expect(call({ toolName: "hub", input: { op: "wait" } }, context)).toEqual({ input: { op: "wait", from: "DuoPeer", timeoutMs: 2000 } });
  expect(call({ toolName: "task", input: valid }, context)).toBeUndefined();
  expect(call({ toolName: "task", input: valid }, context).block).toBe(true);
  model = "qwen";
  await hooks.get("before_agent_start")!({}, context);
  expect(active).toEqual(["hub", "yield"]);
  for (const name of ["read", "bash", "edit", "task"]) expect(call({ toolName: name, input: {} }, context).block).toBe(true);
  model = "other";
  expect(call({ toolName: "hub", input: { op: "send", to: "Main", message: "x" } }, context).block).toBe(true);
  await expect(hooks.get("before_agent_start")!({}, context)).rejects.toThrow("ROOM_SCOPE");
  expect(active).toEqual([]);
});

test("stock Duo flat single-peer tasks preserve all dispatch restrictions", () => {
  const flat = { context: "room", agent: "duo-peer", name: "DuoPeer", task: "coordinate" };
  expect(roomDecision(scope, parent, "task", flat, false)).toBe(true);
  for (const patch of [{ model: "other" }, { cwd: "/elsewhere" }, { agent: "other" }, { name: "OtherPeer" }, { tools: ["bash"] }, { tasks: [valid.tasks[0]] }, { outputSchema: {} }]) expect(roomDecision(scope, parent, "task", { ...flat, ...patch }, false)).toBe(false);
  expect(roomDecision(scope, child, "task", flat, false)).toBe(false);
  expect(roomDecision(scope, parent, "task", flat, true)).toBe(false);
});

test("room diagnostics expose only bounded schema names and allowlisted error codes", async () => {
  const { roomArgumentShape, roomTerminalCode } = await import("../../../scripts/omp/room_diagnostics.mjs");
  const summary = roomArgumentShape({ task: "PRIVATE TEXT", model: "PRIVATE MODEL", "PRIVATE KEY": 1 });
  expect(summary).toEqual({ shape: "flat", keys: ["model", "task"], unknown_keys: 1 });
  expect(JSON.stringify(summary)).not.toContain("PRIVATE");
  expect(roomTerminalCode("private detail [stream:REMOTE_WORKER]")).toEqual({ phase: "stream", code: "REMOTE_WORKER" });
  expect(roomTerminalCode("private detail")).toEqual({ phase: "unknown", code: "UNKNOWN" });
});

test("batch identity metadata must repeat the same exact single pinned peer", () => {
  const hybrid = { ...valid, agent: "duo-peer", name: "DuoPeer" };
  expect(roomDecision(scope, parent, "task", hybrid, false)).toBe(true);
  expect(roomDecision(scope, parent, "task", { ...valid, agent: "duo-peer" }, false)).toBe(true);
  expect(roomDecision(scope, parent, "task", { ...valid, name: "DuoPeer" }, false)).toBe(true);
  for (const patch of [{ agent: "other" }, { name: "OtherPeer" }, { model: "other" }, { cwd: "/elsewhere" }, { tools: ["bash"] }, { outputSchema: {} }, { task: "ambiguous second task" }]) expect(roomDecision(scope, parent, "task", { ...hybrid, ...patch }, false)).toBe(false);
  for (const patch of [{ agent: "other" }, { name: "OtherPeer" }, { model: "other" }, { cwd: "/elsewhere" }, { tools: ["bash"] }, { outputSchema: {} }]) expect(roomDecision(scope, parent, "task", { ...hybrid, tasks: [{ ...valid.tasks[0], ...patch }] }, false)).toBe(false);
  expect(roomDecision(scope, parent, "task", { ...hybrid, tasks: [...valid.tasks, ...valid.tasks] }, false)).toBe(false);
  expect(roomDecision(scope, child, "task", hybrid, false)).toBe(false);
  expect(roomDecision(scope, parent, "task", hybrid, true)).toBe(false);
});
