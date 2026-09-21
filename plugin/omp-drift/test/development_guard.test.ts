import { test, expect } from "bun:test";
import { developmentDecision, developmentTools } from "../../../scripts/omp/development_guard.mjs";
const scope = { root: "/task", parent: "p/glm", child: "p/qwen" };
const child = { cwd: scope.root, model: scope.child };
test("development permits only scoped API tools and retains child recursion rejection", () => {
  expect(developmentTools("child")).toEqual(["hub", "yield", "drift_task_read", "drift_task_write", "drift_task_verify"]);
  expect(developmentDecision(scope, child, "drift_task_read", { path: "api.py" }, false)).toBe(true);
  for (const path of ["../owner.json", "/etc/hosts", "scope.json"]) expect(developmentDecision(scope, child, "drift_task_read", { path }, false)).toBe(false);
  for (const name of ["task", "read", "bash", "edit", "write"]) expect(developmentDecision(scope, child, name, {}, false)).toBe(false);
  expect(developmentDecision(scope, child, "drift_task_write", { path: "verify.py", text: "overwrite" }, false)).toBe(false);
});
test("guard failures report fixed reasons and categories without private argument values", async () => {
  const { installDevelopmentGuard } = await import("../../../scripts/omp/development_guard.mjs");
  const { realpathSync } = await import("node:fs");
  const root = realpathSync(process.cwd()), hooks = new Map<string, Function>(), events: unknown[] = [];
  const api = { on: (name: string, fn: Function) => hooks.set(name, fn) };
  installDevelopmentGuard(api, { ...scope, root }, value => events.push(value));
  const context = { models: { current: () => ({ provider: "p", id: "qwen" }) }, cwd: root, sessionManager: { getSessionId: () => "child" } };
  const call = hooks.get("tool_call")!;
  expect(call({ toolName: "drift_task_read", input: { path: "NEVER_EXPORT_PATH" } }, context).block).toBe(true);
  expect(call({ toolName: "NEVER_EXPORT_TOOL", input: { secret: "NEVER_EXPORT_VALUE" } }, context).block).toBe(true);
  expect(events).toEqual([{ kind: "blocked", reason: "TASK_READ_SCOPE", tool_category: "api_read" }, { kind: "blocked", reason: "TOOL_SCOPE", tool_category: "other" }]);
  expect(JSON.stringify(events)).not.toContain("NEVER_EXPORT");
});
