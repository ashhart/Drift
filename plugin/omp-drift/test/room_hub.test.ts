import { test, expect } from "bun:test";
import { roomDecision } from "../../../scripts/omp/room_policy.mjs";
import { boundedRoomHub } from "../../../scripts/omp/room_hub.mjs";
const scope = { root: "/scratch", parent: "p/a", child: "p/b" }, parent = { cwd: scope.root, model: scope.parent };
test("fresh room supports actual discovery and bounded peer-message wait forms", () => {
  for (const input of [{ op: "list" }, { op: "wait" }, { op: "wait", from: "DuoPeer" }, { op: "wait", ids: ["DuoPeer"], timeoutMs: 0 }]) expect(roomDecision(scope, parent, "hub", input, false)).toBe(true);
  expect(boundedRoomHub("parent", { op: "wait" })).toEqual({ op: "wait", from: "DuoPeer", timeoutMs: 2000 });
  expect(boundedRoomHub("child", { op: "wait", timeoutMs: 0 })).toEqual({ op: "wait", from: "Main", timeoutMs: 2000 });
  expect(boundedRoomHub("parent", { op: "list" })).toEqual({ op: "list", limit: 2 });
});
test("process control, unrelated peers, history and long waits remain rejected", () => {
  for (const input of [{ op: "start" }, { op: "jobs" }, { op: "inbox" }, { op: "list", status: "parked" }, { op: "list", limit: 100 }, { op: "wait", from: "other" }, { op: "wait", ids: ["other"] }, { op: "wait", timeoutMs: 3000 }, { op: "wait", name: "process" }, { op: "send", to: "all", message: "x" }]) expect(roomDecision(scope, parent, "hub", input, false)).toBe(false);
});
