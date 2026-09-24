import { canonical, fail, type Tool } from "./worker_protocol";
import type { BeforeToolDispatch } from "./worker_boundary";

export type KvRole = "parent" | "child";
export const kvTask = Object.freeze({ agent: "drift-peer", name: "DriftPeer", task: "Follow your own preassigned Drift role.", context: "Drift lifecycle control only." });
export const kvYield = Object.freeze({ data: Object.freeze({ complete: true }) });

export function kvLifecycleTool(role: KvRole): Tool {
  return { name: role === "parent" ? "task" : "yield",
    description: role === "parent" ? "Start the preassigned Drift peer once without sending it a prompt." : "Finish this Drift participant without sending its answer to the parent.",
    parameters: { type: "object", properties: {}, additionalProperties: false } };
}

export function lifecycleBoundary(exchange: BeforeToolDispatch, role: KvRole): BeforeToolDispatch {
  let admitted = false, finished = false;
  const run: BeforeToolDispatch = async boundary => {
    if (finished || boundary.toolCalls.length > 1 || boundary.toolNames.length !== boundary.toolCalls.length) fail("CAPABILITY");
    const call = boundary.toolCalls[0];
    if (call && (canonical(call.arguments) !== "{}" || boundary.toolNames[0] !== call.name)) fail("CAPABILITY");
    if (call?.name === "task") {
      if (role !== "parent" || admitted) fail("CAPABILITY");
      admitted = true;
    } else {
      if (role === "parent" && !admitted) fail("CAPABILITY");
      if (call && call.name !== "drift_sync" && !(role === "child" && call.name === "yield")) fail("CAPABILITY");
      if (role === "child" && !call) fail("CAPABILITY");
      if (!call || call.name === "yield") finished = true;
    }
    await exchange(boundary);
  };
  run.abort = () => { finished = true; exchange.abort?.(); };
  return run;
}
