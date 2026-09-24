import { fail } from "./worker_protocol";
export interface WorkerChannels {
  experimental_multi_turn?: boolean;
  memory_mode: "no-link" | "linked";
  communication_mode?: "text_and_artifacts" | "kv_only";
  session_binding?: "fixed";
  subagent_role?: "parent" | "child";
}
export function validateChannels(entry: WorkerChannels): void {
  if (entry.communication_mode !== undefined && (!["text_and_artifacts", "kv_only"].includes(entry.communication_mode) || entry.experimental_multi_turn !== true)) fail("CONFIG");
  if (entry.session_binding !== undefined && (entry.session_binding !== "fixed" || entry.memory_mode !== "linked" || entry.experimental_multi_turn !== true || entry.communication_mode === undefined)) fail("CONFIG");
  if (entry.communication_mode === "kv_only" && (entry.memory_mode !== "linked" || entry.session_binding !== "fixed")) fail("CONFIG");
  if (entry.subagent_role !== undefined && (entry.communication_mode !== "kv_only" || !["parent", "child"].includes(entry.subagent_role))) fail("CONFIG");
}
export function textDuoAllowed(entry: WorkerChannels): boolean {
  validateChannels(entry);
  return entry.experimental_multi_turn === true && (entry.memory_mode === "no-link" || entry.communication_mode === "text_and_artifacts");
}
