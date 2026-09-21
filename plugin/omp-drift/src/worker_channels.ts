import { fail } from "./worker_protocol";
export interface WorkerChannels {
  experimental_multi_turn?: boolean;
  memory_mode: "no-link" | "linked";
  communication_mode?: "text_and_artifacts";
  session_binding?: "fixed";
}
export function validateChannels(entry: WorkerChannels): void {
  if (entry.communication_mode !== undefined && (entry.communication_mode !== "text_and_artifacts" || entry.experimental_multi_turn !== true)) fail("CONFIG");
  if (entry.session_binding !== undefined && (entry.session_binding !== "fixed" || entry.memory_mode !== "linked" || entry.experimental_multi_turn !== true || entry.communication_mode !== "text_and_artifacts")) fail("CONFIG");
}
export function textDuoAllowed(entry: WorkerChannels): boolean {
  validateChannels(entry);
  return entry.experimental_multi_turn === true && (entry.memory_mode === "no-link" || entry.communication_mode === "text_and_artifacts");
}
