import { fail } from "./worker_protocol";
import type { OwnerConfig } from "./worker_registration";

export function requireKvExchange(config: OwnerConfig, settings: { boundaryPath?: string; exchangePath?: string }): boolean {
  if (!config.workers.some(entry => entry.communication_mode === "kv_only")) return false;
  if (config.workers.length !== 2 || !config.workers.every(entry => entry.communication_mode === "kv_only")
      || !settings.boundaryPath || !settings.exchangePath) fail("CAPABILITY");
  if (config.workers.some(entry => entry.subagent_role !== undefined)
      && config.workers.map(entry => entry.subagent_role).sort().join() !== "child,parent") fail("CAPABILITY");
  return true;
}
