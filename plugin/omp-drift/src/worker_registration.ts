import { validateChannels, type WorkerChannels } from "./worker_channels";
import { createHash } from "node:crypto";
import { readFileSync, realpathSync } from "node:fs";
import { isAbsolute } from "node:path";
import { fail, record, validateBinding, type Identity, type Limits, type ExpectedBackend } from "./worker_protocol";

export interface Command { executable: string; sha256: string; args: string[]; cwd: string; env: Record<string, string>; }
export interface WorkerEntry extends WorkerChannels { experimental_multi_turn?: boolean; identity: Identity; limits: Limits; expected: ExpectedBackend; context_window: number; memory_mode: "no-link" | "linked"; command: Command; artifacts: Array<{ path: string; sha256: string }>; }
export interface OwnerConfig { version: 1; experimental: true; workers: WorkerEntry[]; }
const hash = (data: Buffer): string => createHash("sha256").update(data).digest("hex");
function pinned(path: string, expected: string): void {
  if (typeof path !== "string" || !isAbsolute(path) || !/^[a-f0-9]{64}$/.test(expected) || hash(readFileSync(realpathSync(path))) !== expected) fail("PIN");
}
export function validateWorkerConfig(value: unknown): OwnerConfig {
  if (!record(value) || value.version !== 1 || value.experimental !== true || !Array.isArray(value.workers) || !value.workers.length || value.workers.length > 2) fail("CONFIG");
  const ids = new Set<string>(); const workers = new Set<string>();
  for (const entry of value.workers) {
    if (!record(entry) || !record(entry.identity) || !record(entry.limits) || !record(entry.command) || !record(entry.expected)) fail("CONFIG");
    if (entry.experimental_multi_turn !== undefined && typeof entry.experimental_multi_turn !== "boolean") fail("CONFIG");
    validateChannels(entry as unknown as WorkerChannels);
    validateBinding(entry.identity as unknown as Identity, entry.limits as unknown as Limits);
    if (!["no-link", "linked"].includes(String(entry.memory_mode))) fail("CONFIG");
    const id = String(entry.identity.model_id), worker = String(entry.identity.worker);
    if (ids.has(id) || workers.has(worker) || !Number.isSafeInteger(entry.context_window) || Number(entry.context_window) < Number(entry.limits.max_output_tokens)) fail("CONFIG");
    ids.add(id); workers.add(worker);
    if (!["live", "fixture"].includes(String(entry.expected.backend)) || !Array.isArray(entry.expected.nativeStates) || entry.expected.nativeStates.length !== 1) fail("CONFIG");
    const state = entry.expected.nativeStates[0];
    if (!["retained", "reconstructed", "fixture"].includes(state) || (entry.expected.backend === "fixture") !== (state === "fixture")) fail("CONFIG");
    const command = entry.command;
    if (!Array.isArray(command.args) || command.args.some(arg => typeof arg !== "string") || typeof command.cwd !== "string" || !isAbsolute(command.cwd) || !record(command.env) || Object.values(command.env).some(v => typeof v !== "string")) fail("CONFIG");
    if (!Array.isArray(entry.artifacts) || !entry.artifacts.length) fail("PIN");
    pinned(String(command.executable), String(command.sha256));
    for (const artifact of entry.artifacts) { if (!record(artifact)) fail("PIN"); pinned(String(artifact.path), String(artifact.sha256)); }
  }
  return value as unknown as OwnerConfig;
}
export function loadWorkerConfig(path: string, sha256: string): OwnerConfig {
  const raw = readFileSync(path);
  if (!/^[a-f0-9]{64}$/.test(sha256) || hash(raw) !== sha256) fail("PIN");
  return validateWorkerConfig(JSON.parse(raw.toString("utf8")));
}
