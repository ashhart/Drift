export interface Limits {
  max_input_bytes: number; max_output_tokens: number; max_session_tokens: number; max_turns: number; deadline_ms: number;
}
export interface Identity { session: string; worker: string; model_id: string; model_sha256: string; translator_sha256: string; }
export interface Envelope { v: 1; session: string; worker: string; seq: number; op: string; payload: Record<string, unknown>; }
export interface WorkerTransport { send(message: Envelope): void; subscribe(message: (value: unknown) => void, failure: (error: Error) => void): void; close(): void; }
export interface Tool { name: string; description: string; parameters: Record<string, unknown>; }
export interface Terminal { reason: "stop" | "tool_use" | "length"; usage: { input_tokens: number; output_tokens: number }; }
export type StreamEvent = { op: "text"; payload: { text: string } } | { op: "tool_call"; payload: { call_id: string; name: string; arguments: Record<string, unknown> } };
export interface ExpectedBackend { backend: "fixture" | "live"; nativeStates: Array<"retained" | "reconstructed" | "fixture">; }
export const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
export const integer = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
export function fail(code = "PROTOCOL"): never { throw new Error("DRIFT_WORKER_" + code); }
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (record(value)) return "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
  const encoded = JSON.stringify(value);
  if (encoded === undefined) return fail();
  return encoded;
}
export function validateBinding(identity: Identity, limits: Limits): void {
  if (!/^[A-Za-z0-9_.-]{1,96}$/.test(identity.session) || !/^[A-Za-z0-9_.-]{1,96}$/.test(identity.worker)) fail();
  if (typeof identity.model_id !== "string" || !identity.model_id || identity.model_id.length > 256) fail();
  if (![identity.model_sha256, identity.translator_sha256].every(value => /^[a-f0-9]{64}$/.test(value))) fail();
  const keys = ["max_input_bytes", "max_output_tokens", "max_session_tokens", "max_turns", "deadline_ms"];
  if (Object.keys(limits).sort().join() !== keys.sort().join() || !Object.values(limits).every(value => integer(value) && value > 0)) fail("LIMIT");
  if (limits.max_input_bytes > 4 * 1024 * 1024 || limits.deadline_ms > 300000 || limits.max_output_tokens > limits.max_session_tokens) fail("LIMIT");
}
export function envelope(value: unknown, identity: Identity, limit: number): Envelope {
  if (!record(value) || Buffer.byteLength(JSON.stringify(value)) > limit) return fail("LIMIT");
  if (Object.keys(value).sort().join() !== ["v", "session", "worker", "seq", "op", "payload"].sort().join()) return fail();
  if (value.v !== 1 || value.session !== identity.session || value.worker !== identity.worker || !integer(value.seq) || value.seq === 0 || typeof value.op !== "string" || !record(value.payload)) return fail();
  return value as unknown as Envelope;
}
