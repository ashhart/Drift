import { WorkerClient } from "../src/worker_client";
import type { Envelope, WorkerTransport } from "../src/worker_protocol";
export const identity = { session: "gate-session", worker: "gate-worker", model_id: "fixture", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) };
export const model = { id: "fixture", api: "fixture", provider: "fixture" };
export const context = { messages: [{ role: "user", content: "public fixture" }], tools: [{ name: "echo", description: "fixture", parameters: {} }] };
export const limits = { max_input_bytes: 4096, max_output_tokens: 16, max_session_tokens: 64, max_turns: 4, deadline_ms: 500 };
export class Sink {
  events: any[] = [];
  private resolve!: (event: any) => void;
  ended = new Promise<any>(resolve => { this.resolve = resolve; });
  push(event: any) { this.events.push(structuredClone(event)); if (["done", "error"].includes(event.type)) this.resolve(event); }
}
export class Transport implements WorkerTransport {
  messages: Envelope[] = [];
  listener = (_: unknown) => {};
  failure = (_: Error) => {};
  closed = false;
  active?: Envelope;
  send(message: Envelope) {
    this.messages.push(message);
    queueMicrotask(() => {
      if (message.op === "stream") { this.active = message; return; }
      const payload = message.op === "open" ? { model_id: identity.model_id, model_sha256: identity.model_sha256, translator_sha256: identity.translator_sha256, limits, backend: "fixture", capabilities: { native_state: "fixture", tool_calls: true, cancellation: true } } : {};
      this.listener({ ...message, op: message.op === "open" ? "opened" : message.op === "close" ? "closed" : message.op + "_ack", payload });
    });
  }
  subscribe(listener: (message: unknown) => void, failure: (error: Error) => void) { this.listener = listener; this.failure = failure; }
  close() { this.closed = true; }
  emit(op: string, payload: Record<string, unknown>) { this.listener({ ...this.active!, op, payload }); }
  calls() {
    this.emit("text", { text: "prefix" });
    this.emit("tool_call", { call_id: "one", name: "echo", arguments: { secret: "public test sentinel" } });
    this.emit("text", { text: "suffix" });
  }
  terminal(reason = "tool_use") { this.emit("terminal", { reason, usage: { input_tokens: 3, output_tokens: 2 } }); }
}
export function setup(clock?: () => number) {
  const transport = new Transport();
  const client = new WorkerClient(transport, identity, limits, { backend: "fixture", nativeStates: ["fixture"] }, clock);
  return { transport, client };
}
export async function until(predicate: () => unknown) {
  for (let count = 0; count < 100; count++) { if (predicate()) return; await Bun.sleep(2); }
  throw new Error("fixture condition did not arrive");
}
export function deferred() { let resolve!: () => void; const promise = new Promise<void>(done => { resolve = done; }); return { promise, resolve }; }
