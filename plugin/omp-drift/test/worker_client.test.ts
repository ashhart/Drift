import { describe, expect, test } from "bun:test";
import { WorkerClient } from "../src/worker_client";
import type { Envelope, WorkerTransport } from "../src/worker_protocol";

const identity = { session: "fixture-session", worker: "fixture-worker", model_id: "fixture/model", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) };
const limits = { max_input_bytes: 32768, max_output_tokens: 16, max_session_tokens: 32, max_turns: 4, deadline_ms: 100 };

class Fixture implements WorkerTransport {
  messages: Envelope[] = [];
  listener = (_: unknown) => {};
  closed = false;
  nativeState = "fixture";
  backend = "fixture";
  reply = (request: Envelope): void => {
    const payload = request.op === "open" ? { model_id: identity.model_id, model_sha256: identity.model_sha256, translator_sha256: identity.translator_sha256, limits, backend: this.backend, capabilities: { native_state: this.nativeState, tool_calls: true, cancellation: true } } : request.op === "tool_result" ? { call_id: request.payload.call_id } : {};
    this.emit(request, request.op === "open" ? "opened" : request.op + "_ack", payload);
  };
  send(message: Envelope) { this.messages.push(message); queueMicrotask(() => this.reply(message)); }
  subscribe(listener: (message: unknown) => void, _failure: (error: Error) => void) { this.listener = listener; }
  close() { this.closed = true; }
  emit(request: Envelope, op: string, payload: Record<string, unknown>) { this.listener({ ...request, op, payload }); }
}

function client(transport = new Fixture()) {
  return { transport, client: new WorkerClient(transport, identity, limits, { backend: "fixture", nativeStates: ["fixture"] }) };
}

describe("private worker client", () => {
  test("binds an exact manifest and sends own prompts on the private channel", async () => {
    const f = client();
    await f.client.open(["own system"], []);
    await f.client.ownPrompt("own prompt");
    expect(f.transport.messages.map(m => m.op)).toEqual(["open", "own_prompt"]);
    expect(f.transport.messages[1].payload).toEqual({ text: "own prompt" });
  });
  test("additive controls use the exact new private operation and reject widened acknowledgements", async () => {
    const f = client(); await f.client.open([], []);
    await f.client.ownControlAppend(["fresh control"]);
    expect(f.transport.messages[1].op).toBe("own_control_append");
    expect(f.transport.messages[1].payload).toEqual({ system_prompt: ["fresh control"] });
    await expect(f.client.ownControlAppend([])).rejects.toThrow("PROTOCOL");
    f.transport.reply = request => f.transport.emit(request, "own_control_append_ack", { unexpected: true });
    await expect(f.client.ownControlAppend(["next"])).rejects.toThrow("PROTOCOL");
    expect(f.transport.closed).toBe(true);
  });
  test("rejects changed worker identity and poisons the client", async () => {
    const f = client();
    f.transport.reply = request => f.transport.listener({ ...request, session: "other", op: "opened", payload: {} });
    await expect(f.client.open([], [])).rejects.toThrow("PROTOCOL");
    expect(f.transport.closed).toBe(true);
  });
  test("refuses unqualified backend capability", async () => {
    const f = client();
    f.transport.nativeState = "reconstructed";
    await expect(f.client.open([], [])).rejects.toThrow("CAPABILITY");
  });
  test("accounts prefill and refuses duplicate tool results", async () => {
    const f = client();
    await f.client.open([], [{ name: "echo", description: "fixture", parameters: {} }]);
    const defaultReply = f.transport.reply;
    f.transport.reply = request => {
      if (request.op !== "stream") return defaultReply(request);
      f.transport.emit(request, "tool_call", { call_id: "call-1", name: "echo", arguments: {} });
      f.transport.emit(request, "terminal", { reason: "tool_use", usage: { input_tokens: 9, output_tokens: 2 } });
    };
    await f.client.stream(16, () => {});
    await f.client.toolResult("call-1", "result", false);
    await expect(f.client.toolResult("call-1", "duplicate", false)).rejects.toThrow("PROTOCOL");
    expect(f.client.tokensUsed).toBe(11);
  });
  test("rejects extra opened capability fields instead of accepting a widened contract", async () => {
    const f = client();
    const reply = f.transport.reply;
    const emit = f.transport.emit.bind(f.transport);
    f.transport.emit = (request, op, payload) => emit(request, op, { ...payload, surprise: true });
    await expect(f.client.open([], [])).rejects.toThrow("CAPABILITY");
    f.transport.reply = reply;
  });
  test("accepts explicitly qualified reconstructed sessions and counts every prefill", async () => {
    const transport = new Fixture(); transport.nativeState = "reconstructed"; transport.backend = "live";
    const worker = new WorkerClient(transport, identity, limits, { backend: "live", nativeStates: ["reconstructed"] });
    await worker.open([], []);
    transport.reply = request => transport.emit(request, "terminal", { reason: "stop", usage: { input_tokens: 8, output_tokens: 1 } });
    await worker.stream(8, () => {}); await worker.stream(8, () => {});
    expect(worker.tokensUsed).toBe(18);
  });
  test("one absolute deadline bounds every operation instead of renewing its budget", async () => {
    const transport = new Fixture(); let now = 0;
    const worker = new WorkerClient(transport, identity, limits, { backend: "fixture", nativeStates: ["fixture"] }, () => now);
    await worker.open([], []); now = limits.deadline_ms + 1;
    await expect(worker.ownPrompt("late")).rejects.toThrow("TIMEOUT");
    expect(transport.messages.length).toBe(1);
    expect(transport.closed).toBe(true);
  });
  test("acknowledged cancellation rejects the matching stream", async () => {
    const f = client(); await f.client.open([], []);
    f.transport.reply = request => { if (request.op === "cancel") f.transport.emit(request, "cancelled", { target_seq: request.payload.target_seq }); };
    const streamed = f.client.stream(16, () => {}).catch(error => error);
    await f.client.cancel();
    expect((await streamed).message).toBe("DRIFT_WORKER_CANCELLED");
    expect(f.client.activeSequence).toBeUndefined();
    f.client.abort();
  });
  test("cancellation waits for acknowledgement and poisons on missing ack", async () => {
    const f = client();
    await f.client.open([], []);
    f.transport.reply = () => {};
    const streamed = f.client.stream(16, () => {}).catch(error => error);
    await expect(f.client.cancel()).rejects.toThrow();
    expect(f.transport.closed).toBe(true);
    expect(await streamed).toBeInstanceOf(Error);
  });
});
