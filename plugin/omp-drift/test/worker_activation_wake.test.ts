import { expect, test } from "bun:test";
import { WorkerClient } from "../src/worker_client";
import type { Envelope, WorkerTransport } from "../src/worker_protocol";

const identity = { session: "wake-session", worker: "child", model_id: "fixture/model", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) };
const limits = { max_input_bytes: 32768, max_output_tokens: 8, max_session_tokens: 32, max_turns: 2, deadline_ms: 1000 };
const receipt = { v: 1, session: identity.session, seq: 3, op: "appended", source_worker: "parent", target_worker: "child", rows: 2, sha256: "c".repeat(64), foreign_total: 4 };

class Transport implements WorkerTransport {
  messages: Envelope[] = [];
  listener = (_value: unknown) => {};
  closed = false;
  wakeReply: Record<string, unknown> = { activation_seq: 3, foreign_total: 4 };
  send(message: Envelope): void {
    this.messages.push(message);
    queueMicrotask(() => {
      const payload = message.op === "open" ? { model_id: identity.model_id, model_sha256: identity.model_sha256,
        translator_sha256: identity.translator_sha256, limits, backend: "live", capabilities: { native_state: "retained", tool_calls: true, cancellation: true } } : this.wakeReply;
      this.listener({ ...message, op: message.op === "open" ? "opened" : "activation_wake_ack", payload });
    });
  }
  subscribe(listener: (value: unknown) => void): void { this.listener = listener; }
  close(): void { this.closed = true; }
}

async function setup() {
  const transport = new Transport();
  const client = new WorkerClient(transport, identity, limits, { backend: "live", nativeStates: ["retained"] });
  await client.open([], []);
  return { transport, client };
}

test("wake sends only the bound applied sequence and row count, never a prompt", async () => {
  const { client, transport } = await setup();
  try {
    await client.activationWake(receipt);
    expect(transport.messages.map(message => message.op)).toEqual(["open", "activation_wake"]);
    expect(transport.messages[1].payload).toEqual({ activation_seq: 3, foreign_total: 4 });
  } finally { client.abort(); }
});

for (const mutation of [{ session: "other" }, { target_worker: "other" }, { op: "staged" }, { seq: 0 },
  { foreign_total: 1 }, { rows: 0 }, { sha256: "invalid" }, { text: "forbidden" }, { source_worker: "" },
  { source_worker: "child" }, { foreign_total: 1_000_001 }, { rows: 4097, foreign_total: 4097 }]) {
  test(`wake refuses an unbound or widened applied receipt ${JSON.stringify(mutation)}`, async () => {
    const { client, transport } = await setup();
    try {
      await expect(client.activationWake({ ...receipt, ...mutation })).rejects.toThrow("DRIFT_WORKER_PROTOCOL");
      expect(transport.messages).toHaveLength(1);
    } finally { client.abort(); }
  });
}

for (const reply of [{ activation_seq: 2, foreign_total: 4 }, { activation_seq: 3, foreign_total: 4, extra: true }]) {
  test("wake poisons the client on an altered acknowledgement", async () => {
    const { client, transport } = await setup();
    transport.wakeReply = reply;
    await expect(client.activationWake(receipt)).rejects.toThrow("DRIFT_WORKER_PROTOCOL");
    expect(transport.closed).toBe(true);
  });
}
