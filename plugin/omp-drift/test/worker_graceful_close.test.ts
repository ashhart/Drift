import { expect, test } from "bun:test";
import { workerProvider } from "../src/worker_provider";
import { WorkerClient } from "../src/worker_client";
import { context, deferred, identity, limits, model, setup, Sink, Transport, until } from "./worker_boundary_fixture";
const create = workerProvider as any;
const ops = (f: any) => f.transport.messages.map((message: any) => message.op);

test("user abort while parked closes the worker gracefully before poisoning the transport", async () => {
  const f = setup(), controller = new AbortController(); let signal: AbortSignal | undefined;
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: (value: any) => { signal = value.signal; return new Promise(() => {}); } });
  const sink = provider(model, context, { signal: controller.signal });
  await until(() => f.transport.active); f.transport.calls(); f.transport.terminal(); await until(() => signal);
  controller.abort(); expect((await sink.ended).reason).toBe("aborted");
  await until(() => f.transport.closed);
  expect(ops(f)).toContain("close");
  expect(ops(f).indexOf("close")).toBeGreaterThan(ops(f).indexOf("cancel"));
  expect(f.transport.closed).toBe(true);
});

test("rejected boundary release still closes the worker gracefully", async () => {
  const f = setup();
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: async () => { throw new Error("private rejected receipt"); } });
  const sink = provider(model, context);
  await until(() => f.transport.active); f.transport.calls(); f.transport.terminal();
  expect((await sink.ended).type).toBe("error");
  await until(() => f.transport.closed);
  expect(ops(f)).toContain("close");
});

test("a worker that never acknowledges close is poisoned within the grace budget", async () => {
  class Stalling extends Transport { send(message: any) { if (message.op === "close") { this.messages.push(message); return; } super.send(message); } }
  const transport = new Stalling();
  const { WorkerClient } = await import("../src/worker_client");
  const { identity } = await import("./worker_boundary_fixture");
  const client = new WorkerClient(transport as any, identity, limits, { backend: "fixture", nativeStates: ["fixture"] });
  const provider = create(client, () => new Sink(), { beforeToolDispatch: async () => { throw new Error("private rejected receipt"); }, closeGraceMs: 40 });
  const sink = provider(model, context);
  await until(() => transport.active); transport.calls(); transport.terminal();
  const started = performance.now();
  expect((await sink.ended).type).toBe("error");
  await until(() => transport.closed);
  expect(performance.now() - started).toBeLessThan(400);
  expect(transport.messages.map(m => m.op)).toContain("close");
  expect(transport.closed).toBe(true);
});

test("an already dead client is not asked to close again", async () => {
  const f = setup(), release = deferred();
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: () => release.promise });
  const sink = provider(model, context);
  await until(() => f.transport.active); f.transport.calls(); f.transport.terminal();
  await until(() => f.transport.messages.length > 0);
  f.transport.failure(new Error("private transport failure"));
  expect((await sink.ended).type).toBe("error");
  release.resolve();
  expect(ops(f).filter((op: string) => op === "close").length).toBe(0);
});

// Answers cancel and close a little later, as the Python worker does: one cancel per turn, then only close.
class Worker extends Transport {
  cancelled = false;
  send(message: any) {
    if (!["cancel", "close"].includes(message.op)) return super.send(message);
    this.messages.push(message);
    setTimeout(() => {
      const refused = message.op === "cancel" && this.cancelled;
      if (message.op === "cancel") this.cancelled = true;
      const op = refused ? "error" : message.op === "cancel" ? "cancelled" : "closed";
      this.listener({ ...message, op, payload: refused ? { code: "PROTOCOL" } : message.op === "cancel" ? { target_seq: message.payload.target_seq } : {} });
    }, 5);
  }
}
const worker = () => { const transport = new Worker(); return { transport, client: new WorkerClient(transport, identity, limits, { backend: "fixture", nativeStates: ["fixture"] }) }; };
const endings = (transport: Transport) => transport.messages.map(message => message.op).filter(op => op === "cancel" || op === "close");

test("a turn's abort and the owner's close of the same worker share one cancel and one close", async () => {
  const { transport, client } = worker(), controller = new AbortController();
  const sink = create(client, () => new Sink(), {})(model, context, { signal: controller.signal });
  await until(() => transport.active);
  controller.abort(); await client.close();
  expect((await sink.ended).reason).toBe("aborted");
  expect(endings(transport)).toEqual(["cancel", "close"]);
  expect(transport.closed).toBe(true);
});

test("the owner's close of a parked turn is not undone by the turn closing the same worker", async () => {
  const { transport, client } = worker(), controller = new AbortController(); let parked = false;
  const sink = create(client, () => new Sink(), { beforeToolDispatch: () => { parked = true; return new Promise(() => {}); } })(model, context, { signal: controller.signal });
  await until(() => transport.active); transport.calls(); transport.terminal(); await until(() => parked);
  controller.abort(); await client.close();
  expect((await sink.ended).reason).toBe("aborted");
  expect(endings(transport)).toEqual(["close"]);
  expect(transport.closed).toBe(true);
});
