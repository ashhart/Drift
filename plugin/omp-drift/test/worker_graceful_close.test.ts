import { expect, test } from "bun:test";
import { workerProvider } from "../src/worker_provider";
import { context, deferred, limits, model, setup, Sink, Transport, until } from "./worker_boundary_fixture";
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
