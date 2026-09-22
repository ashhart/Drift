import { expect, test } from "bun:test";
import { workerProvider } from "../src/worker_provider";
import { context, deferred, identity, limits, model, setup, Sink, until } from "./worker_boundary_fixture";
const create = workerProvider as any;
const calls = (sink: Sink) => sink.events.filter(event => event.type.startsWith("toolcall") || event.type === "done");

test("worker startup failure poisons the peer boundary before any dispatch hook", async () => {
  const f = setup(); let aborted = 0, entered = 0;
  const hook = Object.assign(async () => { entered++; }, { abort: () => { aborted++; } });
  f.transport.send = () => queueMicrotask(() => f.transport.failure(new Error('startup failed')));
  const sink = create(f.client, () => new Sink(), { beforeToolDispatch: hook })(model, context);
  expect((await sink.ended).type).toBe('error');
  expect(entered).toBe(0); expect(aborted).toBe(1); expect(f.transport.closed).toBe(true);
});

test("terminal gate withholds the bounded tool tail and preserves semantic order", async () => {
  const f = setup(), release = deferred(); let boundary: any;
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: (value: any) => { boundary = value; return release.promise; } });
  const sink = provider(model, context);
  try {
    await until(() => f.transport.active); f.transport.calls(); await Bun.sleep(5);
    expect(boundary).toBeUndefined(); expect(calls(sink)).toEqual([]);
    expect(sink.events.at(-1).partial.content).toEqual([{ type: "text", text: "prefix" }]);
    f.transport.terminal(); await until(() => boundary);
    expect(boundary.identity).toEqual(identity); expect(boundary.toolNames).toEqual(["echo"]);
    expect(Object.keys(boundary).sort()).toEqual(["identity", "signal", "toolCalls", "toolNames"]);
    expect(boundary.toolCalls).toEqual([{ name: 'echo', arguments: { secret: 'public test sentinel' } }]);
    boundary.toolCalls[0].arguments.secret = 'changed by hook';
    expect(boundary.signal).toBeInstanceOf(AbortSignal); expect(boundary.signal.aborted).toBe(false);
    expect(Object.isFrozen(boundary.identity)).toBe(true); expect(Object.isFrozen(boundary.toolNames)).toBe(true);
    expect(calls(sink)).toEqual([]); release.resolve(); const result = await sink.ended;
    expect(result.type).toBe("done");
    expect(result.message.content.map((part: any) => part.type)).toEqual(["text", "toolCall", "text"]);
    expect(result.message.content[2].text).toBe("suffix");
    expect(result.message.content[1].arguments.secret).toBe('public test sentinel');
  } finally { release.resolve(); f.client.abort(); }
});

test("default provider still emits tool events before terminal with no hook", async () => {
  const f = setup(), sink = create(f.client, () => new Sink())(model, context);
  try { await until(() => f.transport.active); f.transport.calls(); expect(calls(sink).length).toBe(2); f.transport.terminal(); expect((await sink.ended).type).toBe("done"); }
  finally { f.client.abort(); }
});

test("user abort while parked closes transport without publishing or provider reuse", async () => {
  const f = setup(), controller = new AbortController(); let signal: AbortSignal | undefined;
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: (value: any) => { signal = value.signal; return new Promise(() => {}); } });
  const sink = provider(model, context, { signal: controller.signal });
  await until(() => f.transport.active); f.transport.calls(); f.transport.terminal(); await until(() => signal);
  controller.abort(); expect((await sink.ended).reason).toBe("aborted");
  expect(signal!.aborted).toBe(true); expect(calls(sink)).toEqual([]); expect(f.transport.closed).toBe(true);
  const sent = f.transport.messages.length; expect((await provider(model, context).ended).type).toBe("error"); expect(f.transport.messages.length).toBe(sent);
});

test("remaining absolute lifetime bounds a hanging hook rather than renewing deadline", async () => {
  let now = 0; const f = setup(() => now); let signal: AbortSignal | undefined;
  const sink = create(f.client, () => new Sink(), { beforeToolDispatch: (value: any) => { signal = value.signal; return new Promise(() => {}); } })(model, context);
  await until(() => f.transport.active); f.transport.calls(); now = limits.deadline_ms - 25;
  const started = performance.now(); f.transport.terminal(); const result = await sink.ended;
  expect(result.type).toBe("error"); expect(result.error.errorMessage).toContain("TIMEOUT");
  expect(performance.now() - started).toBeLessThan(150); expect(signal!.aborted).toBe(true);
  expect(calls(sink)).toEqual([]); expect(f.transport.closed).toBe(true);
});

test("peer rejection and transport failure never release pending tools", async () => {
  for (const fault of ["reject", "transport", "close"]) {
    const f = setup(); let entered = false;
    const provider = create(f.client, () => new Sink(), { beforeToolDispatch: async () => { entered = true; if (fault === "reject") throw new Error("private rejected receipt"); return new Promise(() => {}); } });
    const sink = provider(model, context); await until(() => f.transport.active); f.transport.calls(); f.transport.terminal(); await until(() => entered);
    if (fault === "transport") f.transport.failure(new Error("private transport failure"));
    if (fault === "close") await f.client.close();
    expect((await sink.ended).type).toBe("error"); expect(calls(sink)).toEqual([]); expect(f.transport.closed).toBe(true);
    expect(JSON.stringify(sink.events)).not.toContain("private rejected receipt");
    const sent = f.transport.messages.length; expect((await provider(model, context).ended).type).toBe("error"); expect(f.transport.messages.length).toBe(sent);
  }
});

test("concurrent generation cannot release or displace the parked transaction", async () => {
  const f = setup(), release = deferred(); let entered = false;
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: () => { entered = true; return release.promise; } });
  const first = provider(model, context);
  try {
    await until(() => f.transport.active); f.transport.calls(); f.transport.terminal(); await until(() => entered);
    expect((await provider(model, context).ended).type).toBe("error"); expect(f.transport.closed).toBe(false); expect(calls(first)).toEqual([]);
    release.resolve(); expect((await first.ended).type).toBe("done");
  } finally { release.resolve(); f.client.abort(); }
});

test("a no-tool terminal also waits before done becomes visible", async () => {
  const f = setup(), release = deferred(); let names: string[] | undefined;
  const sink = create(f.client, () => new Sink(), { beforeToolDispatch: (value: any) => { names = value.toolNames; return release.promise; } })(model, context);
  try {
    await until(() => f.transport.active); f.transport.emit("text", { text: "complete" }); f.transport.terminal("stop"); await until(() => names);
    expect(names).toEqual([]); expect(calls(sink)).toEqual([]); release.resolve(); expect((await sink.ended).type).toBe("done");
  } finally { release.resolve(); f.client.abort(); }
});

test("late hook resolution after EOF cannot publish calls or revive provider", async () => {
  const f = setup(), release = deferred(); let signal: AbortSignal | undefined;
  const provider = create(f.client, () => new Sink(), { beforeToolDispatch: (value: any) => { signal = value.signal; return release.promise; } });
  const sink = provider(model, context);
  await until(() => f.transport.active); f.transport.calls(); f.transport.terminal(); await until(() => signal);
  f.transport.failure(new Error("EOF")); expect((await sink.ended).type).toBe("error"); expect(signal!.aborted).toBe(true);
  release.resolve(); await Bun.sleep(5); expect(calls(sink)).toEqual([]);
  expect(sink.events.filter(event => event.type === "error").length).toBe(1);
  const sent = f.transport.messages.length; expect((await provider(model, context).ended).type).toBe("error"); expect(f.transport.messages.length).toBe(sent);
});

test("worker output limit remains authoritative before tool buffering or hook", async () => {
  const f = setup(); let entered = false;
  const sink = create(f.client, () => new Sink(), { beforeToolDispatch: async () => { entered = true; } })(model, context);
  await until(() => f.transport.active);
  for (let index = 0; index < 3; index++) f.transport.emit("tool_call", { call_id: "call" + index, name: "echo", arguments: { value: "x".repeat(1500) } });
  const result = await sink.ended;
  expect(result.type).toBe("error"); expect(result.error.errorMessage).toContain("LIMIT");
  expect(entered).toBe(false); expect(calls(sink)).toEqual([]); expect(f.transport.closed).toBe(true);
});
