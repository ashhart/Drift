import { expect, test } from "bun:test";
import { WorkerClient } from "../src/worker_client";
import { WorkerProcess } from "../src/worker_process";
import { WorkerContext } from "../src/worker_context";
import { WorkerMapping } from "../src/worker_mapping";
import { workerProvider } from "../src/worker_provider";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const limits = { max_input_bytes: 65536, max_output_tokens: 128, max_session_tokens: 1024, max_turns: 2, deadline_ms: 1000 };
const model = { id: "fixture", api: "fixture", provider: "fixture" };
const tool = { name: "drift_contract_echo", description: "fixture", parameters: {} };
class Sink {
  events: any[] = [];
  private resolve!: (event: any) => void;
  ended = new Promise<any>(resolve => { this.resolve = resolve; });
  push(event: any) { this.events.push(event); if (["done", "error"].includes(event.type)) this.resolve(event); }
}
function makeClient(root: string) {
  const fixture = resolve(import.meta.dir, "../../../scripts/omp/worker_fixture.py");
  return new WorkerClient(new WorkerProcess("/Library/Frameworks/Python.framework/Versions/3.11/bin/python3", ["-S", fixture], { cwd: root, env: { DRIFT_WORKER_REPORT: root + "/worker.json" }, maxBytes: 65536 }),
    { session: "fixture", worker: "fixture", model_id: "fixture", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) }, limits, { backend: "fixture", nativeStates: ["fixture"] });
}
test("owned child maps text, tool call and terminal and sends the result once", async () => {
  const root = mkdtempSync(tmpdir() + "/worker-test-");
  const client = makeClient(root);
  try {
    const provider = workerProvider(client, () => new Sink(), { closeOnStop: true });
    const user = { role: "user", content: "fixture" };
    const first = await provider(model, { messages: [user], tools: [tool] }).ended;
    expect(first.message.errorMessage).toBeUndefined();
    expect(first.reason).toBe("toolUse");
    expect(first.message.content.map((part: any) => part.type)).toEqual(["text", "toolCall"]);
    first.message.content[0].signature = undefined;
    const second = await provider(model, { messages: [user, first.message, { role: "toolResult", toolCallId: "fixture-call", content: [{ type: "text", text: "fixture-ok" }], isError: false }], tools: [tool] }).ended;
    expect(second.reason).toBe("stop");
    expect(client.tokensUsed).toBe(10);
    expect(JSON.parse(await Bun.file(root + "/worker.json").text())).toEqual({ open: 1, own_prompt: 1, tool_result: 1, stream: 2, close: 1 });
  } finally { client.abort(); rmSync(root, { recursive: true, force: true }); }
});
test("changed own history cannot silently reconstruct or replay the worker", async () => {
  const state = new WorkerContext(); const sent: unknown[] = [];
  const client = { open: async () => {}, ownPrompt: async (text: string) => sent.push(text) } as unknown as WorkerClient;
  await state.ingest(client, { messages: [{ role: "user", content: "first" }] });
  await expect(state.ingest(client, { messages: [{ role: "user", content: "changed" }] })).rejects.toThrow("CONTEXT");
  expect(sent).toEqual(["first"]);
});
test("error and length map to OMP terminal events", () => {
  const sink = new Sink(); const mapping = new WorkerMapping(sink, model);
  mapping.done({ reason: "length", usage: { input_tokens: 7, output_tokens: 3 } });
  expect(sink.events.at(-1).reason).toBe("length");
  expect(mapping.message.usage.totalTokens).toBe(10);
  const failure = new Sink(); new WorkerMapping(failure, model).error(true);
  expect(failure.events.at(-1).reason).toBe("aborted");
});

test("only allowlisted codes and phases cross the diagnostic boundary", () => {
  const sink = new Sink(); const mapping = new WorkerMapping(sink, model);
  mapping.error(false, "REMOTE_LIMIT", "stream");
  expect(mapping.message.errorMessage).toBe("Private worker session failed [stream:REMOTE_LIMIT]");
  const other = new WorkerMapping(new Sink(), model); other.error(false, "private text", "private context");
  expect(other.message.errorMessage).toBe("Private worker session failed [unknown:UNKNOWN]");
});

test("text deltas preserve one semantic text block before the tool call", () => {
  const sink = new Sink(); const mapping = new WorkerMapping(sink, model);
  for (const text of ["hello", " ", "world", "\n"]) mapping.event({ op: "text", payload: { text } });
  mapping.event({ op: "tool_call", payload: { call_id: "c", name: "echo", arguments: {} } });
  expect(mapping.message.content).toEqual([{ type: "text", text: "hello world\n" }, { type: "toolCall", id: "c", name: "echo", arguments: {} }]);
  expect(sink.events.filter(event => event.type === "text_start").length).toBe(1);
  expect(sink.events.filter(event => event.type === "text_end").length).toBe(1);
});
