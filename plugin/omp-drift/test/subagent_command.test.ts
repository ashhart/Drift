import { expect, mock, test } from "bun:test";

// The command's default registration imports OMP's stream class, which only OMP provides.
mock.module("@oh-my-pi/pi-ai", () => ({ AssistantMessageEventStream: class {} }));
const { installSubagentCommand } = await import("../../../scripts/omp/subagent_command.mjs");

const workers = ["parent", "child"].map(role => ({ communication_mode: "kv_only", subagent_role: role, identity: { model_id: role } }));

function harness({ closeFails = false } = {}) {
  const calls: string[] = [], notes: [string, string][] = [];
  const api = {
    on() {},
    getActiveTools: () => ["native"],
    setModel: async (model: { id: string }) => { calls.push("model " + model.id); return true; },
    setActiveTools: async (tools: string[]) => { calls.push("tools " + tools.join()); },
  };
  const runtime = { config: { workers }, settings: {}, enable() { calls.push("enable"); },
    async close() { calls.push("close"); if (closeFails) throw new Error("DRIFT_WORKER_CLOSED"); } };
  const release = { epochs: 0, poll: () => "waiting", stop() {} };
  const command = (installSubagentCommand as any)(api, () => runtime, async () => release);
  const context = (session = "owner") => ({
    sessionManager: { getSessionId: () => session, getBranch: () => [] },
    ui: { notify: (text: string, kind: string) => { notes.push([kind, text]); } },
    abort() { calls.push("abort"); }, isIdle: () => true, hasPendingMessages: () => false, cwd: "/",
    models: { resolve: (name: string) => ({ id: name }), current: () => ({ id: "previous" }) },
  });
  return { command, context, calls, notes };
}

test("a refused subagent request leaves the live pair running", async () => {
  const h = harness();
  await h.command("enable", h.context());
  expect(h.calls).toEqual(["model drift-experimental/parent", "enable"]);
  await h.command("enabel", h.context());
  await h.command("disable", h.context("another session"));
  await h.command("enable", h.context());
  expect(h.notes.slice(1).map(([kind]) => kind)).toEqual(["error", "error", "error"]);
  expect(h.calls).not.toContain("close");
  await h.command("status", h.context());
  expect(h.notes.at(-1)).toEqual(["info", "Drift subagent: enabled; 0 staged paired epochs."]);
  await h.command("disable", h.context());
  expect(h.calls.slice(2)).toEqual(["abort", "close", "model previous", "tools native"]);
});

test("disable restores the previous model and tools when a worker does not close in order", async () => {
  const h = harness({ closeFails: true });
  await h.command("enable", h.context());
  await h.command("disable", h.context());
  expect(h.calls.slice(2)).toEqual(["abort", "close", "model previous", "tools native"]);
  expect(h.notes.at(-1)![0]).toBe("error");
  expect(h.notes.at(-1)![1]).toContain("without an ordered close");
});

test("a second disable does not switch the session back again", async () => {
  const h = harness();
  await h.command("enable", h.context());
  await h.command("disable", h.context());
  await h.command("disable", h.context());
  expect(h.calls.filter(call => call.startsWith("model ") || call.startsWith("tools "))).toEqual(["model drift-experimental/parent", "model previous", "tools native"]);
  expect(h.notes.at(-1)).toEqual(["info", "Drift workers closed; native resources require their independent cleanup receipts."]);
});
