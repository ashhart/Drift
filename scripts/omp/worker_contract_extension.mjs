import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai";
import { WorkerClient } from "./worker_client.ts";
import { WorkerProcess } from "./worker_process.ts";
import { workerProvider } from "./worker_provider.ts";

const facts = { registered: false, stream_calls: 0, tool_calls: 0, tool_events: 0, other_tools: 0, closed: false, tokens: 0 };
const save = () => writeFileSync(process.env.DRIFT_CONTRACT_REPORT, JSON.stringify(facts));
const limits = { max_input_bytes: 65536, max_output_tokens: 128, max_session_tokens: 1024, max_turns: 2, deadline_ms: 5000 };
export default function extension(api) {
  const transport = new WorkerProcess("/usr/bin/python3", ["-S", fileURLToPath(new URL("./worker_fixture.py", import.meta.url))], {
    cwd: process.cwd(), env: { PATH: "/usr/bin:/bin", DRIFT_WORKER_REPORT: process.env.DRIFT_WORKER_REPORT }, maxBytes: limits.max_input_bytes,
  });
  const client = new WorkerClient(transport, { session: "fixture-session", worker: "fixture-worker", model_id: "fixture", model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) }, limits, { backend: "fixture", nativeStates: ["fixture"] });
  const generate = workerProvider(client, () => {
    const stream = new AssistantMessageEventStream();
    const push = stream.push.bind(stream);
    stream.push = event => { if (event.type === "done" && event.reason !== "toolUse") { facts.closed = true; facts.tokens = client.tokensUsed; save(); } push(event); };
    return stream;
  }, { closeOnStop: true });
  api.registerTool({ name: "drift_contract_echo", label: "Fixture", description: "Return fixed fixture value", parameters: { type: "object", properties: {} },
    async execute() { facts.tool_calls++; save(); return { content: [{ type: "text", text: "fixture-ok" }], details: {} }; } });
  api.on("session_start", async () => api.setActiveTools(["drift_contract_echo"]));
  api.on("tool_call", event => { if (event.toolName === "drift_contract_echo") facts.tool_events++; else facts.other_tools++; save(); });
  api.on("agent_end", async () => { await client.close(); facts.closed = true; facts.tokens = client.tokensUsed; save(); });
  api.registerProvider("drift-worker-fixture", { api: "drift-worker-fixture-api", baseUrl: "https://example.invalid/never-called", apiKey: "synthetic-not-a-credential",
    models: [{ id: "fixture", name: "Fixture", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 128 }],
    streamSimple(model, context, options) { facts.stream_calls++; save(); return generate(model, context, options); },
  });
  facts.registered = true; save();
}
