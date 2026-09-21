import { textDuoAllowed } from "../../plugin/omp-drift/src/worker_channels.ts";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai";
import { loadWorkerConfig, validateWorkerConfig } from "../../plugin/omp-drift/src/worker_registration.ts";
import { WorkerClient } from "../../plugin/omp-drift/src/worker_client.ts";
import { WorkerProcess } from "../../plugin/omp-drift/src/worker_process.ts";
import { WorkerRoutes } from "../../plugin/omp-drift/src/worker_routes.ts";
import { workerProvider } from "../../plugin/omp-drift/src/worker_provider.ts";
import { createWorkerBoundary, workerBoundarySettings } from "./worker_boundary_registration.mjs";

export default function experimentalWorkers(api) {
  const config = loadWorkerConfig(process.env.DRIFT_WORKER_CONFIG, process.env.DRIFT_WORKER_CONFIG_SHA256);
  const settings = workerBoundarySettings();
  const boundaryPin = JSON.stringify(settings);
  const key = Symbol.for("drift.experimental.workers." + process.env.DRIFT_WORKER_CONFIG_SHA256);
  const pool = globalThis[key] ??= { active: new Map(), routes: new WorkerRoutes(), boundaryPin, owner: undefined };
  if (pool.boundaryPin !== boundaryPin) throw new Error("DRIFT_WORKER_CAPABILITY");
  const { active, routes } = pool;
  api.registerProvider("drift-experimental", {
    api: "drift-private-worker-v1", baseUrl: "https://example.invalid/private-stdio-only", apiKey: "private-worker-no-network-key",
    models: config.workers.map(entry => ({ id: entry.identity.model_id, name: `EXPERIMENTAL ${entry.identity.model_id} (${entry.expected.nativeStates[0]}, ${entry.memory_mode}, ${entry.communication_mode ?? "default-channels"}, ${entry.session_binding ?? "derived-session"})`, reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: entry.context_window, maxTokens: entry.limits.max_output_tokens })),
    streamSimple(model, context, options) {
      const entry = config.workers.find(worker => worker.identity.model_id === model.id);
      if (!entry) throw new Error("DRIFT_WORKER_CONFIG");
      const route = routes.bind(entry, options?.sessionId);
      if (pool.owner === undefined) pool.owner = options?.sessionId ?? null;
      let session = active.get(route.key);
      if (!session) {
        validateWorkerConfig({ ...config, workers: [entry] });
        const beforeToolDispatch = createWorkerBoundary(settings, { identity: route.identity, ompSession: options?.sessionId, cwd: process.cwd() });
        const command = entry.command;
        const transport = new WorkerProcess(command.executable, command.args, { cwd: command.cwd, env: command.env, maxBytes: entry.limits.max_input_bytes });
        const client = new WorkerClient(transport, route.identity, entry.limits, entry.expected);
        session = { client, ompSession: options?.sessionId, textDuo: textDuoAllowed(entry), generate: workerProvider(client, () => new AssistantMessageEventStream(), { closeOnStop: !entry.experimental_multi_turn, allowSystemUpdates: entry.experimental_multi_turn, allowTextDuo: textDuoAllowed(entry), beforeToolDispatch }) };
        active.set(route.key, session);
      }
      return session.generate(model, context, options);
    },
  });
  api.on("tool_execution_start", (event, context) => {
    const id = context.sessionManager.getSessionId();
    const sessions = [...active.values()].filter(session => session.ompSession === id && session.textDuo);
    for (const session of sessions) session.generate.recordExecution(event.toolCallId, event.toolName, event.args);
  });
  api.on("session_shutdown", async (_event, context) => {
    const id = context.sessionManager.getSessionId();
    // The owning session's teardown ends the process, so a later subagent teardown cannot be relied on to close its worker.
    const terminal = pool.owner === null || pool.owner === id;
    const doomed = [...active.entries()].filter(([, session]) => terminal || session.ompSession === id);
    for (const [routeKey] of doomed) active.delete(routeKey);
    await Promise.all(doomed.map(async ([, session]) => { try { await session.client.close(); } finally { session.client.abort(); } }));
  });
}
