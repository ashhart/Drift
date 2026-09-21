import { beforeWorkerDispatch, WorkerToolTail, type BeforeToolDispatch } from "./worker_boundary";
import { WorkerContext, type OwnContext } from "./worker_context";
import { WorkerMapping, type EventSink } from "./worker_mapping";
import type { WorkerClient } from "./worker_client";

const CLOSE_GRACE_MS = 1000;

export function workerProvider<T extends EventSink>(client: WorkerClient, createStream: () => T, lifecycle: { closeOnStop?: boolean; allowSystemUpdates?: boolean; allowTextDuo?: boolean; closeGraceMs?: number; beforeToolDispatch?: BeforeToolDispatch } = {}) {
  const contextState = new WorkerContext(lifecycle.allowSystemUpdates, lifecycle.allowTextDuo);
  // A cancelled or rejected turn still owes the worker an ordered close before the transport dies.
  const settle = async (): Promise<void> => {
    if (client.lifecycleSignal.aborted) return;
    const budget = Math.max(0, Math.min(lifecycle.closeGraceMs ?? CLOSE_GRACE_MS, client.remainingLifetimeMs));
    const expiry = new Promise<void>(resolve => { const timer = setTimeout(resolve, budget); timer.unref?.(); });
    await Promise.race([client.close().catch(() => {}), expiry]);
  };
  let busy = false, poisoned = false;
  const generate = (model: { id: string; api: string; provider: string }, context: OwnContext, options?: { signal?: AbortSignal; maxTokens?: number }): T => {
    const stream = createStream();
    const mapping = new WorkerMapping(stream, model);
    const execute = async () => {
      if (poisoned) { mapping.error(false, "CLOSED", "context"); return; }
      if (busy) { mapping.error(false); return; }
      busy = true;
      let cancellation: Promise<void> | undefined;
      let phase = "context";
      const cancel = () => { cancellation = client.cancel(); cancellation.catch(() => {}); };
      try {
        if (model.id !== client.identity.model_id || options?.signal?.aborted) throw new Error("DRIFT_WORKER_CONTEXT");
        await contextState.ingest(client, context);
        if (options?.signal?.aborted) throw new Error("DRIFT_WORKER_CANCELLED");
        phase = "stream";
        const tail = lifecycle.beforeToolDispatch ? new WorkerToolTail(mapping) : undefined;
        const generation = client.stream(options?.maxTokens ?? client.limits.max_output_tokens, event => tail ? tail.event(event) : mapping.event(event));
        options?.signal?.addEventListener("abort", cancel, { once: true });
        const terminal = await generation;
        if (cancellation) { await cancellation; throw new Error("DRIFT_WORKER_CANCELLED"); }
        if (lifecycle.beforeToolDispatch) {
          await beforeWorkerDispatch(client, lifecycle.beforeToolDispatch, tail!.toolNames, options?.signal);
          if (options?.signal?.aborted) throw new Error("DRIFT_WORKER_CANCELLED");
          if (client.lifecycleSignal.aborted) throw client.lifecycleSignal.reason;
          if (client.remainingLifetimeMs <= 0) throw new Error("DRIFT_WORKER_TIMEOUT");
          tail!.release();
        }
        contextState.remember(mapping.message);
        if (lifecycle.closeOnStop && terminal.reason !== "tool_use") { phase = "close"; await client.close(); }
        mapping.done(terminal);
      } catch (error) {
        poisoned = true;
        const code = error instanceof Error && error.message.startsWith("DRIFT_WORKER_") ? error.message.slice(13) : "UNKNOWN";
        let acknowledged = false;
        if (cancellation) { try { await cancellation; acknowledged = true; } catch {} }
        await settle();
        client.abort(); mapping.error(acknowledged || Boolean(lifecycle.beforeToolDispatch && options?.signal?.aborted), code, phase);
      } finally { options?.signal?.removeEventListener("abort", cancel); busy = false; }
    };
    void execute(); return stream;
  };
  return Object.assign(generate, { recordExecution: (id: string, name: string, args: unknown) => contextState.recordExecution(id, name, args) });
}
