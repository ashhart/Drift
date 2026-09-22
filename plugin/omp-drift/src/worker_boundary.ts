import type { WorkerClient } from "./worker_client";
import type { Identity, StreamEvent } from "./worker_protocol";
import type { WorkerMapping } from "./worker_mapping";

export type BoundaryToolCall = Readonly<{ name: string; arguments: Record<string, unknown> }>;
export type WorkerBoundary = Readonly<{ identity: Readonly<Identity>; toolNames: readonly string[]; toolCalls: readonly BoundaryToolCall[]; signal: AbortSignal }>;
export type BeforeToolDispatch = ((boundary: WorkerBoundary) => Promise<void>) & { abort?: () => void };

export class WorkerToolTail {
  private events: StreamEvent[] = [];
  private buffering = false;
  readonly toolNames: string[] = [];
  readonly toolCalls: BoundaryToolCall[] = [];
  constructor(private mapping: WorkerMapping) {}
  event(event: StreamEvent): void {
    if (event.op === "tool_call") {
      this.buffering = true; this.toolNames.push(event.payload.name);
      this.toolCalls.push(structuredClone({ name: event.payload.name, arguments: event.payload.arguments }));
    }
    if (this.buffering) this.events.push(structuredClone(event));
    else this.mapping.event(event);
  }
  release(): void { for (const event of this.events) this.mapping.event(event); this.events = []; }
}

export async function beforeWorkerDispatch(client: WorkerClient, hook: BeforeToolDispatch, toolNames: readonly string[], userSignal?: AbortSignal, toolCalls: readonly BoundaryToolCall[] = []): Promise<void> {
  const controller = new AbortController();
  const cancel = () => controller.abort(new Error("DRIFT_WORKER_CANCELLED"));
  const closed = () => controller.abort(client.lifecycleSignal.reason);
  const remaining = client.remainingLifetimeMs;
  const timeout = () => controller.abort(new Error("DRIFT_WORKER_TIMEOUT"));
  userSignal?.addEventListener("abort", cancel, { once: true });
  client.lifecycleSignal.addEventListener("abort", closed, { once: true });
  const timer = setTimeout(timeout, Math.max(0, remaining));
  if (userSignal?.aborted) cancel();
  else if (client.lifecycleSignal.aborted) closed();
  else if (remaining <= 0) timeout();
  const boundary = Object.freeze({ identity: Object.freeze(structuredClone(client.identity)), toolNames: Object.freeze([...toolNames]), toolCalls: Object.freeze(toolCalls.map(call => Object.freeze(structuredClone(call)))), signal: controller.signal });
  let rejectAbort: (() => void) | undefined;
  try {
    await new Promise<void>((resolve, reject) => {
      rejectAbort = () => reject(controller.signal.reason);
      controller.signal.addEventListener("abort", rejectAbort, { once: true });
      if (controller.signal.aborted) { rejectAbort(); return; }
      Promise.resolve().then(() => {
        if (controller.signal.aborted) throw controller.signal.reason;
        return hook(boundary);
      }).then(resolve, reject);
    });
    if (controller.signal.aborted) throw controller.signal.reason;
    if (client.remainingLifetimeMs <= 0) { timeout(); throw controller.signal.reason; }
  } catch (error) {
    controller.abort(error); throw error;
  } finally {
    clearTimeout(timer); userSignal?.removeEventListener("abort", cancel);
    client.lifecycleSignal.removeEventListener("abort", closed);
    if (rejectAbort) controller.signal.removeEventListener("abort", rejectAbort);
  }
}
