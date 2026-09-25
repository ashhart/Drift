import { WorkerLedger } from "./worker_ledger";
import { activationWakeBinding } from "./worker_activation_receipt";
import { canonical, envelope, fail, record, validateBinding, type Envelope, type ExpectedBackend, type Identity, type Limits, type StreamEvent, type Terminal, type Tool, type WorkerTransport } from "./worker_protocol";

type Pending = { op: string; reply: string; resolve: (value: Record<string, unknown>) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout>; event?: (event: StreamEvent) => void; };
export class WorkerClient {
  readonly identity: Identity;
  readonly limits: Limits;
  private ledger: WorkerLedger;
  private lifecycle = new AbortController();
  private seq = 0;
  private opened = false;
  private dead = false;
  private active?: number;
  private deadline?: number;
  private lifetime?: ReturnType<typeof setTimeout>;
  private pending = new Map<number, Pending>();
  private cancelling?: { target: number; done: Promise<void> };
  private closing?: Promise<void>;
  constructor(private transport: WorkerTransport, identity: Identity, limits: Limits, private expected: ExpectedBackend, private clock: () => number = () => performance.now()) {
    validateBinding(identity, limits);
    this.identity = structuredClone(identity); this.limits = structuredClone(limits);
    this.ledger = new WorkerLedger(this.limits);
    transport.subscribe(value => this.receive(value), () => this.poison(new Error("DRIFT_WORKER_CLOSED")));
  }
  get lifecycleSignal(): AbortSignal { return this.lifecycle.signal; }
  get remainingLifetimeMs(): number { return this.dead ? 0 : Math.max(0, (this.deadline ?? this.clock()) - this.clock()); }
  get tokensUsed(): number { return this.ledger.tokensUsed; }
  get activeSequence(): number | undefined { return this.active; }
  private poison(error: Error): void {
    this.dead = true; clearTimeout(this.lifetime); this.lifecycle.abort(error);
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pending.clear(); this.active = undefined; this.transport.close();
  }
  abort(): void { this.poison(new Error("DRIFT_WORKER_CLOSED")); }
  private receive(value: unknown): void {
    if (this.dead) return;
    try {
      const message = envelope(value, this.identity, this.limits.max_input_bytes);
      const pending = this.pending.get(message.seq);
      if (!pending) fail();
      if (message.op === "error") {
        const code = message.payload.code;
        fail(["PROTOCOL", "LIMIT", "CAPABILITY", "WORKER"].includes(String(code)) ? "REMOTE_" + code : "WORKER");
      }
      if (pending.op === "stream" && message.op !== "terminal") {
        pending.event!(this.ledger.event(message.op, message.payload)); return;
      }
      if (message.op !== pending.reply) fail();
      let payload = message.payload;
      if (pending.op === "stream") payload = this.ledger.terminal(payload) as unknown as Record<string, unknown>;
      clearTimeout(pending.timer); this.pending.delete(message.seq);
      if (this.active === message.seq) this.active = undefined;
      pending.resolve(payload);
    } catch (error) { this.poison(error instanceof Error ? error : new Error("DRIFT_WORKER_PROTOCOL")); }
  }
  private request(op: string, reply: string, payload: Record<string, unknown>, event?: (event: StreamEvent) => void): Promise<Record<string, unknown>> {
    if (this.dead || (this.pending.size && op !== "cancel")) return Promise.reject(new Error("DRIFT_WORKER_PROTOCOL"));
    if (op === "open" && this.deadline === undefined) {
      this.deadline = this.clock() + this.limits.deadline_ms;
      this.lifetime = setTimeout(() => this.poison(new Error("DRIFT_WORKER_TIMEOUT")), this.limits.deadline_ms);
      this.lifetime.unref?.();
    }
    const remaining = (this.deadline ?? this.clock()) - this.clock();
    if (remaining <= 0) { this.poison(new Error("DRIFT_WORKER_TIMEOUT")); return Promise.reject(new Error("DRIFT_WORKER_TIMEOUT")); }
    const message: Envelope = { v: 1, session: this.identity.session, worker: this.identity.worker, seq: ++this.seq, op, payload };
    if (Buffer.byteLength(JSON.stringify(message)) > this.limits.max_input_bytes) return Promise.reject(new Error("DRIFT_WORKER_LIMIT"));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.poison(new Error("DRIFT_WORKER_TIMEOUT")), remaining);
      this.pending.set(message.seq, { op, reply, resolve, reject, timer, event });
      if (op === "stream") this.active = message.seq;
      try { this.transport.send(message); } catch { this.poison(new Error("DRIFT_WORKER_TRANSPORT")); }
    });
  }
  async open(systemPrompt: string[], tools: Tool[]): Promise<void> {
    if (this.opened || systemPrompt.some(text => typeof text !== "string")) return fail();
    this.ledger.setTools(tools);
    const { model_id, model_sha256, translator_sha256 } = this.identity;
    const binding = { model_id, model_sha256, translator_sha256, limits: this.limits };
    const result = await this.request("open", "opened", { ...binding, system_prompt: systemPrompt, tools });
    const received = { model_id: result.model_id, model_sha256: result.model_sha256, translator_sha256: result.translator_sha256, limits: result.limits };
    if (Object.keys(result).sort().join() !== "backend,capabilities,limits,model_id,model_sha256,translator_sha256"
        || canonical(binding) !== canonical(received) || result.backend !== this.expected.backend || !record(result.capabilities)
        || Object.keys(result.capabilities).sort().join() !== "cancellation,native_state,tool_calls"
        || !this.expected.nativeStates.includes(result.capabilities.native_state as never) || result.capabilities.tool_calls !== true || result.capabilities.cancellation !== true) {
      this.abort(); return fail("CAPABILITY");
    }
    this.opened = true;
  }
  async ownControl(systemPrompt: string[]): Promise<void> {
    if (!this.opened || !Array.isArray(systemPrompt) || systemPrompt.some(value => typeof value !== "string")) return fail();
    const result = await this.request("own_control", "own_control_ack", { system_prompt: systemPrompt });
    if (Object.keys(result).length) { this.abort(); fail(); }
  }
  async ownControlAppend(systemPrompt: string[]): Promise<void> {
    if (!this.opened || !Array.isArray(systemPrompt) || !systemPrompt.length || systemPrompt.some(value => typeof value !== "string")) return fail();
    const result = await this.request("own_control_append", "own_control_append_ack", { system_prompt: systemPrompt });
    if (Object.keys(result).length) { this.abort(); fail(); }
  }
  async ownPrompt(text: string): Promise<void> {
    if (!this.opened || typeof text !== "string") return fail();
    const result = await this.request("own_prompt", "own_prompt_ack", { text });
    if (Object.keys(result).length) { this.abort(); fail(); }
  }
  async activationWake(appliedReceipt: unknown): Promise<void> {
    if (!this.opened) return fail();
    const payload = activationWakeBinding(appliedReceipt, this.identity);
    const result = await this.request("activation_wake", "activation_wake_ack", { ...payload });
    if (canonical(result) !== canonical(payload)) { this.abort(); fail(); }
  }
  async toolResult(callId: string, text: string, isError: boolean): Promise<void> {
    if (!this.opened || !this.ledger.pending.has(callId) || typeof text !== "string" || typeof isError !== "boolean") return fail();
    const result = await this.request("tool_result", "tool_result_ack", { call_id: callId, text, is_error: isError });
    if (canonical(result) !== canonical({ call_id: callId })) { this.abort(); fail(); }
    this.ledger.pending.delete(callId);
  }
  async stream(maxTokens: number, event: (event: StreamEvent) => void): Promise<Terminal> {
    if (!this.opened) return fail();
    this.ledger.start(maxTokens);
    return await this.request("stream", "terminal", { max_tokens: maxTokens }, event) as unknown as Terminal;
  }
  // A turn's abort and its owner's close may both cancel it; the worker answers one cancel per turn.
  cancel(): Promise<void> {
    const target = this.active;
    if (target === undefined) return Promise.resolve();
    if (this.cancelling?.target !== target) this.cancelling = { target, done: this.cancelTurn(target) };
    return this.cancelling.done;
  }
  private async cancelTurn(target: number): Promise<void> {
    const result = await this.request("cancel", "cancelled", { target_seq: target });
    if (canonical(result) !== canonical({ target_seq: target })) { this.abort(); fail(); }
    const pending = this.pending.get(target);
    if (pending) { clearTimeout(pending.timer); this.pending.delete(target); pending.reject(new Error("DRIFT_WORKER_CANCELLED")); }
    this.active = undefined;
  }
  // A cancelled turn's settle and the owner's teardown share one ordered close.
  close(): Promise<void> {
    if (this.dead && !this.closing) return Promise.resolve();
    return this.closing ??= this.closeOnce();
  }
  private async closeOnce(): Promise<void> {
    if (this.active !== undefined) await this.cancel();
    const result = await this.request("close", "closed", {});
    if (Object.keys(result).length) { this.abort(); fail(); }
    this.dead = true; clearTimeout(this.lifetime); this.lifecycle.abort(new Error("DRIFT_WORKER_CLOSED")); this.transport.close();
  }
}
