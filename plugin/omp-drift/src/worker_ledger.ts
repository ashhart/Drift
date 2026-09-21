import { fail, integer, record, type Limits, type StreamEvent, type Terminal, type Tool } from "./worker_protocol";

export class WorkerLedger {
  tokensUsed = 0;
  turns = 0;
  readonly pending = new Set<string>();
  private seen = new Set<string>();
  private tools = new Set<string>();
  private bytes = 0;
  private events = 0;
  private maxTokens = 0;
  constructor(private readonly limits: Limits) {}
  setTools(tools: Tool[]): void {
    for (const tool of tools) {
      if (!/^[A-Za-z0-9_.-]{1,96}$/.test(tool.name) || this.tools.has(tool.name) || !record(tool.parameters)) fail();
      this.tools.add(tool.name);
    }
  }
  start(maxTokens: number): void {
    if (this.pending.size || !integer(maxTokens) || maxTokens < 1 || maxTokens > this.limits.max_output_tokens) fail("LIMIT");
    if (++this.turns > this.limits.max_turns || this.tokensUsed + maxTokens > this.limits.max_session_tokens) fail("LIMIT");
    this.bytes = 0; this.events = 0; this.maxTokens = maxTokens;
  }
  event(op: string, payload: Record<string, unknown>): StreamEvent {
    this.bytes += Buffer.byteLength(JSON.stringify(payload));
    if (++this.events > this.maxTokens * 8 + 32 || this.bytes > this.limits.max_input_bytes) fail("LIMIT");
    if (op === "text" && typeof payload.text === "string" && payload.text.length > 0 && Object.keys(payload).length === 1) return { op, payload: { text: payload.text } };
    if (op !== "tool_call" || Object.keys(payload).sort().join() !== "arguments,call_id,name") return fail();
    const { call_id: id, name, arguments: args } = payload;
    if (typeof id !== "string" || !/^[A-Za-z0-9_.:-]{1,128}$/.test(id) || this.seen.has(id) || typeof name !== "string" || !this.tools.has(name) || !record(args)) return fail();
    this.pending.add(id); this.seen.add(id);
    return { op, payload: { call_id: id, name, arguments: args } };
  }
  terminal(payload: Record<string, unknown>): Terminal {
    if (Object.keys(payload).sort().join() !== "reason,usage" || !record(payload.usage)) return fail();
    const { input_tokens: input, output_tokens: output } = payload.usage;
    if (Object.keys(payload.usage).sort().join() !== "input_tokens,output_tokens" || !integer(input) || !integer(output) || output > this.maxTokens) return fail("LIMIT");
    const reason = payload.reason;
    if (!["stop", "tool_use", "length"].includes(String(reason)) || (reason === "tool_use") !== (this.pending.size > 0)) return fail();
    this.tokensUsed += input + output;
    if (this.tokensUsed > this.limits.max_session_tokens) return fail("LIMIT");
    return { reason: reason as Terminal["reason"], usage: { input_tokens: input, output_tokens: output } };
  }
}
