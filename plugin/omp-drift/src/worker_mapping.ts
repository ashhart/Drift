import type { StreamEvent, Terminal } from "./worker_protocol";
export interface EventSink { push(event: Record<string, unknown>): void; }
export class WorkerMapping {
  readonly message: Record<string, any>;
  private textIndex?: number;
  constructor(private sink: EventSink, model: { id: string; api: string; provider: string }) {
    this.message = { role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "stop", timestamp: Date.now() };
    sink.push({ type: "start", partial: this.message });
  }
  event(event: StreamEvent): void {
    if (event.op === "text") {
      if (this.textIndex === undefined) {
        this.textIndex = this.message.content.length;
        this.message.content.push({ type: "text", text: "" });
        this.sink.push({ type: "text_start", contentIndex: this.textIndex, partial: this.message });
      }
      this.message.content[this.textIndex!].text += event.payload.text;
      this.sink.push({ type: "text_delta", contentIndex: this.textIndex, delta: event.payload.text, partial: this.message });
    } else {
      this.finishText();
      const contentIndex = this.message.content.length;
      const part = { type: "toolCall", id: event.payload.call_id, name: event.payload.name, arguments: event.payload.arguments };
      this.message.content.push(part);
      this.sink.push({ type: "toolcall_start", contentIndex, partial: this.message });
      this.sink.push({ type: "toolcall_end", contentIndex, toolCall: part, partial: this.message });
    }
  }
  private finishText(): void {
    if (this.textIndex === undefined) return;
    this.sink.push({ type: "text_end", contentIndex: this.textIndex, content: this.message.content[this.textIndex].text, partial: this.message });
    this.textIndex = undefined;
  }
  done(terminal: Terminal): void {
    this.finishText();
    this.message.stopReason = terminal.reason === "tool_use" ? "toolUse" : terminal.reason;
    Object.assign(this.message.usage, { input: terminal.usage.input_tokens, output: terminal.usage.output_tokens, totalTokens: terminal.usage.input_tokens + terminal.usage.output_tokens });
    this.sink.push({ type: "done", reason: this.message.stopReason, message: this.message });
  }
  error(aborted: boolean, code = "UNKNOWN", phase = "unknown"): void {
    const allowed = /^(REMOTE_)?(PROTOCOL|LIMIT|CAPABILITY|WORKER)$|^(CONTEXT|SESSION|SESSION_LIMIT|TIMEOUT|CANCELLED|CLOSED|TRANSPORT|UNKNOWN)$/;
    const diagnostic = allowed.test(code) ? code : "UNKNOWN";
    const step = ["context", "stream", "close", "unknown"].includes(phase) ? phase : "unknown";
    this.message.stopReason = aborted ? "aborted" : "error";
    this.message.errorMessage = `${aborted ? "Private worker cancelled" : "Private worker session failed"} [${step}:${diagnostic}]`;
    this.sink.push({ type: "error", reason: this.message.stopReason, error: this.message });
  }
}
