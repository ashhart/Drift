import { ExecutionReceipts } from "./worker_receipts";
import { assistantSemantics } from "./worker_semantics";
import { canonical, fail, record, type Tool } from "./worker_protocol";
import type { WorkerClient } from "./worker_client";

export interface OwnContext { systemPrompt?: string | string[]; messages: Array<Record<string, unknown>>; tools?: Tool[]; }
export function textContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value) || value.some(part => !record(part) || part.type !== "text" || typeof part.text !== "string")) return fail("CAPABILITY");
  return value.map(part => part.text).join("\n");
}
function fingerprint(message: Record<string, unknown>): string {
  if (message.role === "assistant") return canonical({ role: message.role, content: assistantSemantics(message.content) });
  if (message.role === "user" || message.role === "developer") return canonical({ role: message.role, text: textContent(message.content) });
  if (message.role === "toolResult") return canonical({ role: message.role, id: message.toolCallId, text: textContent(message.content), error: message.isError === true });
  return fail("CAPABILITY");
}
export class WorkerContext {
  private prefix: string[] = [];
  private configuration?: string;
  private system?: string;
  private base?: string;
  private controls: string[] = [];
  private receipts = new ExecutionReceipts();
  private amendments: string[] = [];
  constructor(private allowSystemUpdates = false, private allowTextDuo = false) {}
  recordExecution(id: string, name: string, args: unknown): void {
    if (!this.allowTextDuo) fail("CAPABILITY");
    this.receipts.observe(id, name, args);
  }
  async ingest(client: WorkerClient, context: OwnContext): Promise<void> {
    const base = context.systemPrompt === undefined ? [] : typeof context.systemPrompt === "string" ? [context.systemPrompt] : context.systemPrompt;
    const controls = context.messages.filter(message => message.role === "developer").map(message => textContent(message.content));
    if (controls.length && !this.allowTextDuo) fail("CAPABILITY");
    const fingerprints = context.messages.map(fingerprint);
    const changes = this.allowTextDuo ? this.receipts.reconcile(this.prefix, fingerprints) : [];
    if (!this.allowTextDuo && this.prefix.some((value, index) => fingerprints[index] !== value)) fail("CONTEXT");
    const system = [...base, ...controls, ...this.amendments, ...changes];
    const tools = (context.tools ?? []).map(tool => ({ name: tool.name, description: tool.description, parameters: JSON.parse(JSON.stringify(tool.parameters)) }));
    const config = canonical(tools);
    const systemKey = canonical(system), baseKey = canonical(base);
    if (this.system !== undefined && this.system !== systemKey && !this.allowSystemUpdates) fail("CAPABILITY");
    if (this.configuration !== undefined && config !== this.configuration) fail("CAPABILITY");
    const fresh = context.messages.slice(this.prefix.length);
    if (!fresh.length || fresh.some(message => !["user", "toolResult", "developer"].includes(String(message.role)))) fail("CONTEXT");
    if (this.configuration === undefined) { await client.open(system, tools); this.configuration = config; }
    else if (this.system !== systemKey) {
      if (this.base === baseKey && this.controls.every((value, index) => controls[index] === value)) {
        const additions = [...controls.slice(this.controls.length), ...changes];
        if (additions.length) await client.ownControlAppend(additions);
      } else await client.ownControl(system);
    }
    for (const message of fresh) {
      if (message.role === "developer") continue;
      if (message.role === "user") await client.ownPrompt(textContent(message.content));
      else {
        if (typeof message.toolCallId !== "string") fail("CONTEXT");
        await client.toolResult(message.toolCallId, textContent(message.content), message.isError === true);
      }
    }
    this.system = systemKey; this.base = baseKey; this.controls = [...controls];
    this.prefix = fingerprints; this.amendments.push(...changes); this.receipts.settle();
  }
  remember(message: Record<string, unknown>): void { this.prefix.push(fingerprint(message)); this.receipts.expect(message); }
}
