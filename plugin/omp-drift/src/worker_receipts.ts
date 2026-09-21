import { canonical, fail, record } from "./worker_protocol";
export class ExecutionReceipts {
  private calls = new Map<string, { name: string; arguments: Record<string, unknown> }>();
  private observed = new Map<string, Record<string, unknown>>();
  expect(message: Record<string, any>): void {
    for (const part of message.content) if (part.type === "toolCall") this.calls.set(part.id, structuredClone({ name: part.name, arguments: part.arguments }));
  }
  observe(id: string, name: string, args: unknown): void {
    const call = this.calls.get(id);
    if (!call || call.name !== name || !record(args)) return fail("CONTEXT");
    const previous = this.observed.get(id);
    if (previous && canonical(previous) !== canonical(args)) return fail("CONTEXT");
    this.observed.set(id, structuredClone(args));
  }
  reconcile(before: string[], after: string[]): string[] {
    for (let index = 0; index < before.length; index++) {
      if (before[index] === after[index]) continue;
      const message = JSON.parse(before[index]);
      if (message.role !== "assistant") fail("CONTEXT");
      const updated = JSON.parse(after[index]);
      if (updated.role !== "assistant" || !Array.isArray(updated.content) || updated.content.length !== message.content.length) fail("CONTEXT");
      for (const [position, part] of message.content.entries()) {
        if (canonical(part) === canonical(updated.content[position])) continue;
        if (part.type === "toolCall" && this.observed.has(part.id)) part.arguments = this.observed.get(part.id);
      }
      if (canonical(message) !== after[index]) fail("CONTEXT");
    }
    return [...this.observed].filter(([id, args]) => canonical(this.calls.get(id)!.arguments) !== canonical(args))
      .map(([id, args]) => "Controller-observed tool execution amendment: " + canonical({ call_id: id, name: this.calls.get(id)!.name, arguments: args }));
  }
  settle(): void { for (const id of this.observed.keys()) this.calls.delete(id); this.observed.clear(); }
}
