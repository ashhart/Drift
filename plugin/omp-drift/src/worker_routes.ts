import { validateChannels } from "./worker_channels";
import { createHash } from "node:crypto";
import { fail, type Identity } from "./worker_protocol";
import type { WorkerEntry } from "./worker_registration";
export class WorkerRoutes {
  private bound = new Map<string, string>();
  bind(entry: WorkerEntry, ompSession: unknown): { key: string; identity: Identity } {
    validateChannels(entry);
    if (!entry.experimental_multi_turn) return { key: entry.identity.model_id, identity: entry.identity };
    if (typeof ompSession !== "string" || !ompSession || ompSession.length > 256) fail("SESSION");
    const worker = entry.identity.worker;
    const existing = this.bound.get(worker);
    if (existing !== undefined && existing !== ompSession) fail("SESSION_LIMIT");
    this.bound.set(worker, ompSession);
    if (entry.session_binding === "fixed") return { key: worker + ":fixed:" + entry.identity.session, identity: { ...entry.identity } };
    const digest = createHash("sha256").update(entry.identity.session + "\0" + ompSession).digest("hex").slice(0, 32);
    return { key: worker + ":" + digest, identity: { ...entry.identity, session: entry.identity.session.slice(0, 24) + "-" + digest } };
  }
}
