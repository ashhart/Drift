import { fail, integer, record, type Identity } from "./worker_protocol";

export interface ActivationWake { activation_seq: number; foreign_total: number; }

export function activationWakeBinding(receipt: unknown, identity: Identity): ActivationWake {
  if (!record(receipt) || Object.keys(receipt).sort().join() !== "foreign_total,op,rows,seq,session,sha256,source_worker,target_worker,v") return fail();
  if (receipt.v !== 1 || receipt.op !== "appended" || receipt.session !== identity.session || receipt.target_worker !== identity.worker) return fail();
  if (typeof receipt.source_worker !== "string" || !/^[A-Za-z0-9_.-]{1,96}$/.test(receipt.source_worker)
      || receipt.source_worker === receipt.target_worker) return fail();
  if (typeof receipt.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.sha256)) return fail();
  if (!integer(receipt.seq) || receipt.seq < 1 || !integer(receipt.rows) || receipt.rows < 1 || receipt.rows > 4096
      || !integer(receipt.foreign_total) || receipt.foreign_total < receipt.rows || receipt.foreign_total > 1_000_000) return fail();
  return { activation_seq: receipt.seq, foreign_total: receipt.foreign_total };
}
