import { exchangeRequest, validateExchanged } from './exchange_client.mjs';

const fail = () => { throw new Error('DRIFT_WORKER_CANCELLED'); };
const sha = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const fields = ['v','session','sequence','mode','source_worker','target_worker','rows','source_rows','copies','bytes','source_sha256','text_bytes','state'];

export function createStagedExchange({ socket, route, timeoutMs, expected }) {
  if (!expected || !Array.isArray(expected.ranks) || !expected.ranks.length) fail();
  let poisoned = false, pending, busy = false;
  const records = [];
  const request = (frame, boundary) => {
    if (poisoned || !(boundary?.signal instanceof AbortSignal) || boundary.signal.aborted) fail();
    return exchangeRequest(socket, { route, ...frame }, { signal: boundary.signal, timeoutMs });
  };
  const confirm = async boundary => {
    if (!pending) return;
    const reply = await request({ op: 'confirm', sequence: pending.sequence }, boundary);
    if (reply.op !== 'confirmed' || reply.route !== route) fail();
    const record = validateExchanged({ ...reply, op: 'exchanged' }, route, expected, records.length);
    for (const key of fields.filter(key => key !== 'state')) if (record[key] !== pending[key]) fail();
    if (boundary.signal.aborted) fail();
    records.push(record); pending = undefined;
  };
  const guarded = async operation => {
    if (poisoned || busy) { poisoned = true; fail(); }
    busy = true;
    try { return await operation(); } catch { poisoned = true; fail(); } finally { busy = false; }
  };
  const run = boundary => guarded(async () => {
    await confirm(boundary);
    const reply = await request({ op: 'stage' }, boundary), value = reply.delivery;
    if (reply.op !== 'staged' || reply.route !== route || !value || typeof value !== 'object'
        || Object.keys(value).sort().join() !== [...fields].sort().join()
        || value.state !== 'STAGED_NOT_APPLIED' || value.v !== 1 || value.mode !== 'drift' || value.text_bytes !== 0
        || value.session !== expected.exchange_session || value.sequence !== records.length
        || value.source_worker !== expected.source_worker || value.target_worker !== expected.target_worker
        || !Number.isSafeInteger(value.rows) || value.rows <= 0 || value.rows > 1000000
        || !Number.isSafeInteger(value.source_rows) || value.source_rows <= 0
        || !Number.isSafeInteger(value.copies) || value.copies < 1 || value.copies > 64
        || value.rows !== value.source_rows * value.copies || !sha(value.source_sha256)
        || !Number.isSafeInteger(value.bytes) || value.bytes <= 0 || value.bytes > 67108864) fail();
    pending = structuredClone(value);
    if (boundary.signal.aborted) fail();
    return structuredClone(pending);
  });
  Object.defineProperty(run, 'poisoned', { get: () => poisoned });
  return Object.assign(run, { records, finish: boundary => guarded(() => confirm(boundary)) });
}
