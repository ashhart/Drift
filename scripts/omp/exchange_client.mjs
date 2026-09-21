import { createConnection } from 'node:net';

const LIMIT = 65536;

export class ExchangeClientError extends Error {}

const fail = code => { throw new ExchangeClientError(code); };

export function exchangeRequest(path, frame, { signal, timeoutMs = 30000 } = {}) {
  // One bounded request per call: the coordinator owns the cursors, this side owns nothing.
  return new Promise((resolve, reject) => {
    let settled = false, raw = '';
    const socket = createConnection({ path });
    const done = (error, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', onAbort);
      socket.destroy();
      error ? reject(error) : resolve(value);
    };
    const onAbort = () => done(new ExchangeClientError('EXCHANGE_CANCELLED'));
    const timer = setTimeout(() => done(new ExchangeClientError('EXCHANGE_TIMEOUT')), timeoutMs);
    if (signal) {
      if (signal.aborted) return done(new ExchangeClientError('EXCHANGE_CANCELLED'));
      signal.addEventListener('abort', onAbort, { once: true });
    }
    socket.on('error', () => done(new ExchangeClientError('EXCHANGE_TRANSPORT')));
    socket.on('close', () => done(new ExchangeClientError('EXCHANGE_EOF')));
    socket.on('connect', () => socket.write(JSON.stringify(frame) + '\n'));
    socket.on('data', chunk => {
      raw += chunk;
      if (raw.length > LIMIT) return done(new ExchangeClientError('EXCHANGE_LIMIT'));
      const end = raw.indexOf('\n');
      if (end < 0) return;
      let reply;
      try { reply = JSON.parse(raw.slice(0, end)); } catch { return done(new ExchangeClientError('EXCHANGE_REPLY')); }
      if (raw.indexOf('\n', end + 1) >= 0) return done(new ExchangeClientError('EXCHANGE_REPLY'));
      done(null, reply);
    });
  });
}

export function validateExchanged(reply, route, expected, sequence) {
  // A reply is evidence only when it is the exchange this boundary asked for and every rank applied it.
  if (!reply || typeof reply !== 'object' || Array.isArray(reply)) fail('EXCHANGE_REPLY');
  if (reply.op === 'error') fail(typeof reply.code === 'string' ? reply.code : 'EXCHANGE_FAILED');
  if (reply.op !== 'exchanged' || reply.route !== route) fail('EXCHANGE_REPLY');
  const published = reply.published;
  if (!published || typeof published !== 'object') fail('EXCHANGE_REPLY');
  if (published.mode !== 'drift') fail('EXCHANGE_MODE');
  if (published.text_bytes !== 0) fail('EXCHANGE_TEXT_FALLBACK');
  if (!Number.isSafeInteger(published.rows) || published.rows <= 0) fail('EXCHANGE_ROWS');
  if (!Array.isArray(published.applied_ranks) || !published.applied_ranks.length) fail('EXCHANGE_RANKS');
  if (!Array.isArray(published.receipts) || published.receipts.length !== published.applied_ranks.length) fail('EXCHANGE_RECEIPTS');
  if (!published.receipts.every(value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value))) fail('EXCHANGE_RECEIPTS');
  if (expected) {
    if (published.v !== 1 || published.session !== expected.exchange_session || published.sequence !== sequence
        || published.source_worker !== expected.source_worker || published.target_worker !== expected.target_worker) fail('EXCHANGE_BINDING');
    if (JSON.stringify([...published.applied_ranks].sort()) !== JSON.stringify([...expected.ranks].sort())) fail('EXCHANGE_RANKS');
    if (!Number.isSafeInteger(published.source_rows) || published.source_rows <= 0
        || !Number.isSafeInteger(published.copies) || published.copies <= 0 || published.copies > 64
        || published.rows !== published.source_rows * published.copies) fail('EXCHANGE_ROWS');
    if (!Number.isSafeInteger(published.bytes) || published.bytes <= 0 || published.bytes > 67108864
        || typeof published.source_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(published.source_sha256)) fail('EXCHANGE_REPLY');
  }
  return published;
}
