import { expect, test } from 'bun:test';
import { createServer } from 'node:net';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createExchangeBoundary } from '../../../scripts/omp/provider_exchange_boundary.mjs';

test('parked snapshot receiver stages now and proves application at the next boundary', async () => {
  const root = mkdtempSync(join(tmpdir(), 'staged-exchange-')), socket = join(root, 'x.sock'), operations: string[] = [];
  const expected = { exchange_session: 's', source_worker: 'qwen', target_worker: 'glm', ranks: ['rank0', 'rank1'] };
  let applied = false;
  const record = (sequence: number) => ({ v: 1, session: 's', sequence, mode: 'drift', source_worker: 'qwen', target_worker: 'glm',
    rows: 2, source_rows: 2, copies: 1, bytes: 100, source_sha256: 'a'.repeat(64), text_bytes: 0 });
  const server = createServer(peer => {
    peer.once('data', bytes => {
      const frame = JSON.parse(bytes.toString()); operations.push(frame.op);
      const reply = frame.op === 'stage' ? { op: 'staged', route: 'r', delivery: { ...record(operations.filter(op => op === 'stage').length - 1), state: 'STAGED_NOT_APPLIED' } }
        : frame.op === 'confirm' && applied ? { op: 'confirmed', route: 'r', published: { ...record(frame.sequence), applied_ranks: expected.ranks, receipts: ['b'.repeat(64), 'b'.repeat(64)] } }
        : { op: 'error', code: 'RECEIVER_PARKED' };
      peer.end(JSON.stringify(reply) + '\n');
    });
  });
  await new Promise<void>(resolve => server.listen(socket, resolve));
  try {
    const exchange = createExchangeBoundary({ socket, route: 'r', timeoutMs: 1000, expected, delivery: 'next_turn_snapshot' });
    const boundary = { signal: new AbortController().signal };
    expect((await exchange(boundary)).state).toBe('STAGED_NOT_APPLIED');
    expect(operations).toEqual(['stage']);
    applied = true;
    expect((await exchange(boundary)).sequence).toBe(1);
    expect(operations).toEqual(['stage', 'confirm', 'stage']);
    await exchange.finish(boundary);
    expect(operations).toEqual(['stage', 'confirm', 'stage', 'confirm']);
    expect(exchange.records.length).toBe(2);
    expect(exchange.poisoned).toBe(false);
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); rmSync(root, { recursive: true, force: true }); }
});

test.each(['extra_receipt', 'mismatched_confirmation', 'missing_rank', 'abort'])('staged %s never counts as a confirmed exchange', async bad => {
  const root = mkdtempSync('/tmp/dxs-'), socket = join(root, 'x.sock');
  const expected = { exchange_session: 's', source_worker: 'qwen', target_worker: 'glm', ranks: ['r0', 'r1'] };
  const record = { v: 1, session: 's', sequence: 0, mode: 'drift', source_worker: 'qwen', target_worker: 'glm',
    rows: 2, source_rows: 2, copies: 1, bytes: 100, source_sha256: 'a'.repeat(64), text_bytes: 0 };
  const server = createServer(peer => peer.once('data', bytes => {
    const frame = JSON.parse(bytes.toString());
    const delivery = { ...record, state: 'STAGED_NOT_APPLIED', ...(bad === 'extra_receipt' ? { receipts: [] } : {}) };
    const published = { ...record, source_sha256: (bad === 'mismatched_confirmation' ? 'c' : 'a').repeat(64),
      applied_ranks: bad === 'missing_rank' ? ['r0'] : expected.ranks, receipts: ['b'.repeat(64), 'b'.repeat(64)] };
    peer.end(JSON.stringify(frame.op === 'stage' ? { op: 'staged', route: 'r', delivery }
      : { op: 'confirmed', route: 'r', published }) + '\n');
  }));
  await new Promise<void>(resolve => server.listen(socket, resolve));
  try {
    const exchange = createExchangeBoundary({ socket, route: 'r', timeoutMs: 1000, expected, delivery: 'next_turn_snapshot' });
    const controller = new AbortController(), boundary = { signal: controller.signal };
    if (bad === 'extra_receipt') await expect(exchange(boundary)).rejects.toThrow('DRIFT_WORKER_CANCELLED');
    else {
      await exchange(boundary);
      if (bad === 'abort') controller.abort();
      await expect(exchange.finish(boundary)).rejects.toThrow('DRIFT_WORKER_CANCELLED');
    }
    expect(exchange.poisoned).toBe(true);
    expect(exchange.records.length).toBe(0);
    await expect(exchange(boundary)).rejects.toThrow('DRIFT_WORKER_CANCELLED');
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); rmSync(root, { recursive: true, force: true }); }
});
