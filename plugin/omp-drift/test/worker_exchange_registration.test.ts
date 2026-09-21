import { expect, test } from 'bun:test';
import { chmodSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { createWorkerBoundary, workerBoundarySettings } from '../../../scripts/omp/worker_boundary_registration.mjs';
import { validateExchanged } from '../../../scripts/omp/exchange_client.mjs';
import { boundaryFixture, sha } from './provider_boundary_fixture';

const expected = (role: string) => ({ worker: role, session: role + '-fixed', model_id: role,
  route: role, exchange_session: 'exchange-1', source_worker: role, target_worker: role === 'parent' ? 'child' : 'parent', ranks: ['rank-a', 'rank-b'] });
const publication = (role: string) => ({ v: 1, session: 'exchange-1', sequence: 0, mode: 'drift',
  source_worker: role, target_worker: role === 'parent' ? 'child' : 'parent', rows: 2, source_rows: 1, copies: 2,
  bytes: 64, source_sha256: 'a'.repeat(64), text_bytes: 0, applied_ranks: ['rank-a', 'rank-b'], receipts: ['b'.repeat(64), 'c'.repeat(64)] });

test('production boundary factory exchanges through the client only while both actors are parked', async () => {
  const f = boundaryFixture();
  const calls: string[] = [];
  const abort = new AbortController(), pending: Promise<unknown>[] = [];
  const socket = f.root + '/x.sock';
  const server = createServer(peer => peer.on('data', raw => {
    expect(existsSync(f.root + '/parent-ready.json')).toBe(true);
    expect(existsSync(f.root + '/child-ready.json')).toBe(true);
    const frame = JSON.parse(String(raw)); calls.push(frame.route);
    peer.write(JSON.stringify({ op: 'exchanged', route: frame.route, published: publication(frame.route) }) + '\n');
  }));
  await new Promise<void>(resolve => server.listen(socket, resolve)); chmodSync(socket, 0o600);
  try {
    const path = f.root + '/exchange.json';
    writeFileSync(path, JSON.stringify({ v: 1, socket, timeout_ms: 500, workers: [expected('parent'), expected('child')] }), { mode: 0o600 });
    const settings = workerBoundarySettings({ DRIFT_PROVIDER_BOUNDARY_CONFIG: f.path, DRIFT_PROVIDER_BOUNDARY_SHA256: f.expected,
      DRIFT_EXCHANGE_CONFIG: path, DRIFT_EXCHANGE_SHA256: sha(readFileSync(path)) });
    const parent = createWorkerBoundary(settings, f.binding('parent'))!, child = createWorkerBoundary(settings, f.binding('child'))!;
    const call = (role: string, toolNames: string[]) => ({ identity: f.binding(role).identity, toolNames, signal: abort.signal });
    await parent(call('parent', ['task'])); expect(calls).toEqual([]);
    pending.push(child(call('child', ['hub'])).catch((e: Error) => e)); const ch = await f.ready('child');
    expect(calls).toEqual([]);
    pending.push(parent(call('parent', ['hub'])).catch((e: Error) => e)); const ph = await f.ready('parent');
    for (let i = 0; i < 100 && !['parent', 'child'].every(role => existsSync(f.root + '/' + role + '-exchange.json')); i++) await Bun.sleep(5);
    expect(calls.sort()).toEqual(['child', 'parent']);
    for (const [role, own, peer] of [['parent', ph, ch], ['child', ch, ph]]) {
      const receipt = readFileSync(f.root + '/' + role + '-exchange.json');
      expect(JSON.parse(String(receipt))).toEqual(publication(role));
      writeFileSync(f.root + '/' + role + '-release.json', JSON.stringify({ v: 1, action: 'resume', nonce: f.config.nonce,
        ready_sha256: own, peer_ready_sha256: peer, exchange_sha256: sha(receipt) }), { mode: 0o600 });
    }
    expect(await Promise.all(pending)).toEqual([undefined, undefined]);
    const source = readFileSync(new URL('../../../scripts/omp/experimental_extension.mjs', import.meta.url), 'utf8');
    expect(source).toContain('const beforeToolDispatch = createWorkerBoundary(settings,');
  } finally { abort.abort(); await Promise.all(pending); server.close(); f.close(); }
});

test('production exchange pins reject other sessions, replay and missing ranks', () => {
  for (const fields of [{ session: 'old' }, { sequence: 1 }, { source_worker: 'other' }, { applied_ranks: ['rank-a'], receipts: ['b'.repeat(64)] }]) {
    expect(() => validateExchanged({ op: 'exchanged', route: 'parent', published: { ...publication('parent'), ...fields } }, 'parent', expected('parent'), 0)).toThrow();
  }
});

test('exchange opt-in requires both hash pins and a provider parking gate', () => {
  expect(() => workerBoundarySettings({ DRIFT_EXCHANGE_CONFIG: '/tmp/x' })).toThrow('DRIFT_WORKER_CAPABILITY');
  expect(() => workerBoundarySettings({ DRIFT_EXCHANGE_CONFIG: '/tmp/x', DRIFT_EXCHANGE_SHA256: 'a'.repeat(64) })).toThrow('DRIFT_WORKER_CAPABILITY');
  expect(createWorkerBoundary(workerBoundarySettings({}), {})).toBeUndefined();
});
