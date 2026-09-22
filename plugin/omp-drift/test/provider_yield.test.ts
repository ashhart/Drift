import { expect, test } from 'bun:test';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createProviderProjectBoundary } from '../../../scripts/omp/provider_project_boundary.mjs';
import { boundaryFixture, sha } from './provider_boundary_fixture';
import { terminalAction } from '../../../scripts/omp/project_boundary_terminal.mjs';

function fixture() {
  const f = boundaryFixture();
  writeFileSync(f.path, JSON.stringify({ ...f.config, v: 2, max_epochs: 1 }), { mode: 0o600 });
  const expected = sha(readFileSync(f.path)), confirmed: string[] = [];
  const hooks = Object.fromEntries(['parent', 'child'].map(role => [role,
    createProviderProjectBoundary(f.path, expected, f.binding(role), { exchange: Object.assign(
      async () => ({ role }), { finish: async () => { confirmed.push(role); } }) })]));
  const call = (role: string, toolNames: string[], toolCalls: unknown[] = []) => ({
    identity: f.binding(role).identity, toolNames, toolCalls, signal: new AbortController().signal });
  const file = (role: string, kind: string) => `${f.root}/${role}-0-${kind}.json`;
  async function epoch() {
    await hooks.parent(call('parent', ['task']));
    const pending = ['parent', 'child'].map(role => hooks[role](call(role, ['drift_tick'])));
    for (let i = 0; i < 100 && !['parent', 'child'].every(role => existsSync(file(role, 'exchange'))); i++) await Bun.sleep(5);
    for (const [role, peer] of [['parent', 'child'], ['child', 'parent']]) {
      writeFileSync(file(role, 'release'), JSON.stringify({ v: 2, epoch: 0, action: 'resume', nonce: f.config.nonce,
        ready_sha256: sha(readFileSync(file(role, 'ready'))), peer_ready_sha256: sha(readFileSync(file(peer, 'ready'))),
        exchange_sha256: sha(readFileSync(file(role, 'exchange'))) }), { mode: 0o600 });
    }
    await Promise.all(pending);
  }
  return { ...f, hooks, call, epoch, confirmed };
}

test('child text-only stop is not terminal and a final yield settles both native owners', async () => {
  const f = fixture();
  try {
    await f.epoch();
    await f.hooks.child(f.call('child', []));
    expect(existsSync(`${f.root}/child-finished.json`)).toBe(false);
    const parent = f.hooks.parent(f.call('parent', []));
    const child = f.hooks.child(f.call('child', ['yield'], [{ name: 'yield', arguments: { result: { data: { complete: true } } } }]));
    await Promise.all([parent, child]);
    expect(f.confirmed.sort()).toEqual(['child', 'parent']);
    expect(existsSync(`${f.root}/child-settled.json`)).toBe(true);
    await f.hooks.parent(f.call('parent', []));
    expect(f.confirmed.sort()).toEqual(['child', 'parent']);
    expect(existsSync(`${f.root}/parent-failed.json`)).toBe(false);
    writeFileSync(`${f.root}/child-settled.json`, '{}', { mode: 0o600 });
    await expect(f.hooks.parent(f.call('parent', []))).rejects.toThrow('DRIFT_WORKER_CANCELLED');
    expect(f.confirmed.sort()).toEqual(['child', 'parent']);
  } finally { f.close(); }
});

test('terminal yield cannot be inferred from its name without validated arguments', async () => {
  const f = fixture();
  try {
    await f.epoch();
    await expect(f.hooks.child(f.call('child', ['yield']))).rejects.toThrow('DRIFT_WORKER_CANCELLED');
    expect(f.confirmed).toEqual([]);
    expect(existsSync(`${f.root}/child-finished.json`)).toBe(false);
  } finally { f.close(); }
});

test('yield types distinguish incremental checkpoints from explicit terminal results', () => {
  const boundary = (args: unknown) => ({ toolNames: ['yield'], toolCalls: [{ name: 'yield', arguments: args }] });
  for (const type of [undefined, null, 'complete']) {
    expect(terminalAction(2, 'child', boundary({ type, result: { data: {} } }))).toBe('finish');
  }
  expect(terminalAction(2, 'child', boundary({ type: ['progress'], result: { data: {} } }))).toBe('exchange');
  expect(terminalAction(2, 'child', boundary({ data: { complete: true } }))).toBe('finish');
  expect(terminalAction(2, 'child', boundary({ result: '{"data":{"complete":true}}' }))).toBe('finish');
  for (const args of [{ type: [], result: { data: {} } }, { type: 1, result: { data: {} } },
    { result: { error: 'failure' } }, { result: { data: {}, error: 'failure' } }, { result: '{}' }]) {
    expect(() => terminalAction(2, 'child', boundary(args))).toThrow();
  }
  expect(() => terminalAction(2, 'parent', boundary({ result: { data: {} } }))).toThrow();
  expect(() => terminalAction(2, 'child', { ...boundary({ result: { data: {} } }), toolNames: ['yield', 'task'] })).toThrow();
});

test('abort while final yield is parked poisons both actors without confirming either', async () => {
  const f = fixture(), controller = new AbortController();
  try {
    await f.epoch();
    const pending = f.hooks.child({ ...f.call('child', ['yield'], [{ name: 'yield', arguments: { result: { data: {} } } }]), signal: controller.signal }).catch((error: Error) => error);
    for (let i = 0; i < 100 && !existsSync(`${f.root}/child-finished.json`); i++) await Bun.sleep(2);
    controller.abort();
    expect((await pending).message).toBe('DRIFT_WORKER_CANCELLED');
    await expect(f.hooks.parent(f.call('parent', []))).rejects.toThrow('DRIFT_WORKER_CANCELLED');
    expect(f.confirmed).toEqual([]);
    expect(existsSync(`${f.root}/child-settled.json`)).toBe(false);
  } finally { f.close(); }
});
