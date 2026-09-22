import { expect, test } from 'bun:test';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createProviderProjectBoundary } from '../../../scripts/omp/provider_project_boundary.mjs';
import { boundaryFixture, sha } from './provider_boundary_fixture';

test('production provider hooks exchange twice then cancel both actors in the third epoch', async () => {
  const f = boundaryFixture(3000), signal = new AbortController(), pending: Promise<unknown>[] = [];
  const calls: string[] = [];
  const file = (role: string, epoch: number, kind: string) => `${f.root}/${role}-${epoch}-${kind}.json`;
  async function waitFile(role: string, epoch: number, kind: string) {
    const path = file(role, epoch, kind);
    for (let i = 0; i < 200 && !existsSync(path); i++) await Bun.sleep(5);
    expect(existsSync(path)).toBe(true);
    return sha(readFileSync(path));
  }
  try {
    writeFileSync(f.path, JSON.stringify({ ...f.config, v: 2, max_epochs: 3 }), { mode: 0o600 });
    const expected = sha(readFileSync(f.path));
    const hooks = ['parent', 'child'].map(role => createProviderProjectBoundary(f.path, expected, f.binding(role), {
      exchange: async () => { calls.push(role); return { role, sequence: calls.length }; },
    }));
    const boundary = (role: string, tool: string) => ({ identity: f.binding(role).identity, toolNames: [tool], signal: signal.signal });
    await hooks[0](boundary('parent', 'task'));
    for (let epoch = 0; epoch < 3; epoch++) {
      const start = pending.length;
      for (const [index, role] of ['parent', 'child'].entries()) {
        pending.push(hooks[index](boundary(role, epoch === 1 && role === 'parent' ? 'task' : 'hub')).catch((error: Error) => error));
      }
      const ready = await Promise.all(['parent', 'child'].map(role => waitFile(role, epoch, 'ready')));
      if (epoch === 2) {
        signal.abort();
        const results = await Promise.all(pending.slice(start));
        expect(results.every(value => value instanceof Error && value.message === 'DRIFT_WORKER_CANCELLED')).toBe(true);
        for (const role of ['parent', 'child']) {
          expect(existsSync(file(role, epoch, 'ready'))).toBe(false);
          expect(existsSync(file(role, epoch, 'invalidated'))).toBe(true);
        }
      } else {
        for (const [index, role] of ['parent', 'child'].entries()) {
          const exchange = await waitFile(role, epoch, 'exchange');
          writeFileSync(file(role, epoch, 'release'), JSON.stringify({ v: 2, epoch, action: 'resume', nonce: f.config.nonce,
            ready_sha256: ready[index], peer_ready_sha256: ready[1-index], exchange_sha256: exchange }), { mode: 0o600 });
        }
        expect(await Promise.all(pending.slice(start))).toEqual([undefined, undefined]);
        expect(calls.filter(role => role === 'parent').length).toBe(epoch + 1);
        expect(calls.filter(role => role === 'child').length).toBe(epoch + 1);
      }
    }
  } finally { signal.abort(); await Promise.all(pending); f.close(); }
});
