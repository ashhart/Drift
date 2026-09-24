import { expect, test } from 'bun:test';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createSubagentRelease } from '../../../scripts/omp/subagent_release.mjs';
import { boundaryFixture, sha } from './provider_boundary_fixture';

function setup() {
  const f = boundaryFixture(3000);
  const config = { ...f.config, v: 2, max_epochs: 2 };
  const put = (path: string, value: unknown) => writeFileSync(path, JSON.stringify(value), { mode: 0o600 });
  put(f.path, config); const pin = sha(readFileSync(f.path));
  const entries = config.actors.map((actor: any) => ({ ...actor, exchange_session: 'exchange',
    source_worker: actor.worker, target_worker: 'peer-' + actor.worker, delivery: 'next_turn_snapshot' }));
  const exchange = f.root + '/exchange.json'; put(exchange, { v: 2, workers: entries });
  const release = createSubagentRelease(f.path, pin, exchange, sha(readFileSync(exchange)));
  const ready = (epoch = 0, mutate = (value: any) => value) => config.actors.forEach((actor: any, index: number) => {
    put(`${f.root}/${actor.role}-${epoch}-ready.json`, { v: 2, epoch, phase: 'post_generation_pre_tool', nonce: config.nonce, config_sha256: pin, ...actor });
    put(`${f.root}/${actor.role}-${epoch}-exchange.json`, mutate({ v: 1, sequence: epoch, session: 'exchange', mode: 'drift',
      text_bytes: 0, state: 'STAGED_NOT_APPLIED', source_worker: entries[index].source_worker, target_worker: entries[index].target_worker,
      rows: 1, source_rows: 1, copies: 1, bytes: 64, source_sha256: 'b'.repeat(64) }));
  });
  return { ...f, release, ready, config, pin, put };
}

test('owner releases only paired fresh stages and stops on a failed rank route', () => {
  const f = setup();
  try {
    expect(f.release.poll()).toBe('waiting'); f.ready();
    expect(f.release.poll()).toBe('released'); expect(f.release.epochs).toBe(1);
    expect(f.release.poll()).toBe('waiting');
    f.put(f.root + '/child-failed.json', { status: 'FAILED' });
    expect(() => f.release.poll()).toThrow('DRIFT_SUBAGENT_RELEASE');
    expect(existsSync(f.root + '/parent-1-release.json')).toBe(false);
  } finally { f.close(); }
});

for (const change of [{ sequence: 1 }, { text_bytes: 1 }, { source_worker: 'other' }, { rows: 0 }, { state: 'APPLIED' }]) {
  test('owner refuses mismatched staging evidence ' + JSON.stringify(change), () => {
    const f = setup();
    try {
      f.ready(0, value => ({ ...value, ...change }));
      expect(() => f.release.poll()).toThrow('DRIFT_SUBAGENT_RELEASE');
      expect(existsSync(f.root + '/parent-0-release.json')).toBe(false);
    } finally { f.close(); }
  });
}
